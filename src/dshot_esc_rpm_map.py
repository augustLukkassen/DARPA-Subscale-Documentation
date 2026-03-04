#!/usr/bin/env python3
"""
DShot ↔ ESC Speed Mapping Script
=================================
Reads a rosbag with telemetry containing esc_rpm[4], converts to rad/s,
segments the data into discrete DShot steps, and fits a 2nd-order
polynomial:  DShot = f(ω / v_bat).

The normalized variable  x = ω / v_bat  matches the C++ function:
    int dshot_value_from_speed(const float speed) {
        float x = omega / v_bat;
        return cap_dshot(int(p1 * x * x + p2 * x + p3));
    }

Usage:
    python3 dshot_esc_rpm_map.py

Configure the parameters in the section below before running.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────── USER CONFIGURATION ────────────────────────────
# Path to the rosbag file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs_subscale/
BAG_PATH = os.path.join(BASE_DIR, "ros_bags", "dshot200_1600_1420kv.bag")

# Telemetry topic that contains esc_rpm
TELEMETRY_TOPIC = "/telemetry117"

# DShot linspace parameters  (start, stop, num_steps)
# These must correspond 1-to-1 with the discrete steps in the bag, in order.
DSHOT_START = 200
DSHOT_END   = 1600
DSHOT_NUM   = 8

# Conversion: ESC reports RPM, we convert to rad/s
# rad/s = RPM * 2π / 60
RPM_TO_RADS = 2.0 * np.pi / 60.0

# Segmentation parameters
# Speed values (rad/s) below this are considered idle (before/after the staircase).
SPEED_IDLE_THRESHOLD = 200 * RPM_TO_RADS  # ~20.9 rad/s

# Smoothing window (number of samples) applied before computing the
# derivative used to detect step transitions.  Increase if noisy.
SMOOTH_WINDOW = 10

# When computing the mean RPM for a plateau, discard the first and last
# TRIM_FRACTION of samples to avoid transients at step edges.
TRIM_FRACTION = 0.30
# ────────────────────────────────────────────────────────────────────────────


def read_esc_rpm(bag_path: str, topic: str):
    """Read all esc_rpm messages and return (timestamps, speed_array, voltage_array).

    Returns
    -------
    times : np.ndarray, shape (N,)
        Time in seconds (relative to bag start).
    rads  : np.ndarray, shape (N, 4)
        ESC speed in rad/s for each of the 4 motors.
    volts : np.ndarray, shape (N,)
        Battery voltage per message.
    """
    import rosbag

    bag = rosbag.Bag(bag_path, "r")
    times, rpms, volts = [], [], []
    t0 = None
    for _, msg, t in bag.read_messages(topics=[topic]):
        ts = t.to_sec()
        if t0 is None:
            t0 = ts
        times.append(ts - t0)
        rpms.append(list(msg.esc_rpm))
        # Read battery voltage (try both possible field names)
        v = getattr(msg, 'batteryVoltage', None)
        if v is None or v == 0:
            v = getattr(msg, 'voltage', 0.0)
        volts.append(float(v))
    bag.close()

    times = np.array(times)
    rpms = np.array(rpms, dtype=float)
    # Convert RPM → rad/s
    rads = rpms * RPM_TO_RADS
    return times, rads, np.array(volts, dtype=float)


def segment_plateaus(times, avg_speed, n_steps, speed_idle_threshold, smooth_window):
    """Segment a monotonically-increasing staircase into N equal time slices.

    The approach is simple and robust:
    1. Crop to the active region (speed > idle threshold).
    2. Divide that time window into N equal-duration slices.
    3. Return the index ranges for each slice.

    Returns a list of (start_idx, end_idx) using indices into the
    *original* (full-length) arrays, in time order (lowest → highest speed).
    """
    # ── 1. Active region ────────────────────────────────────────────────
    active = avg_speed > speed_idle_threshold
    idxs = np.where(active)[0]
    if len(idxs) == 0:
        return []
    act_start, act_end = idxs[0], idxs[-1]

    # ── 2. Divide into N equal time slices ──────────────────────────────
    t_start = times[act_start]
    t_end = times[act_end]
    slice_edges = np.linspace(t_start, t_end, n_steps + 1)

    plateaus = []
    for i in range(n_steps):
        # Find all indices within this time slice
        mask = (times >= slice_edges[i]) & (times < slice_edges[i + 1])
        if i == n_steps - 1:
            # Include the very last sample in the last slice
            mask = (times >= slice_edges[i]) & (times <= slice_edges[i + 1])
        slice_idxs = np.where(mask)[0]
        if len(slice_idxs) > 0:
            plateaus.append((slice_idxs[0], slice_idxs[-1]))

    return plateaus


def trim_plateau_mean(data, start, end, trim_fraction):
    """Return the trimmed-mean of data[start:end+1]."""
    n = end - start + 1
    trim = int(n * trim_fraction)
    if trim < 1:
        trim = 0
    segment = data[start + trim : end - trim + 1]
    if len(segment) == 0:
        segment = data[start:end + 1]
    return np.mean(segment)


def main():
    # ── 1. Read data ────────────────────────────────────────────────────────
    print(f"Reading bag: {os.path.abspath(BAG_PATH)}")
    times, rads, volts = read_esc_rpm(BAG_PATH, TELEMETRY_TOPIC)
    print(f"  Total messages: {len(times)}")
    print(f"  Duration: {times[-1]:.1f} s")
    print(f"  Voltage range: {volts[volts > 0].min():.2f} – {volts.max():.2f} V")

    # Average across 4 motors
    avg_speed = rads.mean(axis=1)

    # ── 2. Segment into plateaus ────────────────────────────────────────────
    plateaus = segment_plateaus(
        times, avg_speed, DSHOT_NUM, SPEED_IDLE_THRESHOLD, SMOOTH_WINDOW
    )
    print(f"  Detected {len(plateaus)} plateaus")

    # ── 3. DShot linspace ───────────────────────────────────────────────────
    dshot_values = np.linspace(DSHOT_START, DSHOT_END, DSHOT_NUM)
    print(f"  DShot values ({DSHOT_NUM} steps): {dshot_values}")

    if len(plateaus) != len(dshot_values):
        print(
            f"\n⚠  WARNING: detected {len(plateaus)} plateaus but expected "
            f"{DSHOT_NUM} DShot steps.\n"
            f"  Adjust SPEED_IDLE_THRESHOLD or "
            f"SMOOTH_WINDOW ({SMOOTH_WINDOW}) until they match.\n"
            f"  Plateau time ranges:"
        )
        for i, (s, e) in enumerate(plateaus):
            print(f"    [{i}] t = {times[s]:.2f} – {times[e]:.2f} s, "
                  f"mean ω ≈ {avg_speed[s:e+1].mean():.1f} rad/s")
        print("\nProceeding with the minimum of the two for the fit.\n")

    n_fit = min(len(plateaus), len(dshot_values))

    # ── 4. Compute mean speed and voltage per plateau, then normalize ───────
    speed_means = np.array([
        trim_plateau_mean(avg_speed, s, e, TRIM_FRACTION)
        for s, e in plateaus[:n_fit]
    ])
    volt_means = np.array([
        trim_plateau_mean(volts, s, e, TRIM_FRACTION)
        for s, e in plateaus[:n_fit]
    ])
    # x = ω / v_bat  (matches C++ function)
    x_means = speed_means / volt_means
    dshot_fit = dshot_values[:n_fit]

    print("\n  Plateau summary:")
    print(f"  {'Step':>4}  {'DShot':>8}  {'rad/s':>10}  {'V_bat':>8}  {'x=ω/V':>10}  {'t_start':>8}  {'t_end':>8}")
    for i in range(n_fit):
        s, e = plateaus[i]
        print(f"  {i:4d}  {dshot_fit[i]:8.1f}  {speed_means[i]:10.1f}  "
              f"{volt_means[i]:8.2f}  {x_means[i]:10.4f}  "
              f"{times[s]:8.2f}  {times[e]:8.2f}")

    # ── 5. Fit 2nd-order polynomial: DShot = f(x) where x = ω / v_bat ──────
    # Polynomial: DShot = p1 * x² + p2 * x + p3
    dshot_shifted = dshot_fit - 48.0
    A = np.column_stack([x_means**2, x_means])
    coeffs_2, _, _, _ = np.linalg.lstsq(A, dshot_shifted, rcond=None)
    p1, p2 = coeffs_2
    p3 = 48.0
    poly = lambda x: p1 * x**2 + p2 * x + 48.0

    print(f"\n  2nd-order polynomial fit  DShot = f(ω / v_bat):")
    print(f"    DShot = {p1:.4f} * x² + {p2:.4f} * x + {p3:.4f}")
    print(f"    where x = ω / v_bat")
    print(f"\n  C++ snippet:")
    print(f"    float p1 = {p1:.4f}f;")
    print(f"    float p2 = {p2:.4f}f;")
    print(f"    float x = omega / v_bat;")
    print(f"    return cap_dshot(int(p1 * x * x + p2 * x + {p3:.1f}f));")

    # R² goodness of fit
    ss_res = np.sum((dshot_fit - poly(x_means)) ** 2)
    ss_tot = np.sum((dshot_fit - np.mean(dshot_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"    R² = {r_squared:.6f}")

    # ── 6. Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw speed time series with plateau regions highlighted
    ax = axes[0]
    ax.plot(times, avg_speed, linewidth=0.5, label="Avg ESC speed")
    colors = plt.cm.viridis(np.linspace(0, 1, n_fit))
    for i in range(n_fit):
        s, e = plateaus[i]
        ax.axvspan(times[s], times[e], alpha=0.2, color=colors[i],
                   label=f"Step {i} (DShot={dshot_fit[i]:.0f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESC speed [rad/s] (avg of 4 motors)")
    ax.set_title("Raw ESC speed with detected plateaus")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: polynomial fit on normalized variable x = ω / v_bat
    ax = axes[1]
    x_plot = np.linspace(0, x_means.max() * 1.1, 300)
    ax.scatter(x_means, dshot_fit, c="red", zorder=5, label="Data points")
    ax.plot(x_plot, poly(x_plot), "b-", label=(
        f"Fit: DShot = {p1:.2f}·x² + {p2:.2f}·x + {p3:.1f}\n"
        f"x = ω / v_bat,  R² = {r_squared:.4f}"
    ))
    ax.set_xlabel("x = ω / v_bat  [rad/s / V]")
    ax.set_ylabel("DShot command value")
    ax.set_title("DShot = f(ω / v_bat)  —  2nd-order polynomial fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "dshot_esc_rpm_fit.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n  Plot saved to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
