#!/usr/bin/env python3
"""
DShot ↔ ESC RPM Mapping Script
===============================
Reads a rosbag with telemetry containing esc_rpm[4], segments the data into
discrete DShot steps, and fits a 2nd-order polynomial:  DShot = f(ESC_RPM).

Usage:
    python3 dshot_esc_rpm_map.py

Configure the parameters in the section below before running.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────── USER CONFIGURATION ────────────────────────────
# Path to the rosbag file
BAG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ros_bags", "0.5_2.2_1420KV.bag"
)

# Telemetry topic that contains esc_rpm
TELEMETRY_TOPIC = "/telemetry117"

# DShot linspace parameters  (start, stop, num_steps)
# These must correspond 1-to-1 with the discrete steps in the bag, in order.
DSHOT_START = 100
DSHOT_END   = 1000
DSHOT_NUM   = 10

# Segmentation parameters
# Minimum gap (seconds) of low / transitional RPM between two plateaus.
# Adjust if your inter-step pause is longer or shorter than 5 s.
GAP_THRESHOLD_S = 2.0

# RPM values below this are considered "idle / gap" between steps.
# Set this somewhat above the noise floor of your idle RPM.
RPM_IDLE_THRESHOLD = 200

# When computing the mean RPM for a plateau, discard the first and last
# TRIM_FRACTION of samples to avoid transients at step edges.
TRIM_FRACTION = 0.15
# ────────────────────────────────────────────────────────────────────────────


def read_esc_rpm(bag_path: str, topic: str):
    """Read all esc_rpm messages and return (timestamps, rpm_array).

    Returns
    -------
    times : np.ndarray, shape (N,)
        Time in seconds (relative to bag start).
    rpms  : np.ndarray, shape (N, 4)
        ESC RPM for each of the 4 motors.
    """
    import rosbag

    bag = rosbag.Bag(bag_path, "r")
    times, rpms = [], []
    t0 = None
    for _, msg, t in bag.read_messages(topics=[topic]):
        ts = t.to_sec()
        if t0 is None:
            t0 = ts
        times.append(ts - t0)
        rpms.append(list(msg.esc_rpm))
    bag.close()

    return np.array(times), np.array(rpms, dtype=float)


def segment_plateaus(times, avg_rpm, gap_threshold_s, rpm_idle_threshold):
    """Identify contiguous plateaus where avg RPM is above idle.

    Returns a list of (start_idx, end_idx) for each plateau.
    """
    above = avg_rpm > rpm_idle_threshold
    plateaus = []
    in_plateau = False
    start = 0

    for i in range(len(above)):
        if above[i] and not in_plateau:
            in_plateau = True
            start = i
        elif not above[i] and in_plateau:
            in_plateau = False
            plateaus.append((start, i - 1))

    # Close the last plateau if it extends to the end
    if in_plateau:
        plateaus.append((start, len(above) - 1))

    # Merge plateaus that are separated by less than gap_threshold_s
    # (in case of brief RPM dips within a single step)
    merged = [plateaus[0]]
    for s, e in plateaus[1:]:
        prev_s, prev_e = merged[-1]
        if times[s] - times[prev_e] < gap_threshold_s:
            merged[-1] = (prev_s, e)  # merge
        else:
            merged.append((s, e))

    return merged


def trim_plateau_mean(avg_rpm, start, end, trim_fraction):
    """Return the trimmed-mean RPM for a plateau."""
    n = end - start + 1
    trim = int(n * trim_fraction)
    if trim < 1:
        trim = 0
    segment = avg_rpm[start + trim : end - trim + 1]
    if len(segment) == 0:
        segment = avg_rpm[start:end + 1]
    return np.mean(segment)


def main():
    # ── 1. Read data ────────────────────────────────────────────────────────
    print(f"Reading bag: {os.path.abspath(BAG_PATH)}")
    times, rpms = read_esc_rpm(BAG_PATH, TELEMETRY_TOPIC)
    print(f"  Total messages: {len(times)}")
    print(f"  Duration: {times[-1]:.1f} s")

    # Average across 4 motors
    avg_rpm = rpms.mean(axis=1)

    # ── 2. Segment into plateaus ────────────────────────────────────────────
    plateaus = segment_plateaus(times, avg_rpm, GAP_THRESHOLD_S, RPM_IDLE_THRESHOLD)
    print(f"  Detected {len(plateaus)} plateaus")

    # ── 3. DShot linspace ───────────────────────────────────────────────────
    dshot_values = np.linspace(DSHOT_START, DSHOT_END, DSHOT_NUM)
    print(f"  DShot values ({DSHOT_NUM} steps): {dshot_values}")

    if len(plateaus) != len(dshot_values):
        print(
            f"\n⚠  WARNING: detected {len(plateaus)} plateaus but expected "
            f"{DSHOT_NUM} DShot steps.\n"
            f"  Adjust RPM_IDLE_THRESHOLD ({RPM_IDLE_THRESHOLD}) or "
            f"GAP_THRESHOLD_S ({GAP_THRESHOLD_S}) until they match.\n"
            f"  Plateau time ranges:"
        )
        for i, (s, e) in enumerate(plateaus):
            print(f"    [{i}] t = {times[s]:.2f} – {times[e]:.2f} s, "
                  f"mean RPM ≈ {avg_rpm[s:e+1].mean():.0f}")
        print("\nProceeding with the minimum of the two for the fit.\n")

    n_fit = min(len(plateaus), len(dshot_values))

    # ── 4. Compute mean RPM per plateau ─────────────────────────────────────
    rpm_means = np.array([
        trim_plateau_mean(avg_rpm, s, e, TRIM_FRACTION)
        for s, e in plateaus[:n_fit]
    ])
    dshot_fit = dshot_values[:n_fit]

    print("\n  Plateau summary:")
    print(f"  {'Step':>4}  {'DShot':>8}  {'Mean RPM':>10}  {'t_start':>8}  {'t_end':>8}")
    for i in range(n_fit):
        s, e = plateaus[i]
        print(f"  {i:4d}  {dshot_fit[i]:8.1f}  {rpm_means[i]:10.1f}  "
              f"{times[s]:8.2f}  {times[e]:8.2f}")

    # ── 5. Fit 2nd-order polynomial: DShot = f(ESC_RPM) ────────────────────
    # Polynomial: DShot = a * RPM^2 + b * RPM + c
    coeffs = np.polyfit(rpm_means, dshot_fit, 2)
    poly = np.poly1d(coeffs)
    a, b, c = coeffs

    print(f"\n  2nd-order polynomial fit  DShot = f(RPM):")
    print(f"    DShot = {a:.6e} * RPM² + {b:.6e} * RPM + {c:.6e}")

    # R² goodness of fit
    ss_res = np.sum((dshot_fit - poly(rpm_means)) ** 2)
    ss_tot = np.sum((dshot_fit - np.mean(dshot_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"    R² = {r_squared:.6f}")

    # ── 6. Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw RPM time series with plateau regions highlighted
    ax = axes[0]
    ax.plot(times, avg_rpm, linewidth=0.5, label="Avg ESC RPM")
    colors = plt.cm.viridis(np.linspace(0, 1, n_fit))
    for i in range(n_fit):
        s, e = plateaus[i]
        ax.axvspan(times[s], times[e], alpha=0.2, color=colors[i],
                   label=f"Step {i} (DShot={dshot_fit[i]:.0f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESC RPM (avg of 4 motors)")
    ax.set_title("Raw ESC RPM with detected plateaus")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: polynomial fit
    ax = axes[1]
    rpm_plot = np.linspace(0, rpm_means.max() * 1.1, 300)
    ax.scatter(rpm_means, dshot_fit, c="red", zorder=5, label="Data points")
    ax.plot(rpm_plot, poly(rpm_plot), "b-", label=(
        f"Fit: DShot = {a:.2e}·RPM² + {b:.2e}·RPM + {c:.2e}\n"
        f"R² = {r_squared:.4f}"
    ))
    ax.set_xlabel("ESC RPM (avg of 4 motors)")
    ax.set_ylabel("DShot command value")
    ax.set_title("DShot = f(ESC RPM)  —  2nd-order polynomial fit")
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
