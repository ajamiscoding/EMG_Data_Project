
"""
EMG Movement Detection & Analysis Pipeline (v4, verbose + dead-channel handling + extra stacked plots)

- Works on an Excel file with multiple sheets.
- Each sheet has:
    * a time column (e.g. 'Time')
    * EMG channels (e.g. 'emg1'..'emg8')

Main stages:
    1) Input & sampling (Fs, Nyquist)
    2) Baseline (rest) analysis & normalization
    3) Preprocessing (HPF + smoothing → envelope-like signal)
    4) Dead-channel detection (flat / zero channels)
    5) Windowing & feature extraction per channel
    6) Consensus across channels (overlapping windows)
    7) Average channel, pulse detection, and movement metrics
    8) Saving tables and multi-stage debug plots per sheet

This version also:
    - Prints a summary of EVERY stage for each sheet
    - Detects dead channels and zeroes them
    - Plots per-step stacked signals:
        * raw EMG
        * env(t)
        * z(t)
        * raw + shaded pulses
        * window features + consensus
        * average activity + pulses
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


# ============================================================
# 1. PARAMETERS
# ============================================================

@dataclass
class PipelineParams:
    # Rest (baseline) segment in seconds
    rest_start_sec: float = 0.0
    rest_end_sec: float = 1.0

    # Pre-processing (for ~100 Hz sampling)
    hp_cutoff_hz: float = 0.5    # high-pass to remove drift
    lp_cutoff_hz: float = 5.0    # low-pass to smooth envelope

    # Windowing (for feature extraction)
    window_ms: float = 100.0     # window length
    overlap_ms: float = 50.0     # overlap between windows

    # Activity threshold (in z-score space)
    z_thresh_active: float = 1.0

    # Consensus across channels: fraction of ACTIVE (alive) channels required
    consensus_min_fraction: float = 0.6

    # Pulse detection thresholds (on the average channel)
    pulse_on_z: float = 0.6
    pulse_off_z: float = 0.5

    # Pulse shape constraints
    pulse_min_duration_ms: float = 150.0
    pulse_merge_gap_ms: float = 300.0


# ============================================================
# 2. BASIC UTILITIES
# ============================================================

def compute_fs(time_s: np.ndarray) -> float:
    """Estimate sampling frequency from time vector (in seconds)."""
    dt = np.median(np.diff(time_s))
    if dt <= 0:
        raise ValueError("Non-positive dt encountered when computing Fs.")
    return 1.0 / dt


def butter_highpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Apply a Butterworth high-pass filter along axis 0."""
    if cutoff_hz <= 0:
        return data
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        # No valid high-pass region
        return data
    b, a = signal.butter(order, cutoff_hz / nyq, btype="highpass")
    return signal.filtfilt(b, a, data, axis=0)


def butter_lowpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Apply a Butterworth low-pass filter along axis 0."""
    if cutoff_hz <= 0:
        return data
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        # No valid low-pass region
        return data
    b, a = signal.butter(order, cutoff_hz / nyq, btype="lowpass")
    return signal.filtfilt(b, a, data, axis=0)


def sliding_window_indices(
    n_samples: int,
    window_samples: int,
    hop_samples: int
) -> List[Tuple[int, int]]:
    """
    Generate (start, end) index pairs for sliding windows.

    The last window is dropped if it doesn't fully fit.
    """
    indices: List[Tuple[int, int]] = []
    if window_samples <= 0 or hop_samples <= 0:
        return indices
    start = 0
    while start + window_samples <= n_samples:
        indices.append((start, start + window_samples))
        start += hop_samples
    return indices


def mad_1d(x: np.ndarray) -> float:
    """Median Absolute Deviation of a 1D array."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def print_channel_stats(name: str, data: np.ndarray, channel_names: List[str], max_channels: int = 8):
    """
    Print basic per-channel stats for a [T x C] array:
        mean, std, min, max
    """
    n_samples, n_channels = data.shape
    print(f"\n  [STATS] {name}: shape = {data.shape}")
    for i in range(min(n_channels, max_channels)):
        ch = channel_names[i] if i < len(channel_names) else f"ch{i}"
        col = data[:, i]
        print(f"    {ch:10s} mean={np.nanmean(col):8.4f}, "
              f"std={np.nanstd(col):8.4f}, "
              f"min={np.nanmin(col):8.4f}, max={np.nanmax(col):8.4f}")
    if n_channels > max_channels:
        print(f"    ... ({n_channels - max_channels} more channels not printed)")


# ============================================================
# 3. STAGE 1 – INPUT & SAMPLING
# ============================================================

def find_time_and_emg_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Heuristically find time column and EMG columns in a DataFrame."""
    # Time column candidates
    time_col_candidates = ["Time", "time", "t", "Zaman", "Timestamp"]
    time_col = None
    for c in df.columns:
        if c in time_col_candidates:
            time_col = c
            break
    if time_col is None:
        # fallback: assume first column is time
        time_col = df.columns[0]

    # EMG channels: columns that are not the time column
    emg_cols = [c for c in df.columns if c != time_col]

    if len(emg_cols) == 0:
        raise ValueError("No EMG columns found in sheet. Please check column names.")

    time_raw = df[time_col].to_numpy(dtype=float)

    # Many times EMG Excel has time in ms; try to detect
    if np.nanmax(np.diff(time_raw)) > 1.0:  # crude check
        time_s = time_raw / 1000.0
    else:
        time_s = time_raw

    emg_matrix = df[emg_cols].to_numpy(dtype=float)

    return time_s, emg_matrix, emg_cols


def load_sheet(xlsx_path: Path, sheet_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load one sheet from the Excel file and return time + EMG matrix."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    time_s, emg_matrix, emg_cols = find_time_and_emg_columns(df)
    return time_s, emg_matrix, emg_cols


# ============================================================
# 4. STAGE 2 – BASELINE & NORMALIZATION
# ============================================================

def select_rest_segment(
    time_s: np.ndarray,
    signal_matrix: np.ndarray,
    rest_start: float,
    rest_end: float
) -> np.ndarray:
    """
    Extract the rest segment used for baseline statistics.
    If the requested range is invalid, fallback to the first 5 seconds
    or the first 20% of the signal.
    """
    mask = (time_s >= rest_start) & (time_s <= rest_end)
    if mask.sum() < 10:
        # Fallback
        fallback_end_idx = int(min(len(time_s) - 1, int(0.2 * len(time_s))))
        fallback_end = min(time_s[0] + 5.0, time_s[fallback_end_idx])
        mask = (time_s >= time_s[0]) & (time_s <= fallback_end)
    return signal_matrix[mask, :]


def compute_baseline_stats(rest_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-channel baseline statistics (mean, std, median, MAD)."""
    mean = np.nanmean(rest_matrix, axis=0)
    std = np.nanstd(rest_matrix, axis=0, ddof=1)
    median = np.nanmedian(rest_matrix, axis=0)
    mad = np.array([mad_1d(rest_matrix[:, i]) for i in range(rest_matrix.shape[1])])

    # Avoid zero MAD
    mad[mad == 0] = 1e-8

    return dict(mean=mean, std=std, median=median, mad=mad)


def zero_dead_channels(
    env_matrix: np.ndarray,
    baseline_stats: Dict[str, np.ndarray],
    channel_names: List[str],
    var_threshold: float = 1e-6,
    mad_threshold: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Detect EMG channels that are effectively dead (flat lines).
    A channel is considered dead if:
        - variance of env < var_threshold, or
        - MAD (from baseline) < mad_threshold

    All dead channels are forced to zero in:
        - env_matrix
        - baseline stats

    Returns:
        env_fixed            -> env with dead channels forced to zero
        baseline_fixed       -> baseline stats with dead channels zeroed
        alive_mask (bool)    -> mask of channels that are alive (not dead)
    """
    variances = np.var(env_matrix, axis=0)
    mad = baseline_stats["mad"]

    dead_mask = (variances < var_threshold) | (mad < mad_threshold)
    alive_mask = ~dead_mask

    print("\n  STEP 3b: Dead-channel detection")
    for i, ch in enumerate(channel_names):
        status = "DEAD  (set to zero)" if dead_mask[i] else "alive"
        print(f"    {ch:10s}  var={variances[i]:.3e}, MAD={mad[i]:.3e}  -> {status}")

    # Zero out dead channels in env
    env_fixed = env_matrix.copy()
    env_fixed[:, dead_mask] = 0.0

    # Fix baseline stats for dead channels
    baseline_fixed: Dict[str, np.ndarray] = {}
    for key in baseline_stats:
        arr = baseline_stats[key].copy()
        arr[dead_mask] = 0.0
        baseline_fixed[key] = arr

    return env_fixed, baseline_fixed, alive_mask


def normalize_signals(env_matrix: np.ndarray, baseline_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Bias-remove and normalize to robust z per channel:
        z_i(t) = (env_i(t) - median_i) / MAD_i
    """
    m0 = baseline_stats["median"]
    mad = baseline_stats["mad"]
    z = (env_matrix - m0[None, :]) / mad[None, :]
    return z


# ============================================================
# 5. STAGE 3 – PREPROCESSING (HPF, LPF, ENVELOPE)
# ============================================================

def preprocess_signals(
    emg_matrix: np.ndarray,
    fs: float,
    params: PipelineParams
) -> np.ndarray:
    """
    Apply high-pass to remove drift, low-pass to smooth,
    and get an envelope-like activity signal per channel.

    For ~100 Hz sampling, we treat the EMG as 'envelope-like'
    but still remove drift and smooth it.
    """
    # High-pass for drift removal
    hp = butter_highpass(emg_matrix, fs, params.hp_cutoff_hz, order=2)
    # Rectify
    rectified = np.abs(hp)
    # Low-pass to get smooth envelope
    env = butter_lowpass(rectified, fs, params.lp_cutoff_hz, order=2)
    return env


# ============================================================
# 6. STAGE 4 – WINDOW ANALYSIS & CONSENSUS
# ============================================================

def compute_window_features(
    z_matrix: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    params: PipelineParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean z-score per window and channel.

    Returns:
        features: shape (n_windows, n_channels)
        t_centers: time of the center of each window (sec)
    """
    n_samples, n_channels = z_matrix.shape

    window_samples = int(round(params.window_ms * 1e-3 * fs))
    overlap_samples = int(round(params.overlap_ms * 1e-3 * fs))
    hop_samples = max(1, window_samples - overlap_samples)

    idx = sliding_window_indices(n_samples, window_samples, hop_samples)
    if len(idx) == 0:
        raise ValueError("No sliding windows generated. Check window_ms/overlap_ms vs signal length.")

    features = np.zeros((len(idx), n_channels), dtype=float)
    t_centers = np.zeros(len(idx), dtype=float)

    for k, (start, end) in enumerate(idx):
        segment = z_matrix[start:end, :]    # shape (window_samples, n_channels)
        features[k, :] = np.nanmean(segment, axis=0)  # mean z per channel
        t_centers[k] = 0.5 * (time_s[start] + time_s[end - 1])

    return features, t_centers


def per_channel_activity(
    features: np.ndarray,
    z_thresh_active: float
) -> np.ndarray:
    """
    Binary activity per window and channel from threshold.
    features: (n_windows, n_channels)
    """
    active = (features > z_thresh_active).astype(int)
    return active


def consensus_windows(
    active_matrix: np.ndarray,
    min_fraction: float
) -> np.ndarray:
    """
    Build consensus mask where enough channels are active.
    active_matrix: shape (n_windows, n_alive_channels), entries 0 or 1.
    min_fraction: e.g. 0.5 means at least 50% of ALIVE channels must be active.
    """
    n_windows, n_channels_alive = active_matrix.shape
    counts = active_matrix.sum(axis=1)
    required = int(np.ceil(min_fraction * n_channels_alive))
    required = max(1, required)
    consensus = counts >= required
    return consensus


# ============================================================
# 7. STAGE 5 – AVERAGE CHANNEL, PULSES, METRICS
# ============================================================

def average_channel(
    features: np.ndarray,
    consensus_mask: np.ndarray,
    alive_mask: np.ndarray
) -> np.ndarray:
    """
    Average normalized feature across ALIVE channels.
    Outside consensus windows, we set value to 0 to emphasize overlap region.
    """
    if alive_mask.sum() == 0:
        avg = np.zeros(features.shape[0], dtype=float)
    else:
        avg = features[:, alive_mask].mean(axis=1)
    avg[~consensus_mask] = 0.0
    return avg


def detect_pulses(
    avg_signal: np.ndarray,
    t_centers: np.ndarray,
    params: PipelineParams
) -> List[Tuple[int, int]]:
    """
    Detect pulses from average window activity using hysteresis and duration rules.

    avg_signal: 1D, length = n_windows
    t_centers: time for each window
    Returns:
        list of (start_index, end_index) in window index space.
    """
    on_thr = params.pulse_on_z
    off_thr = params.pulse_off_z
    min_dur_sec = params.pulse_min_duration_ms * 1e-3
    merge_gap_sec = params.pulse_merge_gap_ms * 1e-3

    n = len(avg_signal)
    pulses: List[Tuple[int, int]] = []

    in_pulse = False
    start_idx = 0

    for i in range(n):
        v = avg_signal[i]

        if not in_pulse:
            if v >= on_thr:
                in_pulse = True
                start_idx = i
        else:
            if v <= off_thr:
                end_idx = i
                dur = t_centers[end_idx] - t_centers[start_idx]
                if dur >= min_dur_sec:
                    pulses.append((start_idx, end_idx))
                in_pulse = False

    if in_pulse:
        end_idx = n - 1
        dur = t_centers[end_idx] - t_centers[start_idx]
        if dur >= min_dur_sec:
            pulses.append((start_idx, end_idx))

    if len(pulses) <= 1:
        return pulses

    # Merge pulses that are too close
    merged: List[Tuple[int, int]] = []
    current_start, current_end = pulses[0]

    for (s, e) in pulses[1:]:
        gap = t_centers[s] - t_centers[current_end]
        if gap <= merge_gap_sec:
            current_end = e
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e

    merged.append((current_start, current_end))
    return merged


def compute_pulse_metrics(
    avg_signal: np.ndarray,
    t_centers: np.ndarray,
    pulses: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Compute metrics (start, end, duration, peak, TR1, TR2, area) for each pulse.

    - Rise time TR1: time from 10% to 90% of peak
    - Fall time TR2: time from 90% to 10% of peak (after the peak)
    """
    rows = []

    for j, (i_start, i_end) in enumerate(pulses):
        seg = avg_signal[i_start:i_end + 1]
        t_seg = t_centers[i_start:i_end + 1]

        if len(seg) < 2:
            continue

        peak_idx = int(np.argmax(seg))
        peak_val = float(seg[peak_idx])
        peak_time = float(t_seg[peak_idx])

        start_time = float(t_seg[0])
        end_time = float(t_seg[-1])
        duration = end_time - start_time

        # area under curve (trapezoidal)
        auc = float(np.trapz(seg, t_seg))

        tr1 = np.nan
        tr2 = np.nan

        if peak_val > 0:
            level_10 = 0.1 * peak_val
            level_90 = 0.9 * peak_val

            # TR1: before peak
            pre_peak_seg = seg[:peak_idx + 1]
            pre_peak_t = t_seg[:peak_idx + 1]

            idx10 = None
            for k in range(len(pre_peak_seg)):
                if pre_peak_seg[k] >= level_10:
                    idx10 = k
                    break
            idx90 = None
            for k in range(len(pre_peak_seg)):
                if pre_peak_seg[k] >= level_90:
                    idx90 = k
                    break

            if idx10 is not None and idx90 is not None and idx90 > idx10:
                tr1 = float(pre_peak_t[idx90] - pre_peak_t[idx10])

            # TR2: after peak
            post_peak_seg = seg[peak_idx:]
            post_peak_t = t_seg[peak_idx:]

            idx90d = None
            for k in range(len(post_peak_seg)):
                if post_peak_seg[k] <= level_90:
                    idx90d = k
                    break
            idx10d = None
            for k in range(len(post_peak_seg)):
                if post_peak_seg[k] <= level_10:
                    idx10d = k
                    break

            if idx90d is not None and idx10d is not None and idx10d > idx90d:
                tr2 = float(post_peak_t[idx10d] - post_peak_t[idx90d])

        rows.append(
            dict(
                pulse_index=j,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                peak_time=peak_time,
                peak_value=peak_val,
                tr1_rise_time=tr1,
                tr2_fall_time=tr2,
                auc=auc,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. STAGE 6 – PIPELINE PER SHEET + STEP-BY-STEP PRINTS
# ============================================================

def run_pipeline_on_sheet(
    xlsx_path: Path,
    sheet_name: str,
    params: PipelineParams
) -> Dict[str, object]:
    """Full pipeline for a single sheet. Returns metrics and intermediate results."""
    # ---------- STEP 1: Load & sampling ----------
    print(f"\n========== SHEET: {sheet_name} ==========")
    print("STEP 1: Loading data and computing sampling rate...")
    time_s, emg, channel_names = load_sheet(xlsx_path, sheet_name)
    fs = compute_fs(time_s)
    nyq = 0.5 * fs

    print(f"  Fs  = {fs:.3f} Hz, Nyq = {nyq:.3f} Hz")
    print(f"  Duration = {time_s[-1] - time_s[0]:.3f} s")
    print(f"  Channels ({len(channel_names)}): {channel_names}")
    print_channel_stats("Raw EMG", emg, channel_names)

    # ---------- STEP 2: Preprocessing ----------
    print("\nSTEP 2: Preprocessing (HPF + rectification + LPF envelope)...")
    env = preprocess_signals(emg, fs, params)
    print_channel_stats("Envelope env(t)", env, channel_names)

    # ---------- STEP 3: Baseline (rest) ----------
    print("\nSTEP 3: Baseline analysis from rest segment...")
    rest = select_rest_segment(time_s, env, params.rest_start_sec, params.rest_end_sec)
    baseline_stats = compute_baseline_stats(rest)
    print("  Baseline (median, MAD) per channel:")
    for ch, med, mad in zip(channel_names, baseline_stats["median"], baseline_stats["mad"]):
        print(f"    {ch:10s} median={med:8.4f}, MAD={mad:8.4f}")

    # ---------- STEP 3b: Dead channels ----------
    env, baseline_stats, alive_mask = zero_dead_channels(env, baseline_stats, channel_names)

    # ---------- STEP 4: Normalization ----------
    print("\nSTEP 4: Normalization (robust z-score using median & MAD)...")
    z = normalize_signals(env, baseline_stats)
    print_channel_stats("z(t)", z, channel_names)

    # ---------- STEP 5: Window features ----------
    print("\nSTEP 5: Windowing and feature extraction (mean z per window/channel)...")
    features, t_centers = compute_window_features(z, time_s, fs, params)
    n_windows, n_channels = features.shape
    print(f"  Windows: {n_windows} (window_ms={params.window_ms}, overlap_ms={params.overlap_ms})")
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Window time range: {t_centers[0]:.3f}s → {t_centers[-1]:.3f}s")
    print_channel_stats("Window mean z", features, channel_names)

    # ---------- STEP 6: Activity & consensus ----------
    print("\nSTEP 6: Per-channel activity and consensus across channels...")
    active_all = per_channel_activity(features, params.z_thresh_active)
    active_alive = active_all[:, alive_mask]
    n_alive = int(alive_mask.sum())
    print(f"  Alive channels: {n_alive} / {n_channels}")
    print(f"  Activity threshold (z): {params.z_thresh_active}")

    # Print how many windows active per alive channel
    for i, ch in enumerate(channel_names):
        if not alive_mask[i]:
            continue
        idx_alive = np.where(alive_mask)[0].tolist().index(i)
        n_active = active_alive[:, idx_alive].sum()
        print(f"    {ch:10s} active in {n_active} / {n_windows} windows")

    consensus = consensus_windows(active_alive, params.consensus_min_fraction)
    consensus_frac = consensus.mean() if n_windows > 0 else 0.0
    print(f"  Consensus rule: at least {params.consensus_min_fraction*100:.0f}% of alive channels active")
    print(f"  Consensus windows: {consensus.sum()} / {n_windows} "
          f"({consensus_frac*100:.1f}%)")

    # ---------- STEP 7: Average channel & pulses ----------
    print("\nSTEP 7: Average channel and pulse detection...")
    avg = average_channel(features, consensus, alive_mask)
    print(f"  Average activity shape: {avg.shape}")
    print(f"  Avg min={avg.min():.4f}, max={avg.max():.4f}, mean={avg.mean():.4f}")

    pulses = detect_pulses(avg, t_centers, params)
    print(f"  Detected pulses: {len(pulses)}")
    for j, (s, e) in enumerate(pulses):
        print(f"    Pulse {j}: {t_centers[s]:.3f}s → {t_centers[e]:.3f}s")

    metrics_df = compute_pulse_metrics(avg, t_centers, pulses)
    print("  Pulse metrics table:")
    if len(metrics_df) == 0:
        print("    (no pulses)")
    else:
        print(metrics_df)

    return dict(
        time=time_s,
        fs=fs,
        nyq=nyq,
        emg=emg,
        env=env,
        z=z,
        features=features,
        t_centers=t_centers,
        active_all=active_all,
        active_alive=active_alive,
        alive_mask=alive_mask,
        consensus=consensus,
        avg=avg,
        pulses=pulses,
        metrics_df=metrics_df,
        baseline_stats=baseline_stats,
        channel_names=channel_names,
    )


# ============================================================
# 9. DEBUG PLOTS – ALL STAGES & CHANNELS
# ============================================================

def plot_debug_stages(
    out_dir: Path,
    sheet_name: str,
    results: Dict[str, object]
) -> None:
    """
    Make multi-panel plots showing all major stages on all channels:

      1) Raw EMG (offset stacked)
      2) Preprocessed envelope env(t) (stacked)
      3) Normalized z(t) (stacked)
      4) Raw EMG with shaded pulses (stacked)
      5) Window-level mean z per channel + consensus
      6) Average activity with detected pulses
    """
    time_s = results["time"]
    emg = results["emg"]
    env = results["env"]
    z = results["z"]
    features = results["features"]          # (n_windows, n_channels)
    t_centers = results["t_centers"]        # (n_windows,)
    consensus = results["consensus"]        # (n_windows,)
    avg = results["avg"]                    # (n_windows,)
    pulses = results["pulses"]
    channel_names: List[str] = results["channel_names"]

    n_samples, n_channels = emg.shape

    # ---------- 1) Raw EMG (offset stacked) ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    max_abs = np.max(np.abs(emg))
    offset = max_abs * 1.2 if max_abs > 0 else 1.0
    for i in range(n_channels):
        ax.plot(time_s, emg[:, i] + i * offset, label=channel_names[i])
    ax.set_title(f"Raw EMG (stacked) – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude + offset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_raw_stacked.png", dpi=200)
    plt.close(fig)

    # ---------- 2) Envelope env(t) per channel (stacked) ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    max_env = np.max(env)
    offset_env = max_env * 1.2 if max_env > 0 else 1.0
    for i in range(n_channels):
        ax.plot(time_s, env[:, i] + i * offset_env, label=channel_names[i])
    ax.set_title(f"Preprocessed Envelope per Channel – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Envelope + offset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_env_stacked.png", dpi=200)
    plt.close(fig)

    # ---------- 3) Normalized z(t) per channel (stacked) ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    max_z = np.nanmax(np.abs(z))
    offset_z = max_z * 1.2 if max_z > 0 else 1.0
    for i in range(n_channels):
        ax.plot(time_s, z[:, i] + i * offset_z, label=channel_names[i])
    ax.set_title(f"Normalized z(t) per Channel – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("z-score + offset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_z_stacked.png", dpi=200)
    plt.close(fig)

    # ---------- 4) Raw EMG with pulses shaded (stacked) ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_channels):
        ax.plot(time_s, emg[:, i] + i * offset, label=channel_names[i])
    # Shade pulses using window centers
    for (i_start, i_end) in pulses:
        ax.axvspan(t_centers[i_start], t_centers[i_end], color="red", alpha=0.15)
    ax.set_title(f"Raw EMG with Detected Pulses (stacked) – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude + offset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_raw_with_pulses.png", dpi=200)
    plt.close(fig)

    # ---------- 5) Window-level features & consensus ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_channels):
        ax.plot(t_centers, features[:, i], alpha=0.7, label=channel_names[i])
    ax.set_title(f"Window Mean z per Channel – {sheet_name}")
    ax.set_xlabel("Time [s] (window centers)")
    ax.set_ylabel("Mean z")
    ax.grid(True, alpha=0.3)

    # Overlay consensus as shaded vertical bands
    if len(t_centers) > 1:
        dt_win = t_centers[1] - t_centers[0]
    else:
        dt_win = 0.0
    for idx, is_cons in enumerate(consensus):
        if is_cons:
            t = t_centers[idx]
            ax.axvspan(t - 0.5 * dt_win, t + 0.5 * dt_win, color="orange", alpha=0.1)

    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_features_consensus.png", dpi=200)
    plt.close(fig)

    # ---------- 6) Average activity with pulses ----------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_centers, avg, label="Average activity (A_avg)", linewidth=1.5)
    ax.axhline(0.0, color="k", linewidth=0.5)

    # Shade pulses
    for (i_start, i_end) in pulses:
        ax.axvspan(t_centers[i_start], t_centers[i_end], color="red", alpha=0.25)

    ax.set_title(f"Average Activity and Detected Pulses – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Average z")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_avg_pulses.png", dpi=200)
    plt.close(fig)


# ============================================================
# 10. SAVE RESULTS PER SHEET
# ============================================================

def save_results_for_sheet(
    out_dir: Path,
    sheet_name: str,
    results: Dict[str, object]
) -> None:
    """Save CSV tables and debug plots for a given sheet."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline stats
    baseline = results["baseline_stats"]
    channel_names: List[str] = results["channel_names"]
    alive_mask: np.ndarray = results["alive_mask"]
    is_dead = (~alive_mask).astype(int)

    baseline_df = pd.DataFrame({
        "channel": channel_names,
        "mean": baseline["mean"],
        "std": baseline["std"],
        "median": baseline["median"],
        "mad": baseline["mad"],
        "is_dead": is_dead,   # 1 = dead / forced zero, 0 = alive
    })
    baseline_df.to_csv(out_dir / f"{sheet_name}_baseline.csv", index=False)

    # Movement metrics
    metrics_df: pd.DataFrame = results["metrics_df"]
    metrics_df.to_csv(out_dir / f"{sheet_name}_pulses.csv", index=False)

    # Window-level summary
    features = results["features"]
    t_centers = results["t_centers"]
    consensus = results["consensus"]
    win_df = pd.DataFrame(features, columns=[f"mean_z_{ch}" for ch in channel_names])
    win_df.insert(0, "t_center", t_centers)
    win_df["consensus"] = consensus.astype(int)
    win_df.to_csv(out_dir / f"{sheet_name}_windows.csv", index=False)

    # Debug plots for all stages
    plot_debug_stages(out_dir, sheet_name, results)


# ============================================================
# 11. RUN WORKBOOK (ALL SHEETS)
# ============================================================

def run_workbook(
    xlsx_path: Path,
    out_dir: Path,
    params: PipelineParams
) -> None:
    """Run the full pipeline for all sheets in an Excel workbook."""
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)

    xls = pd.ExcelFile(xlsx_path)
    for sheet_name in xls.sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        try:
            results = run_pipeline_on_sheet(xlsx_path, sheet_name, params)
            sheet_out_dir = out_dir / sheet_name
            save_results_for_sheet(sheet_out_dir, sheet_name, results)
        except Exception as e:
            print(f"  [ERROR] Sheet '{sheet_name}' failed: {e}")


# ============================================================
# 12. MAIN / CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EMG Movement Detection & Analysis Pipeline (v4, verbose, dead-channel aware, stacked plots)"
    )
    parser.add_argument(
        "--in_xlsx",
        type=str,
        required=True,
        help="Path to input Excel file (e.g. 30Temmuz_Ampute_polyphase.xlsx)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for CSVs and plots.",
    )
    # Optional overrides
    parser.add_argument("--rest_start", type=float, default=0.0)
    parser.add_argument("--rest_end", type=float, default=5.0)
    parser.add_argument("--window_ms", type=float, default=120.0)
    parser.add_argument("--overlap_ms", type=float, default=60.0)
    parser.add_argument("--z_thresh_active", type=float, default=1.0)
    parser.add_argument("--consensus_min_fraction", type=float, default=0.5)
    parser.add_argument("--pulse_on_z", type=float, default=1.0)
    parser.add_argument("--pulse_off_z", type=float, default=0.5)
    parser.add_argument("--pulse_min_duration_ms", type=float, default=150.0)
    parser.add_argument("--pulse_merge_gap_ms", type=float, default=150.0)
    return parser.parse_args()


def build_params_from_args(args: argparse.Namespace) -> PipelineParams:
    return PipelineParams(
        rest_start_sec=args.rest_start,
        rest_end_sec=args.rest_end,
        window_ms=args.window_ms,
        overlap_ms=args.overlap_ms,
        z_thresh_active=args.z_thresh_active,
        consensus_min_fraction=args.consensus_min_fraction,
        pulse_on_z=args.pulse_on_z,
        pulse_off_z=args.pulse_off_z,
        pulse_min_duration_ms=args.pulse_min_duration_ms,
        pulse_merge_gap_ms=args.pulse_merge_gap_ms,
    )


if __name__ == "__main__":
    args = parse_args()
    params = build_params_from_args(args)
    run_workbook(Path(args.in_xlsx), Path(args.out_dir), params)
