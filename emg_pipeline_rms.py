#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EMG Movement Detection & Analysis Pipeline
Per-channel RMS-based detection + global consensus (no amplitude averaging)

Pipeline per sheet:
    1) Load time + EMG channels, compute Fs & Nyquist.
    2) Preprocess:
        - High-pass (remove drift)
        - Rectify
        - Low-pass (envelope)
    3) Baseline (rest interval) statistics per channel (on envelope).
    4) Dead-channel detection (flat / zero channels) → forced to zero.
    5) Normalize to z-score per channel (median & MAD)  [for inspection only].
    6) Windowing:
        - RMS per channel & window (from envelope).
        - Mean z per channel & window (optional, for plots).
    7) Per-channel automatic RMS activation threshold:
        - From baseline RMS windows,
        - θ_i = median_i + k * MAD_i  (per-channel).
    8) Per-channel binary activity per window:
        - active_i(w) = 1 if RMS_i(w) > θ_i.
    9) Global movement signal:
        - channel_count(w) = sum_active_alive_channels,
        - frac_active(w) = channel_count(w) / n_alive.
    10) Pulse detection on frac_active(w)
    11) Save:
        - baseline.csv (per channel, plus is_dead & rms_threshold)
        - pulses.csv (per global pulse)
        - windows.csv (per window: rms_i, mean_z_i, channel_count, frac_active, global_active)
    12) Plot:
        - <sheet>_raw_stacked.png
        - <sheet>_env_stacked.png
        - <sheet>_z_stacked.png
        - <sheet>_raw_with_pulses.png (all channels stacked + global pulses shaded)
        - <sheet>_features_rms_consensus.png (RMS per channel + global_active shading)
        - <sheet>_global_activity.png (fraction of active channels + pulses)
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
    rest_end_sec: float = 5.0

    # Pre-processing (for ~100 Hz sampling)
    hp_cutoff_hz: float = 0.5    # high-pass to remove drift
    lp_cutoff_hz: float = 5.0    # low-pass to smooth envelope

    # Windowing (for feature extraction)
    window_ms: float = 120.0     # window length
    overlap_ms: float = 60.0     # overlap between windows

    # Per-channel automatic RMS threshold: θ_i = median + k * MAD
    per_channel_k: float = 3.0

    # Consensus across channels: fraction of ALIVE channels required
    consensus_min_fraction: float = 0.5

    # Global pulse detection thresholds on frac_active(w)
    # (0..1 = fraction of alive channels)
    pulse_on_frac: float = 0.6
    pulse_off_frac: float = 0.4

    # Pulse shape constraints (time is in seconds)
    pulse_min_duration_ms: float = 150.0
    pulse_merge_gap_ms: float = 150.0


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
        return data
    b, a = signal.butter(order, cutoff_hz / nyq, btype="highpass")
    return signal.filtfilt(b, a, data, axis=0)


def butter_lowpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Apply a Butterworth low-pass filter along axis 0."""
    if cutoff_hz <= 0:
        return data
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
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
    time_col_candidates = ["Time", "time", "t", "Zaman", "Timestamp"]
    time_col = None
    for c in df.columns:
        if c in time_col_candidates:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]

    emg_cols = [c for c in df.columns if c != time_col]
    if len(emg_cols) == 0:
        raise ValueError("No EMG columns found in sheet. Please check column names.")

    time_raw = df[time_col].to_numpy(dtype=float)
    if np.nanmax(np.diff(time_raw)) > 1.0:  # If time is likely in ms
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

    env_fixed = env_matrix.copy()
    env_fixed[:, dead_mask] = 0.0

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
    (Used mainly for inspection/plots; detection is done on RMS.)
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
    Apply high-pass to remove drift, rectification, and low-pass smoothing
    to obtain an envelope-like activity signal per channel.
    """
    hp = butter_highpass(emg_matrix, fs, params.hp_cutoff_hz, order=2)
    rectified = np.abs(hp)
    env = butter_lowpass(rectified, fs, params.lp_cutoff_hz, order=2)
    return env


# ============================================================
# 6. STAGE 4 – WINDOW ANALYSIS (RMS & mean z)
# ============================================================

def compute_window_features_rms_and_z(
    env_matrix: np.ndarray,
    z_matrix: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    params: PipelineParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-window per-channel features:
        - RMS from env
        - mean z-score from z (optional, for inspection)

    Returns:
        rms_features:  shape (n_windows, n_channels)
        mean_z:        shape (n_windows, n_channels)
        t_centers:     time of the center of each window (sec)
    """
    n_samples, n_channels = env_matrix.shape

    window_samples = int(round(params.window_ms * 1e-3 * fs))
    overlap_samples = int(round(params.overlap_ms * 1e-3 * fs))
    hop_samples = max(1, window_samples - overlap_samples)

    idx = sliding_window_indices(n_samples, window_samples, hop_samples)
    if len(idx) == 0:
        raise ValueError("No sliding windows generated. Check window_ms/overlap_ms vs signal length.")

    rms_features = np.zeros((len(idx), n_channels), dtype=float)
    mean_z = np.zeros((len(idx), n_channels), dtype=float)
    t_centers = np.zeros(len(idx), dtype=float)

    for k, (start, end) in enumerate(idx):
        seg_env = env_matrix[start:end, :]   # (window_samples, n_channels)
        seg_z = z_matrix[start:end, :]
        # RMS per channel
        rms_features[k, :] = np.sqrt(np.mean(seg_env ** 2, axis=0))
        # mean z per channel
        mean_z[k, :] = np.nanmean(seg_z, axis=0)
        t_centers[k] = 0.5 * (time_s[start] + time_s[end - 1])

    return rms_features, mean_z, t_centers


# ============================================================
# 7. STAGE 5 – PER-CHANNEL RMS THRESHOLDS + GLOBAL ACTIVITY
# ============================================================

def compute_per_channel_rms_thresholds(
    rms_features: np.ndarray,
    t_centers: np.ndarray,
    rest_start: float,
    rest_end: float,
    alive_mask: np.ndarray,
    k: float = 3.0,
    min_rest_windows: int = 5,
) -> np.ndarray:
    """
    Compute automatic per-channel RMS thresholds from baseline windows:

    For each channel i:
        1. Take RMS_i(w) in rest region.
        2. θ_i = median + k * MAD  (robust).
        3. If not enough rest windows, use entire signal as backup.
        4. If channel is dead (alive_mask[i] == False), θ_i = +∞  (never active).
    """
    n_windows, n_channels = rms_features.shape
    thresholds = np.zeros(n_channels, dtype=float)

    rest_mask = (t_centers >= rest_start) & (t_centers <= rest_end)

    for i in range(n_channels):
        if not alive_mask[i]:
            thresholds[i] = np.inf
            continue

        vals = rms_features[rest_mask, i]
        if vals.size < min_rest_windows:
            vals = rms_features[:, i]

        med = np.nanmedian(vals)
        mad_val = mad_1d(vals)
        if mad_val == 0 or np.isnan(mad_val):
            mad_val = 1e-6

        thresholds[i] = med + k * mad_val

    return thresholds


def per_channel_activity_rms(rms_features: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Binary activity per window and channel based on per-channel RMS thresholds.
        active_i(w) = 1 if rms_features[w, i] > thresholds[i]
    """
    return (rms_features > thresholds[None, :]).astype(int)


def detect_pulses_on_signal(
    signal_1d: np.ndarray,
    t_centers: np.ndarray,
    params: PipelineParams
) -> List[Tuple[int, int]]:
    """
    Detect pulses from a 1D global activity signal using hysteresis and duration rules.

    signal_1d: length = n_windows, e.g. frac_active(w) in [0, 1].
    Uses:
        on_threshold  = params.pulse_on_frac
        off_threshold = params.pulse_off_frac
    """
    on_thr = params.pulse_on_frac
    off_thr = params.pulse_off_frac
    min_dur_sec = params.pulse_min_duration_ms * 1e-3
    merge_gap_sec = params.pulse_merge_gap_ms * 1e-3

    n = len(signal_1d)
    pulses: List[Tuple[int, int]] = []

    in_pulse = False
    start_idx = 0

    for i in range(n):
        v = signal_1d[i]

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

    if in_pulse and n > 0:
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
    signal_1d: np.ndarray,
    t_centers: np.ndarray,
    pulses: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Compute metrics for each global pulse on a generic activity signal (e.g. frac_active).

    Metrics:
        - start_time, end_time, duration
        - peak_time, peak_value
        - auc (area under signal)
        - tr1_rise_time, tr2_fall_time (optionally meaningful if signal is smooth)
    """
    rows = []

    for j, (i_start, i_end) in enumerate(pulses):
        seg = signal_1d[i_start:i_end + 1]
        t_seg = t_centers[i_start:i_end + 1]

        if len(seg) < 2:
            continue

        peak_idx = int(np.argmax(seg))
        peak_val = float(seg[peak_idx])
        peak_time = float(t_seg[peak_idx])

        start_time = float(t_seg[0])
        end_time = float(t_seg[-1])
        duration = end_time - start_time

        auc = float(np.trapz(seg, t_seg))

        tr1 = np.nan
        tr2 = np.nan

        if peak_val > 0:
            level_10 = 0.1 * peak_val
            level_90 = 0.9 * peak_val

            # TR1: before peak (10% to 90%)
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

            # TR2: after peak (90% to 10%)
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
# 8. STAGE 6 – PIPELINE PER SHEET
# ============================================================

def run_pipeline_on_sheet(
    xlsx_path: Path,
    sheet_name: str,
    params: PipelineParams
) -> Dict[str, object]:
    """Full pipeline for a single sheet. Returns metrics and intermediate results."""
    print(f"\n========== SHEET: {sheet_name} ==========")
    print("STEP 1: Loading data and computing sampling rate...")

    time_s, emg, channel_names = load_sheet(xlsx_path, sheet_name)
    fs = compute_fs(time_s)
    nyq = 0.5 * fs

    print(f"  Fs  = {fs:.3f} Hz, Nyq = {nyq:.3f} Hz")
    print(f"  Duration = {time_s[-1] - time_s[0]:.3f} s")
    print(f"  Channels ({len(channel_names)}): {channel_names}")
    print_channel_stats("Raw EMG", emg, channel_names)

    # STEP 2: Preprocessing
    print("\nSTEP 2: Preprocessing (HPF + rectification + LPF envelope)...")
    env = preprocess_signals(emg, fs, params)
    print_channel_stats("Envelope env(t)", env, channel_names)

    # STEP 3: Baseline & dead channels
    print("\nSTEP 3: Baseline analysis from rest segment...")
    rest = select_rest_segment(time_s, env, params.rest_start_sec, params.rest_end_sec)
    baseline_stats = compute_baseline_stats(rest)
    print("  Baseline (median, MAD) per channel on envelope:")
    for ch, med, mad_val in zip(channel_names, baseline_stats["median"], baseline_stats["mad"]):
        print(f"    {ch:10s} median={med:8.4f}, MAD={mad_val:8.4f}")

    env, baseline_stats, alive_mask = zero_dead_channels(env, baseline_stats, channel_names)

    # STEP 4: Normalization to z (for inspection)
    print("\nSTEP 4: Normalization (robust z-score using median & MAD)...")
    z = normalize_signals(env, baseline_stats)
    print_channel_stats("z(t)", z, channel_names)

    # STEP 5: Window features (RMS + mean z)
    print("\nSTEP 5: Windowing and feature extraction (RMS & mean z per window/channel)...")
    rms_features, mean_z, t_centers = compute_window_features_rms_and_z(env, z, time_s, fs, params)
    n_windows, n_channels = rms_features.shape
    print(f"  Windows: {n_windows} (window_ms={params.window_ms}, overlap_ms={params.overlap_ms})")
    print(f"  RMS feature matrix shape: {rms_features.shape}")
    if n_windows > 0:
        print(f"  Window time range: {t_centers[0]:.3f}s → {t_centers[-1]:.3f}s")
    print_channel_stats("RMS(window)", rms_features, channel_names)

    # STEP 6: Per-channel RMS thresholds & activity
    print("\nSTEP 6: Per-channel RMS automatic thresholds and activity...")
    rms_thresholds = compute_per_channel_rms_thresholds(
        rms_features,
        t_centers,
        params.rest_start_sec,
        params.rest_end_sec,
        alive_mask,
        k=params.per_channel_k,
    )
    for ch, th in zip(channel_names, rms_thresholds):
        print(f"    RMS threshold for {ch:10s}: {th:8.4f}")

    active_all = per_channel_activity_rms(rms_features, rms_thresholds)
    active_all[:, ~alive_mask] = 0  # dead channels never active
    active_alive = active_all[:, alive_mask]
    n_alive = int(alive_mask.sum())
    print(f"  Alive channels: {n_alive} / {n_channels}")

    if n_windows > 0 and n_alive > 0:
        for i, ch in enumerate(channel_names):
            if not alive_mask[i]:
                continue
            idx_alive = np.where(alive_mask)[0].tolist().index(i)
            n_active = active_alive[:, idx_alive].sum()
            print(f"    {ch:10s} active in {n_active} / {n_windows} windows")

    # STEP 7: Global activity (fraction of active channels) & pulses
    print("\nSTEP 7: Global activity (fraction of active channels) and pulse detection...")
    if n_alive == 0 or n_windows == 0:
        frac_active = np.zeros(n_windows, dtype=float)
        channel_count = np.zeros(n_windows, dtype=int)
    else:
        channel_count = active_alive.sum(axis=1)
        frac_active = channel_count.astype(float) / float(n_alive)

    print(f"  Fraction active: min={frac_active.min():.4f}, "
          f"max={frac_active.max():.4f}, mean={frac_active.mean():.4f}")

    global_active = frac_active >= params.consensus_min_fraction
    consensus_frac = global_active.mean() if n_windows > 0 else 0.0
    print(f"  Global active windows (frac >= {params.consensus_min_fraction:.2f}): "
          f"{global_active.sum()} / {n_windows} ({consensus_frac*100:.1f}%)")

    pulses = detect_pulses_on_signal(frac_active, t_centers, params)
    print(f"  Detected global pulses: {len(pulses)}")
    for j, (s, e) in enumerate(pulses):
        print(f"    Pulse {j}: {t_centers[s]:.3f}s → {t_centers[e]:.3f}s")

    metrics_df = compute_pulse_metrics(frac_active, t_centers, pulses)
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
        rms=rms_features,
        mean_z=mean_z,
        t_centers=t_centers,
        rms_thresholds=rms_thresholds,
        active_all=active_all,
        active_alive=active_alive,
        alive_mask=alive_mask,
        channel_count=channel_count,
        frac_active=frac_active,
        global_active=global_active,
        pulses=pulses,
        metrics_df=metrics_df,
        baseline_stats=baseline_stats,
        channel_names=channel_names,
    )


# ============================================================
# 9. PLOTS – ALL STAGES & GLOBAL PULSES
# ============================================================

def plot_debug_stages(
    out_dir: Path,
    sheet_name: str,
    results: Dict[str, object]
) -> None:
    """
    Generate plots for each sheet:
      1) Raw EMG (stacked)
      2) Envelope env(t) (stacked)
      3) Normalized z(t) (stacked)
      4) Raw EMG with global pulses shaded (stacked)
      5) Window RMS per channel + global_active shading
      6) Global fraction of active channels + pulses
    """
    time_s = results["time"]
    emg = results["emg"]
    env = results["env"]
    z = results["z"]
    rms_features = results["rms"]
    t_centers = results["t_centers"]
    frac_active = results["frac_active"]
    global_active = results["global_active"]
    pulses = results["pulses"]
    channel_names: List[str] = results["channel_names"]

    n_samples, n_channels = emg.shape
    n_windows = len(t_centers)

    # 1) Raw EMG (stacked)
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

    # 2) Envelope env(t) (stacked)
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

    # 3) Normalized z(t) (stacked) – for inspection
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

    # 4) Raw EMG with global pulses shaded (stacked)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_channels):
        ax.plot(time_s, emg[:, i] + i * offset, label=channel_names[i])
    for (i_start, i_end) in pulses:
        if n_windows == 0:
            continue
        ax.axvspan(t_centers[i_start], t_centers[i_end], color="red", alpha=0.15)
    ax.set_title(f"Raw EMG with Global Movement Pulses (stacked) – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude + offset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_raw_with_pulses.png", dpi=200)
    plt.close(fig)

    # 5) Window RMS per channel + global_active shading
    if n_windows > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(n_channels):
            ax.plot(t_centers, rms_features[:, i], alpha=0.7, label=channel_names[i])
        ax.set_title(f"Window RMS per Channel – {sheet_name}")
        ax.set_xlabel("Time [s] (window centers)")
        ax.set_ylabel("RMS (envelope)")
        ax.grid(True, alpha=0.3)

        dt_win = (t_centers[1] - t_centers[0]) if n_windows > 1 else 0.0
        for idx, is_active in enumerate(global_active):
            if is_active:
                t = t_centers[idx]
                ax.axvspan(t - 0.5 * dt_win, t + 0.5 * dt_win, color="orange", alpha=0.1)

        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sheet_name}_features_rms_consensus.png", dpi=200)
        plt.close(fig)

    # 6) Global fraction of active channels + pulses
    fig, ax = plt.subplots(figsize=(12, 4))
    if n_windows > 0:
        ax.plot(t_centers, frac_active, label="Fraction of active channels", linewidth=1.5)
        ax.axhline(0.0, color="k", linewidth=0.5)

        for (i_start, i_end) in pulses:
            ax.axvspan(t_centers[i_start], t_centers[i_end], color="red", alpha=0.25)

    ax.set_title(f"Global Activity (Fraction of Active Channels) – {sheet_name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fraction active")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sheet_name}_global_activity.png", dpi=200)
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

    baseline = results["baseline_stats"]
    channel_names: List[str] = results["channel_names"]
    alive_mask: np.ndarray = results["alive_mask"]
    rms_thresholds: np.ndarray = results["rms_thresholds"]
    is_dead = (~alive_mask).astype(int)

    # Baseline stats + per-channel RMS threshold
    baseline_df = pd.DataFrame({
        "channel": channel_names,
        "mean_env": baseline["mean"],
        "std_env": baseline["std"],
        "median_env": baseline["median"],
        "mad_env": baseline["mad"],
        "is_dead": is_dead,              # 1 = dead / forced zero, 0 = alive
        "rms_threshold": rms_thresholds,  # threshold used for per-channel activation
    })
    baseline_df.to_csv(out_dir / f"{sheet_name}_baseline.csv", index=False)

    # Global movement metrics
    metrics_df: pd.DataFrame = results["metrics_df"]
    metrics_df.to_csv(out_dir / f"{sheet_name}_pulses.csv", index=False)

    # Window-level summary
    rms_features = results["rms"]
    mean_z = results["mean_z"]
    t_centers = results["t_centers"]
    channel_count = results["channel_count"]
    frac_active = results["frac_active"]
    global_active = results["global_active"]

    win_df = pd.DataFrame(rms_features, columns=[f"rms_{ch}" for ch in channel_names])
    for i, ch in enumerate(channel_names):
        win_df[f"mean_z_{ch}"] = mean_z[:, i]
    win_df.insert(0, "t_center", t_centers)
    win_df["channel_count"] = channel_count
    win_df["frac_active"] = frac_active
    win_df["global_active"] = global_active.astype(int)
    win_df.to_csv(out_dir / f"{sheet_name}_windows.csv", index=False)

    # Debug plots
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
        description="EMG Movement Detection & Analysis Pipeline (RMS-based per-channel detection + global consensus)"
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
    parser.add_argument("--per_channel_k", type=float, default=3.0)
    parser.add_argument("--consensus_min_fraction", type=float, default=0.5)
    parser.add_argument("--pulse_on_frac", type=float, default=0.6)
    parser.add_argument("--pulse_off_frac", type=float, default=0.4)
    parser.add_argument("--pulse_min_duration_ms", type=float, default=150.0)
    parser.add_argument("--pulse_merge_gap_ms", type=float, default=150.0)
    return parser.parse_args()


def build_params_from_args(args: argparse.Namespace) -> PipelineParams:
    return PipelineParams(
        rest_start_sec=args.rest_start,
        rest_end_sec=args.rest_end,
        window_ms=args.window_ms,
        overlap_ms=args.overlap_ms,
        per_channel_k=args.per_channel_k,
        consensus_min_fraction=args.consensus_min_fraction,
        pulse_on_frac=args.pulse_on_frac,
        pulse_off_frac=args.pulse_off_frac,
        pulse_min_duration_ms=args.pulse_min_duration_ms,
        pulse_merge_gap_ms=args.pulse_merge_gap_ms,
    )


if __name__ == "__main__":
    args = parse_args()
    params = build_params_from_args(args)
    run_workbook(Path(args.in_xlsx), Path(args.out_dir), params)
