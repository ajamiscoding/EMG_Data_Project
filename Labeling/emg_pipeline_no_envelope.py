#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EMG-like Movement Detection Pipeline (NO ENVELOPE)

For your data (slow, already envelope-like):
    • Time handling: supports ms or seconds.
    • Preprocessing per channel:
        raw -> high-pass (0.5 Hz) -> abs()  (this is our "activity" signal).
    • Baseline from rest (on abs(HPF)).
    • z-normalization using median & MAD (for inspection).
    • Optional gating ONLY on z(t): tiny |z| -> 0 (logic/plots only).
    • Window RMS per channel (on abs(HPF), not envelope).
    • Per-channel automatic RMS thresholds (median + k*MAD, capped by percentile).
    • Per-channel activity (binary), global consensus via fraction of active channels.
    • Pulse detection on global activity.
    • Stacked plots for:
        - raw
        - high-pass
        - rectified abs(HPF)
        - z(t)
        - raw with pulses
        - RMS + consensus shading
        - global fraction of active channels

Author: you + ChatGPT :)
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

    # Pre-processing (for ~1000 Hz sampling of slow EMG-like signals)
    hp_cutoff_hz: float = 0.5  # high-pass to remove drift

    # Windowing (for feature extraction)
    window_ms: float = 100.0   # window length
    overlap_ms: float = 50.0  # overlap between windows

    # Per-channel automatic RMS threshold: θ_i from baseline windows
    per_channel_k: float = 2.0       # multiplier for MAD
    per_channel_perc: float = 85.0   # percentile cap (e.g. 85th)

    # Gating on z(t): values with |z| < gate_eps set to zero (z only)
    gate_eps: float = 0.01

    # Consensus across channels: fraction of ALIVE channels required
    consensus_min_fraction: float = 0.4

    # Global pulse detection thresholds on frac_active(w)
    pulse_on_frac: float = 0.5
    pulse_off_frac: float = 0.3

    # Pulse shape constraints (time is in seconds)
    pulse_min_duration_ms: float = 150.0
    pulse_merge_gap_ms: float = 250.0


# ============================================================
# 2. BASIC UTILITIES
# ============================================================

def compute_fs(time_s: np.ndarray) -> float:
    dt = np.median(np.diff(time_s))
    if dt <= 0:
        raise ValueError("Non-positive dt encountered when computing Fs.")
    return 1.0 / dt


def butter_highpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if cutoff_hz <= 0:
        return data
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        return data
    b, a = signal.butter(order, cutoff_hz / nyq, btype="highpass")
    return signal.filtfilt(b, a, data, axis=0)


def sliding_window_indices(
    n_samples: int,
    window_samples: int,
    hop_samples: int
) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    if window_samples <= 0 or hop_samples <= 0:
        return indices
    start = 0
    while start + window_samples <= n_samples:
        indices.append((start, start + window_samples))
        start += hop_samples
    return indices


def mad_1d(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def print_channel_stats(name: str, data: np.ndarray, channel_names: List[str], max_channels: int = 8):
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
# 3. INPUT & SAMPLING (ms / seconds)
# ============================================================

def find_time_and_emg_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
        raise ValueError("No EMG columns found in sheet.")

    time_raw = df[time_col].to_numpy(dtype=float)
    dt_raw = np.nanmedian(np.diff(time_raw))
    max_raw = np.nanmax(time_raw)

    # If step ~1 and goes beyond 100 -> assume ms
    if (0.9 <= dt_raw <= 1.1) and (max_raw > 100.0):
        print(f"  Detected time in milliseconds (dt≈{dt_raw}, max={max_raw}). Converting to seconds.")
        time_s = time_raw / 1000.0
    else:
        print(f"  Detected time likely in seconds (dt≈{dt_raw}, max={max_raw}).")
        time_s = time_raw

    emg_matrix = df[emg_cols].to_numpy(dtype=float)
    return time_s, emg_matrix, emg_cols


def load_sheet(xlsx_path: Path, sheet_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    return find_time_and_emg_columns(df)


# ============================================================
# 4. BASELINE & NORMALIZATION
# ============================================================

def select_rest_segment(
    time_s: np.ndarray,
    signal_matrix: np.ndarray,
    rest_start: float,
    rest_end: float
) -> np.ndarray:
    mask = (time_s >= rest_start) & (time_s <= rest_end)
    if mask.sum() < 10:
        # fallback: first ~20% or up to 5s
        fallback_end_idx = int(min(len(time_s) - 1, int(0.2 * len(time_s))))
        fallback_end = min(time_s[0] + 5.0, time_s[fallback_end_idx])
        mask = (time_s >= time_s[0]) & (time_s <= fallback_end)
    return signal_matrix[mask, :]


def compute_baseline_stats(rest_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.nanmean(rest_matrix, axis=0)
    std = np.nanstd(rest_matrix, axis=0, ddof=1)
    median = np.nanmedian(rest_matrix, axis=0)
    mad = np.array([mad_1d(rest_matrix[:, i]) for i in range(rest_matrix.shape[1])])
    mad[mad == 0] = 1e-8
    return dict(mean=mean, std=std, median=median, mad=mad)


def zero_dead_channels(
    act_matrix: np.ndarray,
    baseline_stats: Dict[str, np.ndarray],
    channel_names: List[str],
    var_threshold: float = 1e-6,
    mad_threshold: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    act_matrix is the "activity" signal: abs(HPF(raw)) in our case.
    """
    variances = np.var(act_matrix, axis=0)
    mad = baseline_stats["mad"]

    dead_mask = (variances < var_threshold) | (mad < mad_threshold)
    alive_mask = ~dead_mask

    print("\n  STEP 3b: Dead-channel detection")
    for i, ch in enumerate(channel_names):
        status = "DEAD  (set to zero)" if dead_mask[i] else "alive"
        print(f"    {ch:10s}  var={variances[i]:.3e}, MAD={mad[i]:.3e}  -> {status}")

    act_fixed = act_matrix.copy()
    act_fixed[:, dead_mask] = 0.0

    baseline_fixed: Dict[str, np.ndarray] = {}
    for key in baseline_stats:
        arr = baseline_stats[key].copy()
        arr[dead_mask] = 0.0
        baseline_fixed[key] = arr

    return act_fixed, baseline_fixed, alive_mask


def normalize_signals(act_matrix: np.ndarray, baseline_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Robust normalization:
        z(t) = (act(t) - median) / MAD
    """
    m0 = baseline_stats["median"]
    mad = baseline_stats["mad"]
    return (act_matrix - m0[None, :]) / mad[None, :]


def gate_z_only(z_matrix: np.ndarray, gate_eps: float) -> np.ndarray:
    """
    Gating only on z(t):
        if |z| < gate_eps -> z = 0.
    NOTE: we DO NOT modify act_matrix here; RMS uses the real amplitudes.
    """
    z_clean = z_matrix.copy()
    small_mask = np.abs(z_clean) < gate_eps
    z_clean[small_mask] = 0.0
    return z_clean


# ============================================================
# 5. PREPROCESSING (HPF + abs)
# ============================================================

def preprocess_signals(
    emg_matrix: np.ndarray,
    fs: float,
    params: PipelineParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        hp  : high-pass filtered EMG
        act : abs(hp) as activity signal
    """
    hp = butter_highpass(emg_matrix, fs, params.hp_cutoff_hz, order=2)
    act = np.abs(hp)
    return hp, act


# ============================================================
# 6. WINDOW FEATURES (RMS & mean z)
# ============================================================

def compute_window_features_rms_and_z(
    act_matrix: np.ndarray,
    z_matrix: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    params: PipelineParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_channels = act_matrix.shape

    window_samples = int(round(params.window_ms * 1e-3 * fs))
    overlap_samples = int(round(params.overlap_ms * 1e-3 * fs))
    hop_samples = max(1, window_samples - overlap_samples)

    idx = sliding_window_indices(n_samples, window_samples, hop_samples)
    if len(idx) == 0:
        raise ValueError("No sliding windows generated; check window_ms/overlap_ms vs signal length.")

    rms_features = np.zeros((len(idx), n_channels), dtype=float)
    mean_z = np.zeros((len(idx), n_channels), dtype=float)
    t_centers = np.zeros(len(idx), dtype=float)

    for k, (start, end) in enumerate(idx):
        seg_act = act_matrix[start:end, :]
        seg_z = z_matrix[start:end, :]
        rms_features[k, :] = np.sqrt(np.mean(seg_act ** 2, axis=0))
        mean_z[k, :] = np.nanmean(seg_z, axis=0)
        t_centers[k] = 0.5 * (time_s[start] + time_s[end - 1])

    return rms_features, mean_z, t_centers


# ============================================================
# 7. PER-CHANNEL RMS THRESHOLDS + GLOBAL ACTIVITY
# ============================================================

def compute_per_channel_rms_thresholds(
    rms_features: np.ndarray,
    t_centers: np.ndarray,
    rest_start: float,
    rest_end: float,
    alive_mask: np.ndarray,
    k: float = 2.0,
    perc: float = 85.0,
    min_rest_windows: int = 5,
) -> np.ndarray:
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

        thr_mad = med + k * mad_val
        thr_perc = np.nanpercentile(vals, perc)
        thresholds[i] = min(thr_mad, thr_perc)

    return thresholds


def per_channel_activity_rms(rms_features: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (rms_features > thresholds[None, :]).astype(int)


def detect_pulses_on_signal(
    signal_1d: np.ndarray,
    t_centers: np.ndarray,
    params: PipelineParams
) -> List[Tuple[int, int]]:
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

    # Merge pulses that are close
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = pulses[0]
    for s, e in pulses[1:]:
        gap = t_centers[s] - t_centers[cur_e]
        if gap <= merge_gap_sec:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def compute_pulse_metrics(
    signal_1d: np.ndarray,
    t_centers: np.ndarray,
    pulses: List[Tuple[int, int]]
) -> pd.DataFrame:
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

            pre_seg = seg[:peak_idx + 1]
            pre_t = t_seg[:peak_idx + 1]
            idx10 = idx90 = None
            for k in range(len(pre_seg)):
                if idx10 is None and pre_seg[k] >= level_10:
                    idx10 = k
                if pre_seg[k] >= level_90:
                    idx90 = k
                    break
            if idx10 is not None and idx90 is not None and idx90 > idx10:
                tr1 = float(pre_t[idx90] - pre_t[idx10])

            post_seg = seg[peak_idx:]
            post_t = t_seg[peak_idx:]
            idx90d = idx10d = None
            for k in range(len(post_seg)):
                if idx90d is None and post_seg[k] <= level_90:
                    idx90d = k
                if post_seg[k] <= level_10:
                    idx10d = k
                    break
            if idx90d is not None and idx10d is not None and idx10d > idx90d:
                tr2 = float(post_t[idx10d] - post_t[idx90d])

        rows.append(dict(
            pulse_index=j,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            peak_time=peak_time,
            peak_value=peak_val,
            tr1_rise_time=tr1,
            tr2_fall_time=tr2,
            auc=auc,
        ))

    return pd.DataFrame(rows)


# ============================================================
# 8. PIPELINE PER SHEET
# ============================================================

def run_pipeline_on_sheet(
    xlsx_path: Path,
    sheet_name: str,
    params: PipelineParams
) -> Dict[str, object]:
    print(f"\n========== SHEET: {sheet_name} ==========")
    print("STEP 1: Loading data and computing sampling rate...")

    time_s, emg, channel_names = load_sheet(xlsx_path, sheet_name)
    fs = compute_fs(time_s)
    nyq = 0.5 * fs

    print(f"  Fs  = {fs:.3f} Hz, Nyq = {nyq:.3f} Hz")
    print(f"  Duration = {time_s[-1] - time_s[0]:.3f} s")
    print(f"  Channels ({len(channel_names)}): {channel_names}")
    print_channel_stats("Raw EMG", emg, channel_names)

    # STEP 2: Preprocessing (HPF + abs)
    print("\nSTEP 2: Preprocessing (HPF + abs)...")
    hp, act_raw = preprocess_signals(emg, fs, params)
    print_channel_stats("High-pass filtered", hp, channel_names)
    print_channel_stats("Activity act_raw(t) = |HPF|", act_raw, channel_names)

    # STEP 3: Baseline & dead channels (on act_raw)
    print("\nSTEP 3: Baseline analysis from rest segment on activity...")
    rest = select_rest_segment(time_s, act_raw, params.rest_start_sec, params.rest_end_sec)
    baseline_stats = compute_baseline_stats(rest)
    print("  Baseline (median, MAD) per channel on act_raw:")
    for ch, med, mad_val in zip(channel_names, baseline_stats["median"], baseline_stats["mad"]):
        print(f"    {ch:10s} median={med:8.4f}, MAD={mad_val:8.4f}")

    act, baseline_stats, alive_mask = zero_dead_channels(act_raw, baseline_stats, channel_names)

    # STEP 4: z-normalization + gating on z only
    print("\nSTEP 4: Normalization (z-score) + gating on z(t)...")
    z_raw = normalize_signals(act, baseline_stats)
    print_channel_stats("z_raw(t)", z_raw, channel_names)
    z = gate_z_only(z_raw, gate_eps=params.gate_eps)
    print_channel_stats("z(t) after gating", z, channel_names)

    # STEP 5: Window features (RMS on act, mean z)
    print("\nSTEP 5: Windowing and feature extraction (RMS & mean z per window/channel)...")
    rms_features, mean_z, t_centers = compute_window_features_rms_and_z(act, z, time_s, fs, params)
    n_windows, n_channels = rms_features.shape
    print(f"  Windows: {n_windows} (window_ms={params.window_ms}, overlap_ms={params.overlap_ms})")
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
        perc=params.per_channel_perc,
    )
    for ch, th in zip(channel_names, rms_thresholds):
        print(f"    RMS threshold for {ch:10s}: {th:8.4f}")

    active_all = per_channel_activity_rms(rms_features, rms_thresholds)
    active_all[:, ~alive_mask] = 0
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

    # STEP 7: Global activity & pulses
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
        hp=hp,
        act=act,
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
# 9. PLOTS (stacked)
# ============================================================

def _plot_stacked(
    out_path: Path,
    title: str,
    time_s: np.ndarray,
    data: np.ndarray,
    channel_names: List[str],
    pulses: List[Tuple[int, int]] | None = None,
    t_centers: np.ndarray | None = None,
    ylabel: str = "Amplitude + offset",
):
    n_samples, n_channels = data.shape
    fig, ax = plt.subplots(figsize=(12, 6))

    max_abs = np.max(np.abs(data))
    offset = max_abs * 1.2 if max_abs > 0 else 1.0

    for i in range(n_channels):
        ax.plot(time_s, data[:, i] + i * offset, label=channel_names[i])

    if pulses is not None and t_centers is not None and len(t_centers) > 0:
        for (i_start, i_end) in pulses:
            ax.axvspan(t_centers[i_start], t_centers[i_end], color="red", alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_debug_stages(
    out_dir: Path,
    sheet_name: str,
    results: Dict[str, object]
) -> None:
    time_s = results["time"]
    emg = results["emg"]
    hp = results["hp"]
    act = results["act"]
    z = results["z"]
    rms_features = results["rms"]
    t_centers = results["t_centers"]
    frac_active = results["frac_active"]
    global_active = results["global_active"]
    pulses = results["pulses"]
    channel_names: List[str] = results["channel_names"]

    n_windows = len(t_centers)

    # 1) Raw EMG stacked
    _plot_stacked(
        out_dir / f"{sheet_name}_raw_stacked.png",
        f"Raw EMG (stacked) – {sheet_name}",
        time_s,
        emg,
        channel_names,
    )

    # 2) High-pass filtered stacked
    _plot_stacked(
        out_dir / f"{sheet_name}_hp_stacked.png",
        f"High-pass filtered EMG (stacked) – {sheet_name}",
        time_s,
        hp,
        channel_names,
    )

    # 3) Activity (abs(HPF)) stacked
    _plot_stacked(
        out_dir / f"{sheet_name}_act_stacked.png",
        f"Activity |HPF| per Channel (stacked) – {sheet_name}",
        time_s,
        act,
        channel_names,
    )

    # 4) z(t) stacked
    _plot_stacked(
        out_dir / f"{sheet_name}_z_stacked.png",
        f"Normalized z(t) after gating (stacked) – {sheet_name}",
        time_s,
        z,
        channel_names,
        ylabel="z-score + offset",
    )

    # 5) Raw EMG with global pulses shaded
    _plot_stacked(
        out_dir / f"{sheet_name}_raw_with_pulses.png",
        f"Raw EMG with Global Pulses (stacked) – {sheet_name}",
        time_s,
        emg,
        channel_names,
        pulses=pulses,
        t_centers=t_centers,
    )

    # 6) Window RMS per channel + global_active shading
    if n_windows > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(emg.shape[1]):
            ax.plot(t_centers, rms_features[:, i], alpha=0.7, label=channel_names[i])
        ax.set_title(f"Window RMS per Channel (on |HPF|) – {sheet_name}")
        ax.set_xlabel("Time [s] (window centers)")
        ax.set_ylabel("RMS")
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

    # 7) Global fraction of active channels + pulses
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
# 10. SAVE RESULTS
# ============================================================

def save_results_for_sheet(
    out_dir: Path,
    sheet_name: str,
    results: Dict[str, object]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = results["baseline_stats"]
    channel_names: List[str] = results["channel_names"]
    alive_mask: np.ndarray = results["alive_mask"]
    rms_thresholds: np.ndarray = results["rms_thresholds"]
    is_dead = (~alive_mask).astype(int)

    baseline_df = pd.DataFrame({
        "channel": channel_names,
        "mean_act": baseline["mean"],
        "std_act": baseline["std"],
        "median_act": baseline["median"],
        "mad_act": baseline["mad"],
        "is_dead": is_dead,
        "rms_threshold": rms_thresholds,
    })
    baseline_df.to_csv(out_dir / f"{sheet_name}_baseline.csv", index=False)

    metrics_df: pd.DataFrame = results["metrics_df"]
    metrics_df.to_csv(out_dir / f"{sheet_name}_pulses.csv", index=False)

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

    plot_debug_stages(out_dir, sheet_name, results)


# ============================================================
# 11. RUN WORKBOOK
# ============================================================

def run_workbook(
    xlsx_path: Path,
    out_dir: Path,
    params: PipelineParams
) -> None:
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
        description="EMG-like Movement Detection Pipeline (NO ENVELOPE: HPF + abs + RMS + consensus)"
    )
    parser.add_argument("--in_xlsx", type=str, required=True,
                        help="Path to input Excel file (e.g. Cüneyt_Yılmaz-01.xlsx)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for CSVs and plots.")

    parser.add_argument("--rest_start", type=float, default=0.0)
    parser.add_argument("--rest_end", type=float, default=3.0)
    parser.add_argument("--window_ms", type=float, default=200.0)
    parser.add_argument("--overlap_ms", type=float, default=100.0)

    parser.add_argument("--per_channel_k", type=float, default=2.0)
    parser.add_argument("--per_channel_perc", type=float, default=85.0)

    parser.add_argument("--gate_eps", type=float, default=0.3)

    parser.add_argument("--consensus_min_fraction", type=float, default=0.4)
    parser.add_argument("--pulse_on_frac", type=float, default=0.4)
    parser.add_argument("--pulse_off_frac", type=float, default=0.3)
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
        per_channel_perc=args.per_channel_perc,
        gate_eps=args.gate_eps,
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
