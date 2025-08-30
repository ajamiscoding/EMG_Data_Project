#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG labeling pipeline (single-file version, clean code)
======================================================

This script takes an Excel file where each sheet corresponds to a movement class.
Each sheet must have a "Time" column and 8 EMG channels named like "emg1".."emg8".

Pipeline (simple English):
1) Preprocess EMG per channel:
   - notch 50 Hz (mains), band-pass 20–450 Hz (muscle band)
   - rectify and RMS envelope (200 ms window)
2) Robust normalization per channel:
   - baseline = median of lowest 20%
   - scale = IQR (75th - 25th percentile)
   - x_norm = (x - baseline) / (IQR + 1e-9)
3) Fuse channels at each time by averaging the top-4 values (robust to noise)
4) Window features on the fused signal (200 ms, 50% overlap):
   - RMS, Teager–Kaiser energy (TKEO), Waveform Length (WL)
5) K-Means clustering (k=2) on features → REST vs ACTIVE (higher RMS = ACTIVE)
6) Map window labels back to samples, then clean with min-on/off and gap merge
7) Quality checks: drop segments that are too short/long, require a clear peak
8) Save a CSV with start/end times per attempt and a plot per sheet

Usage:
------
python label_emg.py --in_xlsx 30Temmuz_Ampute_polyphase.xlsx --out_dir labels_out

Extra options (see --help):
  --rms_win_ms 200         # RMS window in ms
  --min_on_ms 150          # minimum active duration to keep
  --min_off_ms 200         # minimum rest duration between attempts
  --merge_gap_ms 250       # merge active parts if gap < this
  --k_top 4                # number of top channels to average for fusion
  --n_clusters 2           # clustering groups (2 = rest/active)
  --notch_freq 50          # mains notch frequency (50 in TR)
  --bp_low 20 --bp_high 450  # band-pass cutoffs
  --peak_min_z 1.0         # required fused z at segment peak
  --channels_at_peak_min 3 # how many channels should be raised at the peak

"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# Config and dataclasses
# ----------------------------------------------------------------------------
    
@dataclass
class Params:
    rms_win_ms: float = 200.0
    hop_fraction: float = 0.5
    min_on_ms: float = 150.0
    min_off_ms: float = 200.0
    merge_gap_ms: float = 250.0
    k_top: int = 4
    n_clusters: int = 2
    notch_freq: float = 50.0
    bp_low: float = 20.0
    bp_high: float = 450.0
    peak_min_z: float = 1.0
    channels_at_peak_min: int = 3
    time_column: str = "Time"
    channel_prefix: str = "emg"

# ----------------------------------------------------------------------------
# Logging and warnings
# ----------------------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ----------------------------------------------------------------------------
# Helper functions (signal processing)
# ----------------------------------------------------------------------------

def notch_filter(sig: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Apply a notch filter at f0 Hz. If fs is too low, return input."""
    w0 = f0 / (fs / 2.0)
    if w0 >= 1.0 or w0 <= 0.0:
        return sig
    b, a = iirnotch(w0=w0, Q=Q)
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter. If invalid params, return input."""
    nyq = 0.5 * fs
    lowc, highc = low / nyq, high / nyq
    if lowc <= 0 or highc >= 1 or lowc >= highc:
        return sig
    b, a = butter(order, [lowc, highc], btype='band')
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def rms_envelope(sig: np.ndarray, win_samples: int) -> np.ndarray:
    """Full-wave rectify, then moving RMS using uniform_filter1d on squared signal."""
    rect = np.abs(sig)
    sq = rect ** 2
    avg = uniform_filter1d(sq, size=max(3, win_samples), mode='nearest')
    env = np.sqrt(np.maximum(avg, 0.0))
    return env

def robust_normalize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return normalized x, baseline, scale using robust stats (baseline=median of lowest 20%, scale=IQR)."""
    n = len(x)
    if n == 0:
        return x, 0.0, 1.0
    sorted_x = np.sort(x)
    k = max(1, int(0.2 * n))
    baseline = float(np.median(sorted_x[:k]))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(max(q75 - q25, 1e-9))
    x_norm = (x - baseline) / iqr
    return x_norm, baseline, iqr

def fuse_topk(mat: np.ndarray, k: int = 4) -> np.ndarray:
    """Fuse channels per time step by averaging the top-k values (robust to noisy/weak channels)."""
    if mat.ndim != 2:
        raise ValueError("mat must be 2D (T x C).")
    k = max(1, min(k, mat.shape[1]))
    sorted_vals = np.sort(mat, axis=1)
    topk = sorted_vals[:, -k:]
    fused = np.mean(topk, axis=1)
    return fused

def teager_kaiser_energy(x: np.ndarray) -> np.ndarray:
    """Teager–Kaiser energy operator (edge-safe)."""
    x = np.asarray(x)
    if len(x) < 3:
        return np.zeros_like(x)
    tke = np.zeros_like(x)
    tke[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tke, 0.0)

def rolling_waveform_length(x: np.ndarray, win: int) -> np.ndarray:
    """Rolling waveform length via cumulative sum of |diff|."""
    d = np.abs(np.diff(x, prepend=x[0]))
    cs = np.cumsum(d)
    wl = cs - np.concatenate(([0], cs[:-win]))
    wl[:win-1] = wl[win-1]
    return wl

def window_indices(n: int, win: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end indices for sliding windows (inclusive start, exclusive end)."""
    if n < win:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    ends = starts + win
    return starts, ends

def segments_from_binary(active_binary: np.ndarray, min_on: int, min_off: int, merge_gap: int) -> List[Tuple[int, int]]:
    """Convert a 0/1 array to segments, enforcing min_on/off and merging short gaps.
    Returns list of (start_index, end_index) with end exclusive.
    """
    a = active_binary.astype(int)
    diff = np.diff(np.concatenate(([0], a, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_on]
    if not segs:
        return []
    # merge gaps < merge_gap
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    # enforce min_off by merging too-close segments
    final = [merged[0]]
    for s, e in merged[1:]:
        ps, pe = final[-1]
        if s - pe < min_off:
            final[-1] = (ps, e)
        else:
            final.append((s, e))
    return final

# ----------------------------------------------------------------------------
# Core processing functions
# ----------------------------------------------------------------------------

def infer_sampling_rate(time_values: np.ndarray) -> Tuple[float, str]:
    """Infer sampling rate from a Time column. If the spacing looks like ms, return fs in Hz and 'ms'."""
    if time_values is None or len(time_values) < 2:
        return 1000.0, "ms"
    diffs = np.diff(time_values.astype(float))
    dt = float(np.median(diffs))
    if dt > 1.0:  # likely milliseconds
        fs = 1000.0 / dt
        return fs, "ms"
    else:
        fs = 1.0 / dt
        return fs, "s"

def collect_emg_columns(df: pd.DataFrame, prefix: str = "emg") -> List[str]:
    """Find EMG columns by prefix (case-insensitive)."""
    cols = [c for c in df.columns if str(c).lower().startswith(prefix.lower())]
    if not cols:
        raise ValueError(f"No EMG columns found with prefix '{prefix}'.")
    return cols

def preprocess_sheet(
    df: pd.DataFrame,
    fs: float,
    p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess all channels, returning: envs_norm (T x C), fused (T,), time (T,)"""
    time = df[p.time_column].values.astype(float)
    emg_cols = collect_emg_columns(df, p.channel_prefix)
    X = df[emg_cols].values.astype(float)  # T x C
    T, C = X.shape

    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))

    envs_norm = np.zeros_like(X)
    for c in range(C):
        sig = X[:, c].astype(float)
        sig = sig - np.median(sig)  # remove DC drift
        sig = notch_filter(sig, fs, f0=p.notch_freq, Q=30.0)
        sig = bandpass_filter(sig, fs, p.bp_low, p.bp_high, order=4)
        env = rms_envelope(sig, win)
        env_norm, _, _ = robust_normalize(env)
        envs_norm[:, c] = env_norm

    fused = fuse_topk(envs_norm, k=min(p.k_top, envs_norm.shape[1]))
    return envs_norm, fused, time

def build_features_on_fused(
    fused: np.ndarray,
    fs: float,
    p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build windowed features (RMS, TKEO, WL) on fused signal.
    Return features array (Nw x 3) and (starts, ends, active_init) where active_init is zeros.
    """
    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))
    hop = max(1, int(round(win * p.hop_fraction)))
    starts, ends = window_indices(len(fused), win, hop)
    if len(starts) == 0:
        return np.zeros((0, 3)), starts, ends, np.zeros_like(fused, dtype=int)

    tkeo = teager_kaiser_energy(fused)
    wl = rolling_waveform_length(fused, win)

    feats = np.zeros((len(starts), 3), dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends)):
        w = fused[s:e]
        rms_w = float(np.sqrt(np.mean(w ** 2))) if len(w) > 0 else 0.0
        tke_w = float(np.mean(tkeo[s:e])) if len(w) > 0 else 0.0
        wl_w = float(wl[e - 1] - (wl[s - 1] if s > 0 else 0.0))
        feats[i] = [rms_w, tke_w, wl_w]

    # Stabilize with log1p and clip to avoid negatives
    feats = np.log1p(np.maximum(feats, 0.0))
    return feats, starts, ends, np.zeros_like(fused, dtype=int)

def cluster_active_windows(
    feats: np.ndarray,
    p: Params
) -> Tuple[np.ndarray, int]:
    """Run KMeans and choose the cluster with higher RMS mean as ACTIVE. Return (labels, active_label)."""
    if len(feats) == 0:
        return np.array([], dtype=int), 1
    km = KMeans(n_clusters=p.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(feats)
    # decide active by the cluster with the higher RMS mean (feature 0)
    rms_means = []
    for k in range(p.n_clusters):
        if np.any(labels == k):
            rms_means.append(feats[labels == k, 0].mean())
        else:
            rms_means.append(-1e9)
    active_label = int(np.argmax(rms_means))
    return labels, active_label

def map_window_labels_to_samples(
    labels: np.ndarray,
    active_label: int,
    starts: np.ndarray,
    ends: np.ndarray,
    total_len: int
) -> np.ndarray:
    """Create a 0/1 array over samples from window labels."""
    active = np.zeros(total_len, dtype=int)
    for lab, s, e in zip(labels, starts, ends):
        if lab == active_label:
            active[s:e] = 1
    return active

def quality_check_segments(
    segs: List[Tuple[int, int]],
    fused: np.ndarray,
    envs_norm: np.ndarray,
    fs: float,
    p: Params,
    time_unit: str
) -> List[Tuple[int, int]]:
    """Filter segments with simple sanity checks."""
    if not segs:
        return []

    # Convert duration limits according to time unit
    min_samp = int(round(p.min_on_ms * fs / 1000.0))  # already enforced
    max_duration_ms = 10000.0  # 10 s default cap; adjust if needed
    max_samp = int(round(max_duration_ms * fs / 1000.0))

    filtered = []
    for (s, e) in segs:
        dur = e - s
        if dur < min_samp or dur > max_samp:
            continue

        # Check fused peak z
        seg_fused = fused[s:e]
        if len(seg_fused) == 0:
            continue
        peak_idx = s + int(np.argmax(seg_fused))
        if fused[peak_idx] < p.peak_min_z:
            continue

        # Require a few channels to be elevated at the peak
        if envs_norm.ndim == 2:
            elevated = (envs_norm[peak_idx, :] >= 1.0).sum()
            if elevated < p.channels_at_peak_min:
                continue

        filtered.append((s, e))

    return filtered

# ----------------------------------------------------------------------------
# Per-sheet processing and plotting
# ----------------------------------------------------------------------------

def process_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    fs: float,
    time_unit: str,
    p: Params,
    out_dir: Path
) -> List[Tuple[str, int, float, float, float]]:
    """Process one sheet and save its plot. Return list of rows for CSV."""
    # Preprocess → envs_norm, fused
    envs_norm, fused, time = preprocess_sheet(df, fs, p)

    # Features on fused
    feats, starts, ends, _ = build_features_on_fused(fused, fs, p)

    # Clustering → window labels
    labels, active_label = cluster_active_windows(feats, p)

    # Map window labels → samplewise active
    active = map_window_labels_to_samples(labels, active_label, starts, ends, len(fused))

    # Convert binary → raw segments
    min_on = int(round(p.min_on_ms * fs / 1000.0))
    min_off = int(round(p.min_off_ms * fs / 1000.0))
    merge_gap = int(round(p.merge_gap_ms * fs / 1000.0))
    segs = segments_from_binary(active, min_on=min_on, min_off=min_off, merge_gap=merge_gap)

    # QC filter
    segs = quality_check_segments(segs, fused, envs_norm, fs, p, time_unit)

    # Save plot
    fig = plt.figure(figsize=(10, 3))
    plt.plot(time, fused, linewidth=1)
    for (s, e) in segs:
        plt.axvspan(time[s], time[e-1], alpha=0.2)
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Fused envelope (norm)")
    plt.title(f"{sheet_name}: detected attempts")
    plt.tight_layout()
    plot_path = out_dir / f"{sheet_name}_attempts.png"
    plt.savefig(plot_path)
    plt.close(fig)

    # Build result rows
    rows: List[Tuple[str, int, float, float, float]] = []
    for i, (s, e) in enumerate(segs, start=1):
        t0 = float(time[s])
        t1 = float(time[min(e - 1, len(time) - 1)])
        dur = t1 - t0
        rows.append((sheet_name, i, t0, t1, dur))

    return rows

# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------

def main(in_xlsx: Path, out_dir: Path, p: Params) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(in_xlsx)
    sheet_names = xls.sheet_names

    # infer fs and time unit from first sheet
    df0 = pd.read_excel(in_xlsx, sheet_name=sheet_names[0])
    fs, time_unit = infer_sampling_rate(df0[p.time_column].values.astype(float))
    logging.info(f"Inferred sampling rate: {fs:.3f} Hz, time unit: {time_unit}")

    all_rows: List[Tuple[str, int, float, float, float]] = []
    for sheet in sheet_names:
        logging.info(f"Processing sheet: {sheet}")
        df = pd.read_excel(in_xlsx, sheet_name=sheet)
        rows = process_sheet(df, sheet, fs, time_unit, p, out_dir)
        all_rows.extend(rows)

    # save CSV
    duration_name = f"Duration_{time_unit}"
    res = pd.DataFrame(all_rows, columns=["Sheet", "Attempt_ID", "Start_time", "End_time", duration_name])
    res = res.sort_values(["Sheet", "Attempt_ID"]).reset_index(drop=True)
    csv_path = out_dir / "detected_attempts.csv"
    res.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")
    logging.info(f"Saved plots under: {out_dir.resolve()}")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Label EMG attempts per sheet using unsupervised clustering on features.")
    ap.add_argument("--in_xlsx", type=str, required=True, help="Path to input Excel file (sheets per class).")
    ap.add_argument("--out_dir", type=str, default="labels_out", help="Directory to save CSV and plots.")
    ap.add_argument("--rms_win_ms", type=float, default=200.0, help="RMS window in milliseconds.")
    ap.add_argument("--hop_fraction", type=float, default=0.5, help="Hop as fraction of window (e.g., 0.5).")
    ap.add_argument("--min_on_ms", type=float, default=150.0, help="Minimum active duration to keep (ms).")
    ap.add_argument("--min_off_ms", type=float, default=200.0, help="Minimum rest duration between attempts (ms).")
    ap.add_argument("--merge_gap_ms", type=float, default=250.0, help="Merge active parts if gap < this (ms).")
    ap.add_argument("--k_top", type=int, default=4, help="Top-k channels to average for fusion.")
    ap.add_argument("--n_clusters", type=int, default=2, help="Number of clusters (2 recommended).")
    ap.add_argument("--notch_freq", type=float, default=50.0, help="Notch frequency (50 for TR).")
    ap.add_argument("--bp_low", type=float, default=20.0, help="Band-pass low cutoff (Hz).")
    ap.add_argument("--bp_high", type=float, default=450.0, help="Band-pass high cutoff (Hz).")
    ap.add_argument("--peak_min_z", type=float, default=1.0, help="Min fused z-score at segment peak.")
    ap.add_argument("--channels_at_peak_min", type=int, default=3,
                    help="Min number of channels elevated (>=1.0) at peak.")
    ap.add_argument("--time_column", type=str, default="Time", help="Name of time column (default 'Time').")
    ap.add_argument("--channel_prefix", type=str, default="emg",
                    help="Prefix of EMG columns (default 'emg').")
    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    params = Params(
        rms_win_ms=args.rms_win_ms,
        hop_fraction=args.hop_fraction,
        min_on_ms=args.min_on_ms,
        min_off_ms=args.min_off_ms,
        merge_gap_ms=args.merge_gap_ms,
        k_top=args.k_top,
        n_clusters=args.n_clusters,
        notch_freq=args.notch_freq,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        peak_min_z=args.peak_min_z,
        channels_at_peak_min=args.channels_at_peak_min,
        time_column=args.time_column,
        channel_prefix=args.channel_prefix,
    )

    main(Path(args.in_xlsx), Path(args.out_dir), params)
