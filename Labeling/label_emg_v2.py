#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG labeling pipeline (single-file version, clean code) — v2
Fix: compute waveform length per window directly (no broadcasting issues).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------------------------------------------------------
# Helper functions (signal processing)
# ----------------------------------------------------------------------------

def notch_filter(sig: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    w0 = f0 / (fs / 2.0)
    if w0 >= 1.0 or w0 <= 0.0:
        return sig
    b, a = iirnotch(w0=w0, Q=Q)
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
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
    rect = np.abs(sig)
    sq = rect ** 2
    avg = uniform_filter1d(sq, size=max(3, win_samples), mode='nearest')
    return np.sqrt(np.maximum(avg, 0.0))

def robust_normalize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    n = len(x)
    if n == 0:
        return x, 0.0, 1.0
    sorted_x = np.sort(x)
    k = max(1, int(0.2 * n))
    baseline = float(np.median(sorted_x[:k]))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(max(q75 - q25, 1e-9))
    return (x - baseline) / iqr, baseline, iqr

def fuse_topk(mat: np.ndarray, k: int = 4) -> np.ndarray:
    k = max(1, min(k, mat.shape[1]))
    sorted_vals = np.sort(mat, axis=1)
    return np.mean(sorted_vals[:, -k:], axis=1)

def teager_kaiser_energy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if len(x) < 3:
        return np.zeros_like(x)
    tke = np.zeros_like(x)
    tke[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tke, 0.0)

def window_indices(n: int, win: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < win:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    ends = starts + win
    return starts, ends

def segments_from_binary(active_binary: np.ndarray, min_on: int, min_off: int, merge_gap: int) -> List[Tuple[int, int]]:
    a = active_binary.astype(int)
    diff = np.diff(np.concatenate(([0], a, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_on]
    if not segs:
        return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    final = [merged[0]]
    for s, e in merged[1:]:
        ps, pe = final[-1]
        if s - pe < min_off:
            final[-1] = (ps, e)
        else:
            final.append((s, e))
    return final

# ----------------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------------

def infer_sampling_rate(time_values: np.ndarray) -> Tuple[float, str]:
    if time_values is None or len(time_values) < 2:
        return 1000.0, "ms"
    dt = float(np.median(np.diff(time_values.astype(float))))
    if dt > 1.0:
        return 1000.0 / dt, "ms"
    return 1.0 / dt, "s"

def collect_emg_columns(df: pd.DataFrame, prefix: str = "emg") -> List[str]:
    cols = [c for c in df.columns if str(c).lower().startswith(prefix.lower())]
    if not cols:
        raise ValueError(f"No EMG columns found with prefix '{prefix}'.")
    return cols

def preprocess_sheet(df: pd.DataFrame, fs: float, p: Params):
    time = df[p.time_column].values.astype(float)
    emg_cols = collect_emg_columns(df, p.channel_prefix)
    X = df[emg_cols].values.astype(float)  # T x C
    T, C = X.shape
    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))

    envs_norm = np.zeros_like(X)
    for c in range(C):
        sig = X[:, c].astype(float)
        sig -= np.median(sig)
        sig = notch_filter(sig, fs, f0=p.notch_freq, Q=30.0)
        sig = bandpass_filter(sig, fs, p.bp_low, p.bp_high, order=4)
        env = rms_envelope(sig, win)
        env_norm, _, _ = robust_normalize(env)
        envs_norm[:, c] = env_norm

    fused = fuse_topk(envs_norm, k=min(p.k_top, C))
    return envs_norm, fused, time

def build_features_on_fused(fused: np.ndarray, fs: float, p: Params):
    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))
    hop = max(1, int(round(win * p.hop_fraction)))
    starts, ends = window_indices(len(fused), win, hop)
    if len(starts) == 0:
        return np.zeros((0, 3)), starts, ends

    feats = np.zeros((len(starts), 3), dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends)):
        w = fused[s:e]
        # RMS
        rms_w = float(np.sqrt(np.mean(w ** 2))) if len(w) > 0 else 0.0
        # TKEO over the window directly
        tke_w = float(np.mean(teager_kaiser_energy(w))) if len(w) > 0 else 0.0
        # Waveform Length directly on the window
        wl_w = float(np.sum(np.abs(np.diff(w)))) if len(w) > 1 else 0.0
        feats[i] = [rms_w, tke_w, wl_w]

    feats = np.log1p(np.maximum(feats, 0.0))
    return feats, starts, ends

def cluster_active_windows(feats: np.ndarray, p: Params):
    if len(feats) == 0:
        return np.array([], dtype=int), 1
    km = KMeans(n_clusters=p.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(feats)
    means = [feats[labels == k, 0].mean() if np.any(labels == k) else -1e9 for k in range(p.n_clusters)]
    active_label = int(np.argmax(means))
    return labels, active_label

def map_window_labels_to_samples(labels, active_label, starts, ends, total_len):
    active = np.zeros(total_len, dtype=int)
    for lab, s, e in zip(labels, starts, ends):
        if lab == active_label:
            active[s:e] = 1
    return active

def quality_check_segments(segs, fused, envs_norm, fs, p: Params):
    if not segs:
        return []
    min_samp = int(round(p.min_on_ms * fs / 1000.0))
    max_duration_ms = 10000.0
    max_samp = int(round(max_duration_ms * fs / 1000.0))

    out = []
    for (s, e) in segs:
        dur = e - s
        if dur < min_samp or dur > max_samp:
            continue
        seg = fused[s:e]
        if len(seg) == 0:
            continue
        peak_idx = s + int(np.argmax(seg))
        if fused[peak_idx] < p.peak_min_z:
            continue
        # require a few channels to be elevated at the peak
        elevated = (envs_norm[peak_idx, :] >= 1.0).sum()
        if elevated < p.channels_at_peak_min:
            continue
        out.append((s, e))
    return out

def process_sheet(df, sheet_name, fs, time_unit, p: Params, out_dir: Path):
    envs_norm, fused, time = preprocess_sheet(df, fs, p)
    feats, starts, ends = build_features_on_fused(fused, fs, p)
    labels, active_label = cluster_active_windows(feats, p)
    active = map_window_labels_to_samples(labels, active_label, starts, ends, len(fused))

    min_on = int(round(p.min_on_ms * fs / 1000.0))
    min_off = int(round(p.min_off_ms * fs / 1000.0))
    merge_gap = int(round(p.merge_gap_ms * fs / 1000.0))
    segs = segments_from_binary(active, min_on, min_off, merge_gap)

    segs = quality_check_segments(segs, fused, envs_norm, fs, p)

    # plot
    plt.figure(figsize=(10,3))
    plt.plot(time, fused, linewidth=1)
    for (s, e) in segs:
        plt.axvspan(time[s], time[e-1], alpha=0.2)
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Fused envelope (norm)")
    plt.title(f"{sheet_name}: detected attempts")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sheet_name}_attempts.png")
    plt.close()

    rows = []
    for i, (s, e) in enumerate(segs, 1):
        t0 = float(time[s]); t1 = float(time[min(e-1, len(time)-1)])
        rows.append((sheet_name, i, t0, t1, t1 - t0))
    return rows

def main(in_xlsx: Path, out_dir: Path, p: Params):
    out_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(in_xlsx)
    sheets = xls.sheet_names

    df0 = pd.read_excel(in_xlsx, sheet_name=sheets[0])
    fs, time_unit = infer_sampling_rate(df0[p.time_column].values.astype(float))
    logging.info(f"Inferred sampling rate: {fs:.3f} Hz, time unit: {time_unit}")

    all_rows = []
    for sheet in sheets:
        logging.info(f"Processing sheet: {sheet}")
        df = pd.read_excel(in_xlsx, sheet_name=sheet)
        all_rows += process_sheet(df, sheet, fs, time_unit, p, out_dir)

    duration_name = f"Duration_{time_unit}"
    res = pd.DataFrame(all_rows, columns=["Sheet","Attempt_ID","Start_time","End_time",duration_name])
    res = res.sort_values(["Sheet","Attempt_ID"]).reset_index(drop=True)
    res.to_csv(out_dir / "detected_attempts.csv", index=False)
    logging.info(f"Saved CSV: {out_dir/'detected_attempts.csv'}")
    logging.info(f"Saved plots under: {out_dir.resolve()}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Label EMG attempts per sheet using unsupervised clustering on features.")
    ap.add_argument("--in_xlsx", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="labels_out")
    ap.add_argument("--rms_win_ms", type=float, default=200.0)
    ap.add_argument("--hop_fraction", type=float, default=0.5)
    ap.add_argument("--min_on_ms", type=float, default=150.0)
    ap.add_argument("--min_off_ms", type=float, default=200.0)
    ap.add_argument("--merge_gap_ms", type=float, default=250.0)
    ap.add_argument("--k_top", type=int, default=4)
    ap.add_argument("--n_clusters", type=int, default=2)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--bp_low", type=float, default=20.0)
    ap.add_argument("--bp_high", type=float, default=450.0)
    ap.add_argument("--peak_min_z", type=float, default=1.0)
    ap.add_argument("--channels_at_peak_min", type=int, default=3)
    ap.add_argument("--time_column", type=str, default="Time")
    ap.add_argument("--channel_prefix", type=str, default="emg")
    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    p = Params(
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
        time_column=args.time_column,
        channel_prefix=args.channel_prefix,
        peak_min_z=args.peak_min_z,
        channels_at_peak_min=args.channels_at_peak_min,
    )
    main(Path(args.in_xlsx), Path(args.out_dir), p)
