#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG labeling pipeline — v3 (with step-by-step plots & debug CSVs)
=================================================================
- Saves plots for: raw channels, filtered channels, per-channel envelopes,
  normalized envelopes, fused signal, features-over-time (RMS/TKEO/WL),
  window cluster scatter, active-binary over time, and final segments overlay.
- Saves per-window features CSV and active-binary CSV for debugging.

Usage:
  python label_emg_v3.py --in_xlsx 30Temmuz_Ampute_polyphase.xlsx --out_dir labels_out
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

# --------------------------- Config ---------------------------

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

# --------------------------- Helpers ---------------------------

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

def robust_normalize(x: np.ndarray):
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

def window_indices(n: int, win: int, hop: int):
    if n < win:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    ends = starts + win
    return starts, ends

def segments_from_binary(active_binary: np.ndarray, min_on: int, min_off: int, merge_gap: int):
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

# --------------------------- Core ---------------------------

def infer_sampling_rate(time_values: np.ndarray):
    if time_values is None or len(time_values) < 2:
        return 1000.0, "ms"
    dt = float(np.median(np.diff(time_values.astype(float))))
    if dt > 1.0:
        return 1000.0 / dt, "ms"
    return 1.0 / dt, "s"

def collect_emg_columns(df: pd.DataFrame, prefix: str = "emg"):
    cols = [c for c in df.columns if str(c).lower().startswith(prefix.lower())]
    if not cols:
        raise ValueError(f"No EMG columns found with prefix '{prefix}'.")
    return cols

def stack_plot(y: np.ndarray, t: np.ndarray, title: str, save_path: Path, scale_per_channel: bool = True):
    """
    Stack channels with vertical offsets for readability.
    y: T x C
    """
    T, C = y.shape
    plt.figure(figsize=(12, 6))
    for c in range(C):
        yc = y[:, c].astype(float)
        if scale_per_channel:
            s = np.std(yc) if np.std(yc) > 0 else 1.0
            yc = yc / s
        offset = c * 3.0
        plt.plot(t, yc + offset, linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("Channels (stacked)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_sheet(df: pd.DataFrame, sheet_name: str, fs: float, time_unit: str, p: Params, out_root: Path):
    # Make per-sheet dir
    sheet_dir = out_root / sheet_name
    sheet_dir.mkdir(parents=True, exist_ok=True)

    # Time and EMG
    time = df[p.time_column].values.astype(float)
    emg_cols = collect_emg_columns(df, p.channel_prefix)
    X_raw = df[emg_cols].values.astype(float)  # T x C
    T, C = X_raw.shape

    # Plot raw (stacked)
    stack_plot(X_raw, time, f"{sheet_name} — RAW channels", sheet_dir / "00_raw_channels.png", scale_per_channel=True)

    # Preprocess per channel
    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))
    X_filt = np.zeros_like(X_raw)     # filtered signals (post notch+bandpass)
    envs = np.zeros_like(X_raw)       # rms envelopes (pre-normalization)
    envs_norm = np.zeros_like(X_raw)  # normalized envelopes

    for c in range(C):
        sig = X_raw[:, c].astype(float)
        sig -= np.median(sig)
        sig = notch_filter(sig, fs, f0=p.notch_freq, Q=30.0)
        sig = bandpass_filter(sig, fs, p.bp_low, p.bp_high, order=4)
        X_filt[:, c] = sig

        env = rms_envelope(sig, win)
        envs[:, c] = env

        env_n, _, _ = robust_normalize(env)
        envs_norm[:, c] = env_n

    # Plot filtered (stacked)
    stack_plot(X_filt, time, f"{sheet_name} — Filtered channels", sheet_dir / "01_filtered_channels.png", scale_per_channel=True)

    # Plot envelopes (stacked)
    stack_plot(envs, time, f"{sheet_name} — Envelopes (RMS)", sheet_dir / "02_envelopes_rms.png", scale_per_channel=True)

    # Plot normalized envelopes (stacked)
    stack_plot(envs_norm, time, f"{sheet_name} — Normalized envelopes", sheet_dir / "03_envelopes_normalized.png", scale_per_channel=False)

    # Fuse channels
    fused = fuse_topk(envs_norm, k=min(p.k_top, C))

    # Plot fused
    plt.figure(figsize=(12, 3))
    plt.plot(time, fused, linewidth=1.0)
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Fused envelope (norm)")
    plt.title(f"{sheet_name} — Fused signal")
    plt.tight_layout()
    plt.savefig(sheet_dir / "04_fused_signal.png")
    plt.close()

    # Build windowed features
    hop = max(1, int(round(win * p.hop_fraction)))
    starts, ends = window_indices(T, win, hop)

    feats = np.zeros((len(starts), 3), dtype=float)  # RMS, TKEO, WL
    centers_time = np.zeros(len(starts), dtype=float)

    for i, (s, e) in enumerate(zip(starts, ends)):
        w = fused[s:e]
        t0 = time[s]; t1 = time[e-1] if e-1 < len(time) else time[-1]
        centers_time[i] = (t0 + t1) / 2.0
        rms_w = float(np.sqrt(np.mean(w ** 2))) if len(w) > 0 else 0.0
        tke_w = float(np.mean(teager_kaiser_energy(w))) if len(w) > 0 else 0.0
        wl_w  = float(np.sum(np.abs(np.diff(w)))) if len(w) > 1 else 0.0
        feats[i] = [rms_w, tke_w, wl_w]

    feats = np.log1p(np.maximum(feats, 0.0))

    # Plot features over time (separate plots to avoid subplots)
    plt.figure(figsize=(12,3))
    plt.plot(centers_time, feats[:,0], linewidth=1.0)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("RMS (log1p)")
    plt.title(f"{sheet_name} — Feature over time: RMS")
    plt.tight_layout()
    plt.savefig(sheet_dir / "05_feat_time_rms.png")
    plt.close()

    plt.figure(figsize=(12,3))
    plt.plot(centers_time, feats[:,1], linewidth=1.0)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("TKEO (log1p)")
    plt.title(f"{sheet_name} — Feature over time: TKEO")
    plt.tight_layout()
    plt.savefig(sheet_dir / "06_feat_time_tkeo.png")
    plt.close()

    plt.figure(figsize=(12,3))
    plt.plot(centers_time, feats[:,2], linewidth=1.0)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("WL (log1p)")
    plt.title(f"{sheet_name} — Feature over time: WL")
    plt.tight_layout()
    plt.savefig(sheet_dir / "07_feat_time_wl.png")
    plt.close()

    # Cluster windows
    if len(feats) == 0:
        labels = np.array([], dtype=int)
        active_label = 1
    else:
        km = KMeans(n_clusters=p.n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(feats)
        means = [feats[labels==k,0].mean() if np.any(labels==k) else -1e9 for k in range(p.n_clusters)]
        active_label = int(np.argmax(means))

    # Plot cluster scatter (RMS vs TKEO)
    if len(feats) > 0:
        plt.figure(figsize=(6,5))
        plt.scatter(feats[:,0], feats[:,1], s=8, c=labels)
        plt.xlabel("RMS (log1p)")
        plt.ylabel("TKEO (log1p)")
        plt.title(f"{sheet_name} — Window clusters (color = cluster)")
        plt.tight_layout()
        plt.savefig(sheet_dir / "08_clusters_scatter_rms_tkeo.png")
        plt.close()

    # Map window labels to sample-wise active
    active = np.zeros(T, dtype=int)
    for lab, (s, e) in zip(labels, zip(starts, ends)):
        if lab == active_label:
            active[s:e] = 1

    # Plot active-binary over time
    plt.figure(figsize=(12,2.5))
    plt.step(time, active, where="post")
    plt.ylim(-0.2, 1.2)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Active")
    plt.title(f"{sheet_name} — Active (binary) from clusters")
    plt.tight_layout()
    plt.savefig(sheet_dir / "09_active_binary.png")
    plt.close()

    # Convert to segments + QC
    min_on  = int(round(p.min_on_ms  * fs / 1000.0))
    min_off = int(round(p.min_off_ms * fs / 1000.0))
    merge_gap = int(round(p.merge_gap_ms * fs / 1000.0))
    segs = segments_from_binary(active, min_on, min_off, merge_gap)

    # QC
    # (drop too short/long; require fused peak >= peak_min_z and >= channels_at_peak_min channels >= 1.0 at the peak)
    segs_qc = []
    max_duration_ms = 10000.0
    max_samp = int(round(max_duration_ms * fs / 1000.0))
    for (s, e) in segs:
        dur = e - s
        if dur < min_on or dur > max_samp:
            continue
        seg_fused = fused[s:e]
        if len(seg_fused) == 0:
            continue
        peak_idx = s + int(np.argmax(seg_fused))
        if fused[peak_idx] < p.peak_min_z:
            continue
        elevated = (envs_norm[peak_idx, :] >= 1.0).sum()
        if elevated < p.channels_at_peak_min:
            continue
        segs_qc.append((s, e))
    segs = segs_qc

    # Plot segments overlay on fused
    plt.figure(figsize=(12,3))
    plt.plot(time, fused, linewidth=1.0)
    for (s, e) in segs:
        plt.axvspan(time[s], time[e-1], alpha=0.2)
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Fused envelope (norm)")
    plt.title(f"{sheet_name} — Final segments")
    plt.tight_layout()
    plt.savefig(sheet_dir / "10_segments_overlay.png")
    plt.close()

    # Save debug CSVs
    # 1) features per window
    feat_df = pd.DataFrame({
        "start_time": time[starts] if len(starts)>0 else [],
        "end_time": time[np.minimum(ends-1, len(time)-1)] if len(ends)>0 else [],
        "center_time": centers_time if len(starts)>0 else [],
        "RMS_log1p": feats[:,0] if len(feats)>0 else [],
        "TKEO_log1p": feats[:,1] if len(feats)>0 else [],
        "WL_log1p": feats[:,2] if len(feats)>0 else [],
        "cluster": labels if len(feats)>0 else [],
        "is_active_cluster": (labels==active_label).astype(int) if len(feats)>0 else []
    })
    feat_df.to_csv(sheet_dir / "features_windows.csv", index=False)

    # 2) active-binary per sample
    act_df = pd.DataFrame({"Time": time, "Active": active})
    act_df.to_csv(sheet_dir / "active_binary.csv", index=False)

    # Return final rows
    rows = []
    for i, (s, e) in enumerate(segs, 1):
        t0 = float(time[s]); t1 = float(time[min(e-1, len(time)-1)])
        rows.append((sheet_name, i, t0, t1, t1 - t0))
    return rows

def main(in_xlsx: Path, out_dir: Path, p: Params):
    out_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(in_xlsx)
    sheets = xls.sheet_names

    # infer fs and time unit
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
    ap = argparse.ArgumentParser(description="Label EMG attempts per sheet + save step-by-step plots.")
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
