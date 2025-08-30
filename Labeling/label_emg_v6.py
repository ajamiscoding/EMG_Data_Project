#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG labeling pipeline — v6_relaxed (auto-threshold, less strict)
- Adaptive filtering (auto picks envelope mode for ~100 Hz)
- Per-channel RMS envelope + robust normalization
- Top-k fusion
- AUTO threshold via percentile scan + hysteresis
- Min-duration + gap-merge + optional median smoothing
- QC relaxed so we don't miss attempts

Recommended run for your file:
  python label_emg_v6_relaxed.py --in_xlsx 30Temmuz_Ampute_polyphase.xlsx --out_dir labels_out --force_mode envelope
"""

from __future__ import annotations
import argparse, logging, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# --------------------------- Config ---------------------------

@dataclass
class Params:
    # Filtering
    force_mode: str = "auto"          # auto | raw | mid | envelope
    notch_freq: float = 50.0
    bp_low: float = 20.0
    bp_high: float = 450.0
    env_highpass_hz: float = 0.5      # for envelope mode (remove drift)
    env_lowpass_hz: float = 6.0       # for envelope mode (smooth)

    # Envelope + fusion
    rms_win_ms: float = 300.0
    k_top: int = 4

    # AUTO threshold (RELAXED defaults)
    pct_start: float = 0.40           # scan percentiles 40%..92% (wider, more sensitive)
    pct_end: float = 0.92
    pct_steps: int = 50
    hysteresis_ratio: float = 0.70    # keep segments ON longer (less strict)

    # Desired activity band (fraction of total samples)
    active_frac_min: float = 0.02     # 2%
    active_frac_max: float = 0.50     # 50%

    # Temporal cleanup (RELAXED)
    min_on_ms: float = 220.0
    min_off_ms: float = 220.0
    merge_gap_ms: float = 350.0
    sample_median_win_ms: float = 150.0  # 0 = off

    # QC (RELAXED): allow weaker/more variable attempts
    peak_min_z: float = 0.9
    channels_at_peak_min_ratio: float = 0.25  # require >= 25% of channels >= 1.0 at peak

    # Columns
    time_column: str = "Time"
    channel_prefix: str = "emg"

# --------------------------- Helpers ---------------------------

def butter_filter(sig: np.ndarray, fs: float, ftype: str, cutoff, order: int = 4):
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        lo, hi = cutoff[0] / nyq, cutoff[1] / nyq
        if lo <= 0 or hi >= 1 or lo >= hi:
            return sig
        b, a = butter(order, [lo, hi], btype='band')
    else:
        wn = cutoff / nyq
        if wn <= 0 or wn >= 1:
            return sig
        b, a = butter(order, wn, btype=ftype)
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def notch_filter(sig: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    w0 = f0 / (fs / 2.0)
    if w0 >= 1.0 or w0 <= 0.0:
        return sig
    b, a = iirnotch(w0=w0, Q=Q)
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

def segments_from_binary(active_binary: np.ndarray, min_on: int, min_off: int, merge_gap: int):
    a = active_binary.astype(int)
    diff = np.diff(np.concatenate(([0], a, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_on]
    if not segs:
        return []
    # merge gaps
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    # enforce min_off
    final = [merged[0]]
    for s, e in merged[1:]:
        ps, pe = final[-1]
        if s - pe < min_off:
            final[-1] = (ps, e)
        else:
            final.append((s, e))
    return final

def infer_sampling_rate(time_values: np.ndarray):
    if time_values is None or len(time_values) < 2:
        return 1000.0, "ms"
    dt = float(np.median(np.diff(time_values.astype(float))))
    if dt > 1.0:
        return 1000.0 / dt, "ms"
    return 1.0 / dt, "s"

def collect_emg_columns(df: pd.DataFrame, prefix: str = "emg"):
    return [c for c in df.columns if str(c).lower().startswith(prefix.lower())]

def decide_mode(fs: float, force_mode: str = "auto") -> str:
    if force_mode in {"raw", "mid", "envelope"}:
        return force_mode
    if fs >= 1000.0: return "raw"
    if fs >= 200.0:  return "mid"
    return "envelope"

def stack_plot(y: np.ndarray, t: np.ndarray, title: str, save_path: Path, scale_per_channel: bool = True):
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
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def median_smooth_binary(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    sm = uniform_filter1d(x.astype(float), size=win, mode='nearest')
    return (sm >= 0.5).astype(int)

# --------------------------- Auto-thresholding ---------------------------

def binary_from_threshold(x: np.ndarray, thr_hi: float, thr_lo: float) -> np.ndarray:
    on = np.zeros(len(x), dtype=int); state = 0
    for i, v in enumerate(x):
        if state == 0:
            if v >= thr_hi: state = 1
        else:
            if v < thr_lo: state = 0
        on[i] = state
    return on

def pick_threshold_auto(fused: np.ndarray, fs: float, p: Params):
    lo_pct = int(round(p.pct_start * 100))
    hi_pct = int(round(p.pct_end * 100))
    percentiles = np.linspace(lo_pct, hi_pct, p.pct_steps)

    best = None
    min_on  = int(round(p.min_on_ms  * fs / 1000.0))
    min_off = int(round(p.min_off_ms * fs / 1000.0))
    merge_gap = int(round(p.merge_gap_ms * fs / 1000.0))
    sm_win = int(round((p.sample_median_win_ms/1000.0)*fs)) if p.sample_median_win_ms>0 else 0

    for pct in percentiles:
        thr_hi = np.percentile(fused, pct)
        thr_lo = p.hysteresis_ratio * thr_hi
        active = binary_from_threshold(fused, thr_hi, thr_lo)
        if sm_win > 1:
            active = median_smooth_binary(active, sm_win)
        segs = segments_from_binary(active, min_on, min_off, merge_gap)
        a_frac = active.mean()
        nseg = len(segs)

        inside = fused[active==1] if np.any(active==1) else np.array([0.0])
        outside = fused[active==0] if np.any(active==0) else np.array([0.0])
        contrast = float(inside.mean() - outside.mean())

        # score: prefer activity in band, decent contrast, not too many segments
        in_band = (p.active_frac_min <= a_frac <= p.active_frac_max)
        penalty = 0.0
        if not in_band:
            penalty += abs(a_frac - np.clip(a_frac, p.active_frac_min, p.active_frac_max))
        score = (2.0 * contrast) - (0.4 * nseg) - (4.0 * penalty)

        cand = dict(pct=float(pct), thr_hi=float(thr_hi), thr_lo=float(thr_lo),
                    a_frac=float(a_frac), nseg=int(nseg), score=float(score),
                    segs=segs, active=active)
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best

# --------------------------- Core per-sheet ---------------------------

def process_sheet(df: pd.DataFrame, sheet_name: str, fs: float, time_unit: str, p: Params, out_root: Path, mode: str):
    sheet_dir = out_root / sheet_name
    sheet_dir.mkdir(parents=True, exist_ok=True)

    time = df[p.time_column].values.astype(float)
    emg_cols = collect_emg_columns(df, p.channel_prefix)
    if not emg_cols:
        logging.warning(f"[{sheet_name}] No EMG columns with prefix '{p.channel_prefix}'")
        return []

    X_raw = df[emg_cols].values.astype(float)
    T, C = X_raw.shape

    # 0) Raw
    stack_plot(X_raw, time, f"{sheet_name} — RAW channels", sheet_dir / "00_raw_channels.png", scale_per_channel=True)

    # 1) Filtering (adaptive)
    X_filt = np.zeros_like(X_raw)
    for c in range(C):
        sig = X_raw[:, c].astype(float) - np.median(X_raw[:, c].astype(float))
        if mode in {"raw","mid"}:
            sig = notch_filter(sig, fs, f0=p.notch_freq, Q=30.0)
            high = min(p.bp_high, 0.45*fs)
            sig = butter_filter(sig, fs, 'bandpass', (p.bp_low, high), order=4)
        else:
            sig = butter_filter(sig, fs, 'highpass', p.env_highpass_hz, order=2)
            sig = butter_filter(sig, fs, 'lowpass',  p.env_lowpass_hz, order=4)
        X_filt[:, c] = sig
    stack_plot(X_filt, time, f"{sheet_name} — Filtered ({mode.upper()})", sheet_dir / "01_filtered_channels.png", scale_per_channel=True)

    # 2) Envelopes + normalization
    win = int(max(3, round(p.rms_win_ms * fs / 1000.0)))
    envs = np.zeros_like(X_filt)
    envs_norm = np.zeros_like(X_filt)
    for c in range(C):
        env = rms_envelope(X_filt[:, c], win)
        envs[:, c] = env
        env_n, _, _ = robust_normalize(env)
        envs_norm[:, c] = env_n
    stack_plot(envs, time, f"{sheet_name} — Envelopes (RMS)", sheet_dir / "02_envelopes_rms.png", scale_per_channel=True)
    stack_plot(envs_norm, time, f"{sheet_name} — Envelopes normalized", sheet_dir / "03_envelopes_normalized.png", scale_per_channel=False)

    # 3) Fuse channels
    fused = fuse_topk(envs_norm, k=min(p.k_top, C))
    plt.figure(figsize=(12,3)); plt.plot(time, fused, linewidth=1.0)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Fused (z)"); plt.title(f"{sheet_name} — Fused signal")
    plt.tight_layout(); plt.savefig(sheet_dir / "04_fused_signal.png"); plt.close()

    # 4) AUTO threshold selection (relaxed)
    best = pick_threshold_auto(fused, fs, p)
    active = best["active"]; segs = best["segs"]; thr_hi, thr_lo = best["thr_hi"], best["thr_lo"]

    # 5) QC relaxed: allow weaker, multi-channel but lower bar
    min_on  = int(round(p.min_on_ms  * fs / 1000.0))
    max_duration_ms = 10000.0
    max_samp = int(round(max_duration_ms * fs / 1000.0))
    need_channels = max(1, int(np.ceil(p.channels_at_peak_min_ratio * C)))

    segs_qc = []
    envs_norm_check = envs_norm
    fused_check = fused
    for (s, e) in segs:
        dur = e - s
        if dur < min_on or dur > max_samp:
            continue
        seg_fused = fused_check[s:e]
        if len(seg_fused) == 0: continue
        peak_idx = s + int(np.argmax(seg_fused))
        if fused_check[peak_idx] < p.peak_min_z:  # relaxed
            continue
        elevated = (envs_norm_check[peak_idx, :] >= 1.0).sum()
        if elevated < need_channels:
            continue
        segs_qc.append((s, e))
    segs = segs_qc

    # 6) Plots
    plt.figure(figsize=(12,3))
    plt.plot(time, fused, linewidth=1.0, label="fused")
    plt.axhline(thr_hi, linestyle="--", linewidth=1.0, label=f"thr_hi={thr_hi:.2f}")
    plt.axhline(thr_lo, linestyle=":",  linewidth=1.0, label=f"thr_lo={thr_lo:.2f}")
    plt.legend(loc="upper right"); plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Fused (z)")
    plt.title(f"{sheet_name} — Auto thresholds (relaxed)")
    plt.tight_layout(); plt.savefig(sheet_dir / "05_auto_thresholds.png"); plt.close()

    plt.figure(figsize=(12,2.5)); plt.step(time, active, where="post")
    plt.ylim(-0.2, 1.2); plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Active")
    plt.title(f"{sheet_name} — Active (binary, auto)")
    plt.tight_layout(); plt.savefig(sheet_dir / "06_active_binary.png"); plt.close()

    plt.figure(figsize=(12,3))
    plt.plot(time, fused, linewidth=1.0)
    for (s, e) in segs:
        plt.axvspan(time[s], time[e-1], alpha=0.2)
    plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Fused (z)")
    plt.title(f"{sheet_name} — Final segments (auto, relaxed)")
    plt.tight_layout(); plt.savefig(sheet_dir / "07_segments_overlay.png"); plt.close()

    # CSV
    pd.DataFrame({"Time": time, "Fused_z": fused, "Active_binary": active}).to_csv(
        sheet_dir / "track_fused_active.csv", index=False
    )

    rows = []
    for i, (s, e) in enumerate(segs, 1):
        t0 = float(time[s]); t1 = float(time[min(e-1, len(time)-1)])
        rows.append((sheet_name, i, t0, t1, t1 - t0))
    return rows

# --------------------------- Main ---------------------------

def main(in_xlsx: Path, out_dir: Path, p: Params):
    out_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(in_xlsx)
    sheets = xls.sheet_names

    df0 = pd.read_excel(in_xlsx, sheet_name=sheets[0])
    fs, time_unit = infer_sampling_rate(df0[p.time_column].values.astype(float))
    mode = decide_mode(fs, p.force_mode)
    logging.info(f"Inferred sampling rate: {fs:.3f} Hz, time unit: {time_unit}, mode: {mode.upper()}")

    all_rows = []
    for sheet in sheets:
        logging.info(f"Processing sheet: {sheet}")
        df = pd.read_excel(in_xlsx, sheet_name=sheet)
        all_rows += process_sheet(df, sheet, fs, time_unit, p, out_dir, mode)

    duration_name = f"Duration_{time_unit}"
    res = pd.DataFrame(all_rows, columns=["Sheet","Attempt_ID","Start_time","End_time",duration_name])
    res = res.sort_values(["Sheet","Attempt_ID"]).reset_index(drop=True)
    res.to_csv(out_dir / "detected_attempts.csv", index=False)
    logging.info(f"Saved CSV: {out_dir/'detected_attempts.csv'}")
    logging.info(f"Saved plots under: {out_dir.resolve()}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Auto-label EMG attempts (relaxed).")
    ap.add_argument("--in_xlsx", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="labels_out")
    ap.add_argument("--force_mode", type=str, default="auto", choices=["auto","raw","mid","envelope"])
    # minimal exposed knobs (you can ignore them)
    ap.add_argument("--rms_win_ms", type=float, default=300.0)
    ap.add_argument("--env_highpass_hz", type=float, default=0.5)
    ap.add_argument("--env_lowpass_hz", type=float, default=6.0)
    # strictness controls (OPTIONAL to override)
    ap.add_argument("--min_on_ms", type=float, default=220.0)
    ap.add_argument("--min_off_ms", type=float, default=220.0)
    ap.add_argument("--merge_gap_ms", type=float, default=350.0)
    ap.add_argument("--sample_median_win_ms", type=float, default=150.0)
    ap.add_argument("--peak_min_z", type=float, default=0.9)
    ap.add_argument("--channels_at_peak_min_ratio", type=float, default=0.25)
    ap.add_argument("--pct_start", type=float, default=0.40)
    ap.add_argument("--pct_end", type=float, default=0.92)
    ap.add_argument("--pct_steps", type=int, default=50)
    ap.add_argument("--hysteresis_ratio", type=float, default=0.70)
    ap.add_argument("--active_frac_min", type=float, default=0.02)
    ap.add_argument("--active_frac_max", type=float, default=0.50)
    ap.add_argument("--k_top", type=int, default=4)
    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    p = Params(
        force_mode=args.force_mode,
        rms_win_ms=args.rms_win_ms,
        env_highpass_hz=args.env_highpass_hz,
        env_lowpass_hz=args.env_lowpass_hz,
        min_on_ms=args.min_on_ms,
        min_off_ms=args.min_off_ms,
        merge_gap_ms=args.merge_gap_ms,
        sample_median_win_ms=args.sample_median_win_ms,
        peak_min_z=args.peak_min_z,
        channels_at_peak_min_ratio=args.channels_at_peak_min_ratio,
        pct_start=args.pct_start,
        pct_end=args.pct_end,
        pct_steps=args.pct_steps,
        hysteresis_ratio=args.hysteresis_ratio,
        active_frac_min=args.active_frac_min,
        active_frac_max=args.active_frac_max,
        k_top=args.k_top,
    )
    main(Path(args.in_xlsx), Path(args.out_dir), p)
