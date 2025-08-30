"""
Multi-sheet movement detection & labeling pipeline
-------------------------------------------------
- Loads an Excel workbook with multiple sheets.
- Profiles each sheet (timing, sampling rate, basic integrity).
- Preprocesses signals (light smoothing, robust normalization).
- Detects movement attempts per channel with hysteresis + rules.
- Fuses channels (vote) to create final labels per sheet.
- Exports:
    * labels_{sheet}.csv              # fused movement attempts (start/end in seconds)
    * channel_metrics_{sheet}.csv     # per-channel stats & thresholds
    * sheets_profile.csv              # one-row summary per sheet

Dependencies: pandas, numpy, (optional) matplotlib for plots.

Usage (edit the paths at the bottom):
    python multi_sheet_labeling.py

Notes:
- This code assumes each sheet contains a time column (e.g., 'Time').
- Signals are expected to be envelope-like EMG or low-bandwidth features (<= 20 Hz).
- If you have raw EMG, compute an envelope first (rectify + low-pass) before running detection.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # optional
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# -------------------------
# Utility helpers
# -------------------------

def find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first likely time column name, or None if not found."""
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["time", "timestamp", "seconds", "sec", "ms", "s "]):
            candidates.append(c)
    # Prefer exact common names
    for pref in ["Time", "time", "Timestamp", "timestamp"]:
        if pref in df.columns:
            return pref
    return candidates[0] if candidates else None


def to_seconds(series: pd.Series) -> Optional[pd.Series]:
    """Convert a time-like series to seconds since start. Returns None if not possible."""
    if np.issubdtype(series.dtype, np.datetime64):
        s = (series - series.iloc[0]).dt.total_seconds()
        return s
    # try numeric
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().mean() < 0.8:
        return None
    diffs = s.diff().dropna()
    rng = s.max() - s.min()
    # Heuristics: if values look like milliseconds or ticks, convert to seconds
    if rng > 1e6 or (diffs.median() and diffs.median() > 1 and s.max() > 10000):
        return (s - s.iloc[0]) / 1000.0
    return s - s.iloc[0]


def estimate_fs(tsec: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Estimate sampling rate from time in seconds. Returns (fs, dt_median, dt_std)."""
    diffs = np.diff(np.asarray(tsec, dtype=float))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None, None, None
    dt_med = float(np.median(diffs))
    dt_std = float(np.std(diffs))
    fs = 1.0 / dt_med if dt_med > 0 else None
    return fs, dt_med, dt_std


def moving_average_centered(x: np.ndarray, win_samples: int) -> np.ndarray:
    """Centered moving average (zero-lag) using convolution."""
    win_samples = max(1, int(win_samples))
    kernel = np.ones(win_samples, dtype=float) / float(win_samples)
    # mode='same' keeps alignment; this is effectively zero-phase for a symmetric kernel
    y = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode='same')
    return y


def robust_median_mad(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    # avoid zero MAD (flat signal): use small epsilon
    if mad == 0:
        mad = 1e-9
    return med, mad


# -------------------------
# Detection logic
# -------------------------

@dataclass
class DetectParams:
    smooth_win_ms: int = 150         # moving-average window for envelope (ms)
    hi_k: float = 3.0                # High threshold: med + hi_k * MAD
    lo_k: float = 2.0                # Low threshold:  med + lo_k * MAD
    min_dur_ms: int = 60             # minimum event duration (ms)
    merge_gap_ms: int = 50           # merge consecutive events if gap <= this (ms)
    refractory_ms: int = 100         # ignore new onset for this after an event (ms)
    onset_snap_pct: float = 0.10     # snap onset to 10% of event peak within the event


def detect_events_hysteresis(env: np.ndarray, fs: float, base_med: float, base_mad: float,
                              params: DetectParams) -> Tuple[List[Tuple[int, int]], float, float]:
    """Detect events with hysteresis on a 1D envelope.
    Returns (events[(s,e),...], hi_thr, lo_thr) in sample indices.
    """
    hi_thr = base_med + params.hi_k * base_mad
    lo_thr = base_med + params.lo_k * base_mad

    above_hi = env >= hi_thr
    above_lo = env >= lo_thr

    events: List[Tuple[int, int]] = []
    active = False
    start = None

    for i in range(len(env)):
        if not active:
            if above_hi[i]:
                active = True
                start = i
        else:
            if not above_lo[i]:
                # candidate end at i-1
                end = i - 1
                events.append((start, end))
                active = False
                start = None
    if active and start is not None:
        events.append((start, len(env) - 1))

    # enforce min duration and merge close gaps
    min_len = int(round(params.min_dur_ms * fs / 1000.0))
    merge_gap = int(round(params.merge_gap_ms * fs / 1000.0))
    filtered: List[Tuple[int, int]] = []
    for (s, e) in events:
        if e - s + 1 >= max(1, min_len):
            if not filtered:
                filtered.append((s, e))
            else:
                ps, pe = filtered[-1]
                if s - pe <= merge_gap:
                    filtered[-1] = (ps, e)
                else:
                    filtered.append((s, e))

    # refractory (suppress very close onsets)
    refr = int(round(params.refractory_ms * fs / 1000.0))
    refractory_filtered: List[Tuple[int, int]] = []
    last_end = -10**9
    for (s, e) in filtered:
        if s - last_end < refr and refractory_filtered:
            # merge into previous
            ps, pe = refractory_filtered[-1]
            refractory_filtered[-1] = (ps, max(pe, e))
            last_end = refractory_filtered[-1][1]
        else:
            refractory_filtered.append((s, e))
            last_end = e

    # onset snapping to 10% of peak within event
    snapped: List[Tuple[int, int]] = []
    for (s, e) in refractory_filtered:
        segment = env[s:e+1]
        if len(segment) == 0:
            snapped.append((s, e))
            continue
        peak = float(np.nanmax(segment))
        target = base_med + params.onset_snap_pct * max(1e-9, (peak - base_med))
        # move left from the first crossing index to where it first reaches target
        idx = s
        for i in range(s, e+1):
            if env[i] >= target:
                idx = i
                break
        snapped.append((idx, e))

    return snapped, hi_thr, lo_thr


def fuse_events_by_vote(per_channel_events: List[List[Tuple[int, int]]], n: int, fs: float,
                        vote_k: int = 2, window_ms: int = 50) -> List[Tuple[int, int]]:
    """Fuse binary per-channel events by a majority vote within a small window."""
    act = np.zeros(n, dtype=int)
    for evs in per_channel_events:
        tmp = np.zeros(n, dtype=int)
        for s, e in evs:
            s = max(0, int(s)); e = min(n-1, int(e))
            tmp[s:e+1] = 1
        act += tmp
    # smoothing vote within window
    w = int(round(window_ms * fs / 1000.0))
    w = max(1, w)
    kernel = np.ones(w, dtype=float)
    sm = np.convolve(act.astype(float), kernel, mode='same')
    fused = sm >= (vote_k)  # if at least vote_k channels active within window

    # binary to segments
    idx = np.where(fused)[0]
    events: List[Tuple[int, int]] = []
    if idx.size > 0:
        s = idx[0]
        p = idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i
            else:
                events.append((s, p))
                s = i
                p = i
        events.append((s, p))
    return events


# -------------------------
# Per-sheet processing
# -------------------------

@dataclass
class ChannelMetrics:
    sheet: str
    channel: str
    n_samples: int
    baseline_median: float
    baseline_mad: float
    hi_thr: float
    lo_thr: float
    n_events: int
    total_active_s: float
    activity_ratio_pct: float


@dataclass
class SheetProfile:
    sheet: str
    rows: int
    cols: int
    duration_s: float
    fs_hz: float
    dt_median_s: float
    dt_std_s: float
    n_channels: int
    n_events_fused: int
    total_active_s_fused: float


def process_sheet(df: pd.DataFrame, sheet_name: str, params: DetectParams,
                  assume_envelope: bool = True,
                  baseline_sec: float = 3.0,
                  smooth_override_ms: Optional[int] = None,
                  vote_k: int = 2) -> Tuple[List[Tuple[float, float]], List[ChannelMetrics], SheetProfile]:
    """Process a single sheet and return fused event labels (in seconds), per-channel metrics, and profile."""
    # Identify time
    time_col = find_time_column(df)
    if time_col is None:
        raise ValueError(f"No time column found in sheet '{sheet_name}'")
    tsec = to_seconds(df[time_col])
    if tsec is None:
        raise ValueError(f"Time column could not be converted to seconds in sheet '{sheet_name}'")
    tsec = pd.Series(tsec).interpolate().bfill().ffill()
    fs, dt_med, dt_std = estimate_fs(tsec)
    if fs is None:
        raise ValueError(f"Could not estimate sampling rate in sheet '{sheet_name}'")

    # Choose smoothing window
    smooth_ms = smooth_override_ms if smooth_override_ms is not None else params.smooth_win_ms
    win = max(1, int(round(smooth_ms * fs / 1000.0)))

    # Pick numeric channels excluding time
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    per_channel_events: List[List[Tuple[int, int]]] = []
    metrics: List[ChannelMetrics] = []

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors='coerce').interpolate().bfill().ffill().values.astype(float)

        # Make an envelope if needed (simple robust envelope)
        if assume_envelope:
            env = moving_average_centered(np.abs(x - np.nanmedian(x)), win)
        else:
            rect = np.abs(x)
            env = moving_average_centered(rect, win)

        # Baseline from first baseline_sec or from the lowest-activity part if too short
        n_baseline = int(round(baseline_sec * fs))
        if len(env) >= n_baseline and n_baseline >= 5:
            base_seg = env[:n_baseline]
        else:
            # fallback: lowest 20% values as baseline
            q = np.quantile(env, 0.2)
            base_seg = env[env <= q]
        base_med, base_mad = robust_median_mad(base_seg)

        events_idx, hi_thr, lo_thr = detect_events_hysteresis(env, fs, base_med, base_mad, params)
        per_channel_events.append(events_idx)

        # stats
        total_active_samples = sum((e - s + 1) for s, e in events_idx)
        total_active_s = total_active_samples / fs
        activity_ratio_pct = 100.0 * total_active_samples / max(1, len(env))

        metrics.append(ChannelMetrics(
            sheet=sheet_name,
            channel=str(col),
            n_samples=int(len(env)),
            baseline_median=float(base_med),
            baseline_mad=float(base_mad),
            hi_thr=float(hi_thr),
            lo_thr=float(lo_thr),
            n_events=int(len(events_idx)),
            total_active_s=float(total_active_s),
            activity_ratio_pct=float(activity_ratio_pct),
        ))

    # Fuse across channels
    n = len(tsec)
    fused_idx = fuse_events_by_vote(per_channel_events, n=n, fs=fs, vote_k=vote_k, window_ms=50)

    # Convert sample indices to seconds
    labels_sec: List[Tuple[float, float]] = []
    for s, e in fused_idx:
        labels_sec.append((float(tsec.iloc[s]), float(tsec.iloc[e])))

    profile = SheetProfile(
        sheet=sheet_name,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        duration_s=float(tsec.iloc[-1] - tsec.iloc[0]),
        fs_hz=float(fs),
        dt_median_s=float(dt_med),
        dt_std_s=float(dt_std),
        n_channels=int(len(numeric_cols)),
        n_events_fused=int(len(labels_sec)),
        total_active_s_fused=float(sum(e - s for s, e in labels_sec)),
    )

    return labels_sec, metrics, profile


# -------------------------
# Plotting utilities
# -------------------------

def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-","_",".") else "_" for ch in str(name))


def make_stage_plots_for_sheet(df: pd.DataFrame, sheet: str, params: DetectParams,
                               baseline_sec: float, vote_k: int, out_dir: str,
                               assume_envelope: bool = True) -> None:
    """Create and save stage-by-stage plots for a sheet.
    Stages:
      1) Raw overview (all channels)
      2) Per-channel: envelope + thresholds + detected events (shaded)
      3) Sum-of-channels z-envelope + fused events
      4) Per-channel: histogram of baseline vs active (z)
    """
    if not _HAS_MPL:
        return

    # Prepare time and channels
    time_col = find_time_column(df)
    tsec = to_seconds(df[time_col])
    t = pd.Series(tsec).interpolate().bfill().ffill().values
    fs, _, _ = estimate_fs(t)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    smooth_ms = params.smooth_win_ms
    win = max(1, int(round(smooth_ms * fs / 1000.0)))
    base_len = int(round(baseline_sec * fs))

    # Compute env, thresholds, events per channel
    env_by_ch: Dict[str, np.ndarray] = {}
    events_by_ch: Dict[str, List[Tuple[int,int]]] = {}
    thr_hi_by_ch: Dict[str, float] = {}
    thr_lo_by_ch: Dict[str, float] = {}
    z_by_ch: Dict[str, np.ndarray] = {}

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors='coerce').interpolate().bfill().ffill().values.astype(float)
        if assume_envelope:
            env = moving_average_centered(np.abs(x - np.nanmedian(x)), win)
        else:
            env = moving_average_centered(np.abs(x), win)
        if len(env) >= base_len and base_len >= 5:
            base_seg = env[:base_len]
        else:
            q = np.quantile(env, 0.2)
            base_seg = env[env <= q]
        bmed, bmad = robust_median_mad(base_seg)
        ev_idx, hi, lo = detect_events_hysteresis(env, fs, bmed, bmad, params)
        env_by_ch[str(col)] = env
        events_by_ch[str(col)] = ev_idx
        thr_hi_by_ch[str(col)] = hi
        thr_lo_by_ch[str(col)] = lo
        z_by_ch[str(col)] = (env - bmed) / max(1e-9, bmad)

    # Fused events
    n = len(t)
    fused_idx = fuse_events_by_vote(list(events_by_ch.values()), n=n, fs=fs, vote_k=vote_k, window_ms=50)

    # Create directories
    sheet_dir = os.path.join(out_dir, _sanitize(sheet))
    os.makedirs(sheet_dir, exist_ok=True)

    # 1) Raw overview
    import matplotlib.pyplot as plt
    rows = len(numeric_cols)
    rows = min(rows, rows) if rows > 0 else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(2.5, 1.6*rows)), sharex=True)
    if rows == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        x = pd.to_numeric(df[col], errors='coerce').interpolate().bfill().ffill().values.astype(float)
        ax.plot(t, x)
        ax.set_ylabel(str(col))
        for s, e in fused_idx:
            ax.axvspan(t[s], t[e], alpha=0.15)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{sheet}: Raw signals with fused events')
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(sheet_dir, 'stage1_raw_overview.png'), dpi=150)
    plt.close(fig)

    # 2) Per-channel envelopes with thresholds and events
    for col in numeric_cols:
        env = env_by_ch[str(col)]
        hi, lo = thr_hi_by_ch[str(col)], thr_lo_by_ch[str(col)]
        evs = events_by_ch[str(col)]
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, env, label='env')
        ax.axhline(hi, linestyle='--', label='High thr')
        ax.axhline(lo, linestyle=':', label='Low thr')
        for s, e in evs:
            ax.axvspan(t[s], t[e], alpha=0.2)
        ax.set_title(f'{sheet} • {col}: Envelope + thresholds + events')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Envelope')
        ax.legend(loc='upper right', framealpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sheet_dir, f"stage2_envelope_{_sanitize(col)}.png"), dpi=150)
        plt.close(fig)

    # 3) Sum-of-channels z-envelope with fused events
    sum_z = np.zeros(n, dtype=float)
    for col in numeric_cols:
        sum_z += z_by_ch[str(col)]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, sum_z, label='sum z-env')
    for s, e in fused_idx:
        ax.axvspan(t[s], t[e], alpha=0.2)
    ax.set_title(f'{sheet}: Sum-of-channels z-envelope + fused events')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Sum z-env')
    ax.legend(loc='upper right', framealpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(sheet_dir, 'stage3_sumz_fused.png'), dpi=150)
    plt.close(fig)

    # 4) Baseline vs active histograms (per channel, z)
    active_mask = np.zeros(n, dtype=bool)
    for s, e in fused_idx:
        active_mask[s:e+1] = True
    for col in numeric_cols:
        z = z_by_ch[str(col)]
        base = z[:base_len] if len(z) >= base_len else z[:max(1, int(0.2*len(z)))]
        act = z[active_mask]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(base, bins=40, alpha=0.6, label='baseline')
        if len(act):
            ax.hist(act, bins=40, alpha=0.6, label='active')
        ax.set_title(f'{sheet} • {col}: baseline vs active (z)')
        ax.set_xlabel('z'); ax.set_ylabel('count')
        ax.legend(framealpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sheet_dir, f"stage4_hist_{_sanitize(col)}.png"), dpi=150)
        plt.close(fig)


# -------------------------
# Workbook-level runner
# -------------------------

def run_workbook(path_xlsx: str,
                 out_dir: str = "labels_out",
                 params: DetectParams = DetectParams(),
                 assume_envelope: bool = True,
                 baseline_sec: float = 3.0,
                 smooth_override_ms: Optional[int] = None,
                 vote_k: int = 2,
                 save_plots: bool = False) -> None:
    """Run the full pipeline on every sheet in the Excel file and save outputs."""
    os.makedirs(out_dir, exist_ok=True)

    xls = pd.ExcelFile(path_xlsx)
    profiles: List[SheetProfile] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(path_xlsx, sheet_name=sheet).dropna(axis=1, how='all')
        try:
            labels_sec, metrics, profile = process_sheet(
                df=df,
                sheet_name=sheet,
                params=params,
                assume_envelope=assume_envelope,
                baseline_sec=baseline_sec,
                smooth_override_ms=smooth_override_ms,
                vote_k=vote_k,
            )
        except Exception as e:
            # Save an empty profile with error info in a CSV note
            profiles.append(SheetProfile(sheet=sheet, rows=int(df.shape[0]), cols=int(df.shape[1]),
                                         duration_s=float('nan'), fs_hz=float('nan'),
                                         dt_median_s=float('nan'), dt_std_s=float('nan'),
                                         n_channels=int(df.shape[1]-1), n_events_fused=0, total_active_s_fused=0.0))
            with open(os.path.join(out_dir, f"ERROR_{sheet}.txt"), 'w', encoding='utf-8') as f:
                f.write(str(e))
            continue

        # Write labels for this sheet
        labels_df = pd.DataFrame(labels_sec, columns=["start_time_s", "end_time_s"]) \
                    .assign(sheet=sheet, label="movement_attempt", confidence=0.8, source="auto", notes="vote>=2")
        labels_df.to_csv(os.path.join(out_dir, f"labels_{sheet}.csv"), index=False)

        # Write channel metrics for this sheet
        ch_df = pd.DataFrame([asdict(m) for m in metrics])
        ch_df.to_csv(os.path.join(out_dir, f"channel_metrics_{sheet}.csv"), index=False)

        # Stage plots (raw, per-channel envelope+thresholds, sum-z fused, histograms)
        if save_plots and _HAS_MPL:
            make_stage_plots_for_sheet(
                df=df,
                sheet=sheet,
                params=params,
                baseline_sec=baseline_sec,
                vote_k=vote_k,
                out_dir=out_dir,
            )

        profiles.append(profile)

    # Save workbook-level profiles
    prof_df = pd.DataFrame([asdict(p) for p in profiles])
    prof_df.to_csv(os.path.join(out_dir, "sheets_profile.csv"), index=False)


# -------------------------
# Example invocation
# -------------------------
if __name__ == "__main__":
    # Edit these paths as needed. If running in the ChatGPT workspace with the provided file, this default works.
    INPUT_XLSX = "30Temmuz_Ampute_polyphase.xlsx"  # path to your multi-sheet workbook
    OUTPUT_DIR = "labels_out"                       # where CSVs/plots will be saved

    # Detection parameters (tweak if needed)
    params = DetectParams(
        smooth_win_ms=150,   # envelope smoothing window
        hi_k=3.0,            # start threshold = median + 3*MAD
        lo_k=2.0,            # continue threshold = median + 2*MAD
        min_dur_ms=60,       # minimum event length
        merge_gap_ms=50,     # merge small gaps inside one attempt
        refractory_ms=100,   # avoid chattering around offsets
        onset_snap_pct=0.10, # snap onset to 10% of peak inside the event
    )

    # Run the pipeline
    run_workbook(
        path_xlsx=INPUT_XLSX,
        out_dir=OUTPUT_DIR,
        params=params,
        assume_envelope=True,     # set False if you feed raw EMG and want rectified+smoothed envelope here
        baseline_sec=3.0,
        smooth_override_ms=None,  # set to an int (ms) to override smoothing per-file
        vote_k=2,                 # require >=2 channels active to call an attempt
        save_plots=True,          # save quick overview plots per sheet (requires matplotlib)
    )

    print(f"Done. Outputs are in: {OUTPUT_DIR}")
