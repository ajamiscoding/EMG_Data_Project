
"""
Robust Multi-sheet Movement Detection & Labeling Pipeline
--------------------------------------------------------
- Loads an Excel workbook with multiple sheets.
- Profiles each sheet (timing, sampling rate, integrity).
- Preprocesses signals (light smoothing).
- Rolling robust z-normalization per channel (median/MAD over a moving window).
- Per-channel hysteresis detection on z.
- Channel fusion by vote + gating using MEAN z (to reduce false positives).
- Rescue rule for short but strong true attempts.
- Stage plots per sheet (optional).

Outputs:
    labels_{sheet}.csv              # fused movement attempts (start/end in seconds)
    channel_metrics_{sheet}.csv     # per-channel stats & thresholds
    sheets_profile.csv              # one-row summary per sheet
    <sheet>/*.png                   # stage plots if save_plots=True

Dependencies: pandas, numpy, (optional) matplotlib for plots.

Usage:
    python multi_sheet_labeling_robust.py
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
    hi_k: float = 3.0                # High threshold on z: med + hi_k * MAD (z units -> med=0, MAD=1)
    lo_k: float = 2.0                # Low threshold on z
    min_dur_ms: int = 60             # minimum event duration (ms)
    merge_gap_ms: int = 50           # merge consecutive events if gap <= this (ms)
    refractory_ms: int = 100         # ignore new onset for this after an event (ms)
    onset_snap_pct: float = 0.10     # snap onset to 10% of event peak within the event

    # Rolling baseline for drift-robust z-scoring
    rolling_baseline_sec: float = 5.0

    # Fused-event gating on MEAN z across channels
    z_fused_min_peak: float = 3.0        # minimum peak mean-z inside fused event
    z_fused_min_area: float = 0.30       # minimum area (integral of mean-z above 0) in z*seconds
    z_fused_min_channels_at_peak: int = 2

    # Rescue tiny but strong events
    z_rescue_single_peak: float = 5.0    # rescue if any single channel z-peak >= this
    min_dur_ms_short: int = 40           # min duration for rescue events (ms)


def detect_events_hysteresis(env: np.ndarray, fs: float, base_med: float, base_mad: float,
                              params: DetectParams) -> Tuple[List[Tuple[int, int]], float, float]:
    """Detect events with hysteresis on a 1D signal (env or z-env).
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
# Rolling z + gating helpers
# -------------------------

def rolling_robust_z(env: np.ndarray, fs: float, roll_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling median/MAD z-score for drift-robust normalization."""
    w = max(5, int(round(roll_sec * fs)))
    s = pd.Series(env)

    med = s.rolling(w, center=True, min_periods=max(3, w//3)).quantile(0.5)

    def _mad(a: pd.Series) -> float:
        m = np.median(a.values)
        return np.median(np.abs(a.values - m))

    mad = s.rolling(w, center=True, min_periods=max(3, w//3)).apply(_mad, raw=False)
    med = med.fillna(method='bfill').fillna(method='ffill').fillna(float(np.nanmedian(env)))
    mad = mad.fillna(method='bfill').fillna(method='ffill').replace(0, 1e-9)

    z = (s - med) / mad
    return z.values.astype(float), med.values.astype(float), mad.values.astype(float)


def gate_fused_events(mean_z: np.ndarray, per_channel_z: List[np.ndarray],
                      events: List[Tuple[int,int]], fs: float, params: DetectParams) -> List[Tuple[int,int]]:
    """Filter fused events using mean z, area and channel participation; rescue strong but tiny ones."""
    accepted: List[Tuple[int,int]] = []
    for (s, e) in events:
        seg = mean_z[s:e+1]
        peak = float(np.nanmax(seg)) if len(seg) else 0.0
        area = float(np.trapz(np.maximum(seg, 0), dx=1.0/fs))  # z*sec
        dur = (e - s + 1) / fs

        # channels active (z >= 2) at the event peak
        peak_idx = s + int(np.argmax(seg)) if len(seg) else s
        ch_at_peak = sum(1 for z in per_channel_z if peak_idx < len(z) and z[peak_idx] >= 2.0)

        if (peak >= params.z_fused_min_peak and
            area >= params.z_fused_min_area and
            ch_at_peak >= params.z_fused_min_channels_at_peak):
            accepted.append((s, e))
        else:
            # rescue rule: any single channel very strong for a short burst
            strong = any((len(z[s:e+1]) and np.nanmax(z[s:e+1]) >= params.z_rescue_single_peak)
                         for z in per_channel_z)
            if strong and dur >= params.min_dur_ms_short / 1000.0:
                accepted.append((s, e))
    return accepted


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
    per_channel_z: List[np.ndarray] = []

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors='coerce').interpolate().bfill().ffill().values.astype(float)

        # Envelope
        if assume_envelope:
            env = moving_average_centered(np.abs(x - np.nanmedian(x)), win)
        else:
            env = moving_average_centered(np.abs(x), win)

        # Rolling robust z (handles drift)
        z, _, _ = rolling_robust_z(env, fs, params.rolling_baseline_sec)
        per_channel_z.append(z)

        # Per-channel detection on z
        events_idx, _, _ = detect_events_hysteresis(z, fs, base_med=0.0, base_mad=1.0, params=params)
        per_channel_events.append(events_idx)

        # stats (use first baseline window for reporting)
        n_baseline = int(round(baseline_sec * fs))
        base_seg = env[:n_baseline] if len(env) >= n_baseline and n_baseline >= 5 else env
        base_med, base_mad = robust_median_mad(base_seg)
        total_active_samples = sum((e - s + 1) for s, e in events_idx)
        total_active_s = total_active_samples / fs
        activity_ratio_pct = 100.0 * total_active_samples / max(1, len(env))

        metrics.append(ChannelMetrics(
            sheet=sheet_name,
            channel=str(col),
            n_samples=int(len(env)),
            baseline_median=float(base_med),
            baseline_mad=float(base_mad),
            hi_thr=float(params.hi_k),  # z-thresholds in z units
            lo_thr=float(params.lo_k),
            n_events=int(len(events_idx)),
            total_active_s=float(total_active_s),
            activity_ratio_pct=float(activity_ratio_pct),
        ))

    # Fuse across channels
    n = len(tsec)
    fused_idx = fuse_events_by_vote(per_channel_events, n=n, fs=fs, vote_k=vote_k, window_ms=50)

    # Gate fused events using MEAN z across channels (scale-invariant wrt #channels)
    mean_z = np.mean(np.vstack(per_channel_z), axis=0) if per_channel_z else np.zeros(n)
    fused_idx = gate_fused_events(mean_z, per_channel_z, fused_idx, fs, params)

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
      2) Per-channel: z-envelope + thresholds + detected events (shaded)
      3) MEAN-of-channels z-envelope + fused events
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

    # Compute z and events per channel
    z_by_ch: Dict[str, np.ndarray] = {}
    events_by_ch: Dict[str, List[Tuple[int,int]]] = {}

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors='coerce').interpolate().bfill().ffill().values.astype(float)
        if assume_envelope:
            env = moving_average_centered(np.abs(x - np.nanmedian(x)), win)
        else:
            env = moving_average_centered(np.abs(x), win)
        z, _, _ = rolling_robust_z(env, fs, roll_sec=params.rolling_baseline_sec)
        ev_idx, _, _ = detect_events_hysteresis(z, fs, base_med=0.0, base_mad=1.0, params=params)
        z_by_ch[str(col)] = z
        events_by_ch[str(col)] = ev_idx

    # Fused events
    n = len(t)
    fused_idx = fuse_events_by_vote(list(events_by_ch.values()), n=n, fs=fs, vote_k=vote_k, window_ms=50)
    mean_z = np.mean(np.vstack([z_by_ch[str(c)] for c in numeric_cols]), axis=0) if numeric_cols else np.zeros(n)
    fused_idx = gate_fused_events(mean_z, [z_by_ch[str(c)] for c in numeric_cols], fused_idx, fs, params)

    # Create directories
    sheet_dir = os.path.join(out_dir, _sanitize(sheet))
    os.makedirs(sheet_dir, exist_ok=True)

    # 1) Raw overview
    import matplotlib.pyplot as plt
    rows = len(numeric_cols) if len(numeric_cols) > 0 else 1
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

    # 2) Per-channel z with thresholds and events
    for col in numeric_cols:
        z = z_by_ch[str(col)]
        evs = events_by_ch[str(col)]
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, z, label='z-env')
        ax.axhline(params.hi_k, linestyle='--', label='High thr (z)')
        ax.axhline(params.lo_k, linestyle=':', label='Low thr (z)')
        for s, e in evs:
            ax.axvspan(t[s], t[e], alpha=0.2)
        ax.set_title(f'{sheet} • {col}: z-envelope + thresholds + events')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('z')
        ax.legend(loc='upper right', framealpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sheet_dir, f"stage2_z_{_sanitize(col)}.png"), dpi=150)
        plt.close(fig)

    # 3) Mean-of-channels z-envelope with gated fused events
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, mean_z, label='mean z-env')
    for s, e in fused_idx:
        ax.axvspan(t[s], t[e], alpha=0.2)
    ax.set_title(f'{sheet}: Mean-of-channels z-envelope + fused events')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean z-env')
    ax.legend(loc='upper right', framealpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(sheet_dir, 'stage3_meanz_fused.png'), dpi=150)
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
                    .assign(sheet=sheet, label="movement_attempt", confidence=0.8, source="auto", notes="vote>=2 + meanZ gate")
        labels_df.to_csv(os.path.join(out_dir, f"labels_{sheet}.csv"), index=False)

        # Write channel metrics for this sheet
        ch_df = pd.DataFrame([asdict(m) for m in metrics])
        ch_df.to_csv(os.path.join(out_dir, f"channel_metrics_{sheet}.csv"), index=False)

        # Stage plots (raw, per-channel z, mean-z fused, histograms)
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
    # Edit these paths as needed.
    INPUT_XLSX = "30Temmuz_Ampute_polyphase.xlsx"  # path to your multi-sheet workbook
    OUTPUT_DIR = "labels_out_robust"                       # where CSVs/plots will be saved

    # Detection parameters (tweak if needed)
    params = DetectParams(
        smooth_win_ms=150,      # envelope smoothing window
        hi_k=3.0,               # z start threshold
        lo_k=2.0,               # z continue threshold
        min_dur_ms=60,          # minimum event length
        merge_gap_ms=50,        # merge small gaps inside one attempt
        refractory_ms=100,      # avoid chattering around offsets
        onset_snap_pct=0.10,    # snap onset to 10% of peak inside the event
        rolling_baseline_sec=5.0,
        z_fused_min_peak=3.0,
        z_fused_min_area=0.30,
        z_fused_min_channels_at_peak=2,
        z_rescue_single_peak=5.0,
        min_dur_ms_short=40,
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
        save_plots=True,          # save stage plots per sheet (requires matplotlib)
    )

    print(f"Done. Outputs are in: {OUTPUT_DIR}")
