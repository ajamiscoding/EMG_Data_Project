# label_emg.py
# -----------------------------------------------------
# EMG labeling: detect movement start/end + optional KMeans noise filtering
# -----------------------------------------------------

import pandas as pd
import numpy as np
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

# ========= USER PARAMS (tweak if needed) =========
EXCEL_FILE      = "30Temmuz_Ampute_polyphase.xlsx"
WRITE_WINDOWS   = False     # True -> also writes window_labels.csv
USE_KMEANS      = True      # True -> filter noisy segments via KMeans(k=2) per sheet

# Segmentation params
WINDOW_MS       = 500.0     # feature window length
OVERLAP         = 0.9       # 50% overlap
THR_K           = 3.0       # baseline mean + K*std
VOTE_CHANNELS   = 3         # window active if >= this many channels fire
MERGE_GAP_MS    = 200.0     # join gaps <= this to avoid splitting one movement
MIN_SEG_MS      = 500.0     # drop micro bursts

# KMeans params
MIN_SEGS_FOR_KM = 6         # need at least this many segments to run kmeans
N_CLUSTERS      = 2         # 2 = intended vs other
SEED            = 0

# ========= HELPERS =========

def estimate_dt_ms(t_ms: np.ndarray) -> float:
    diffs = np.diff(t_ms)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 10.0

def window_indices(n, win, stride):
    i = 0
    while i + win <= n:
        yield i, i + win
        i += stride

def per_window_features(X_win: np.ndarray):
    """
    X_win: shape (win, channels)
    returns dict of per-channel features (RMS, P2P, STD)
    """
    rms = np.sqrt((X_win**2).mean(axis=0))          # (ch,)
    p2p = (X_win.max(axis=0) - X_win.min(axis=0))   # (ch,)
    std = X_win.std(axis=0, ddof=1)                 # (ch,)
    return rms, p2p, std

def segment_sheet(df: pd.DataFrame):
    """
    Segmentation -> returns:
      segments: list of (start_idx, end_idx) in sample indices
      idxs: list of (i0,i1) window sample indices
      time_ms: time array
    """
    time_ms = df["Time"].to_numpy(dtype=float)
    emg_cols = [c for c in df.columns if str(c).lower().startswith("emg")]
    X = df[emg_cols].to_numpy(dtype=float)  # (n, ch)
    n, ch = X.shape
    dt = estimate_dt_ms(time_ms)

    win = max(8, int(round(WINDOW_MS / dt)))
    stride = max(4, int(round(win * (1 - OVERLAP))))

    # Per-window, per-channel features
    W_rms = []
    W_p2p = []
    W_std = []
    idxs = []
    for i0, i1 in window_indices(n, win, stride):
        rms, p2p, std = per_window_features(X[i0:i1, :])
        W_rms.append(rms)
        W_p2p.append(p2p)
        W_std.append(std)
        idxs.append((i0, i1))

    if not W_rms:
        return [], idxs, time_ms, dt

    W_rms = np.vstack(W_rms)  # (num_windows, ch)
    W_p2p = np.vstack(W_p2p)
    W_std = np.vstack(W_std)

    # Channel-wise adaptive threshold (using RMS only is usually enough)
    votes = np.zeros(W_rms.shape[0], dtype=int)
    for c in range(ch):
        col = W_rms[:, c]
        k = max(1, int(0.30 * len(col)))            # quietest 30%
        base = np.sort(col)[:k]
        mu = base.mean()
        sd = base.std(ddof=1) if len(base) > 1 else 1e-6
        thr = mu + THR_K * sd
        votes += (col > thr).astype(int)

    win_active = votes >= VOTE_CHANNELS

    # Merge consecutive active windows -> preliminary segments (sample indices)
    prelim = []
    for is_act, group in groupby(range(len(win_active)), key=lambda i: bool(win_active[i])):
        if not is_act:
            continue
        g = list(group)
        s0 = idxs[g[0]][0]
        s1 = idxs[g[-1]][1]
        prelim.append([s0, s1])

    # Close small gaps
    segs = []
    if prelim:
        cur_s, cur_e = prelim[0]
        for s, e in prelim[1:]:
            gap_ms = (s - cur_e) * dt
            if gap_ms <= MERGE_GAP_MS:
                cur_e = e
            else:
                segs.append([cur_s, cur_e])
                cur_s, cur_e = s, e
        segs.append([cur_s, cur_e])

    # Drop too-short
    segs = [(s, e) for (s, e) in segs if (e - s) * dt >= MIN_SEG_MS]
    return segs, idxs, time_ms, dt

def segment_feature_vector(X: np.ndarray, s: int, e: int):
    """
    Build a feature vector for segment X[s:e, :]
    Features: per-channel RMS, P2P, STD + duration_ms
    """
    seg = X[s:e, :]
    rms = np.sqrt((seg**2).mean(axis=0))      # (ch,)
    p2p = (seg.max(axis=0) - seg.min(axis=0))
    std = seg.std(axis=0, ddof=1)
    feats = np.hstack([rms, p2p, std])        # 3*ch
    return feats

def kmeans_filter_segments(X: np.ndarray, segs, dt_ms: float):
    """
    Returns mask of which segments to KEEP (intended) using KMeans(k=2).
    Safety: if too few segs, or weak separation -> keep all.
    """
    if len(segs) < MIN_SEGS_FOR_KM:
        return np.ones(len(segs), dtype=bool)  # keep all

    # Build feature matrix (3*ch + 1 duration)
    feats = []
    durations = []
    for (s, e) in segs:
        f = segment_feature_vector(X, s, e)
        dur = (e - s) * dt_ms
        feats.append(np.hstack([f, dur]))
        durations.append(dur)
    F = np.vstack(feats)

    # Scale features
    scaler = StandardScaler()
    Fz = scaler.fit_transform(F)

    # KMeans
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
    labels = km.fit_predict(Fz)

    # Decide intended cluster: larger size and/or higher mean RMS (first 8 dims of RMS inside F)
    # Our F layout: [rms(8), p2p(8), std(8), duration]
    ch = X.shape[1]
    rms_block = F[:, :ch]  # (n_segs, 8)
    total_rms = rms_block.sum(axis=1)

    keep_mask = np.zeros(len(segs), dtype=bool)
    for c in range(N_CLUSTERS):
        idx = (labels == c)
        size = idx.sum()
        mean_rms = total_rms[idx].mean() if size > 0 else -np.inf
        km.cluster_centers_[c] = km.cluster_centers_[c]  # just to touch; not used below

    # Choose intended cluster
    # Rule: higher average total RMS wins; tie-breaker = larger size
    rms_means = []
    sizes = []
    for c in range(N_CLUSTERS):
        idx = (labels == c)
        sizes.append(idx.sum())
        rms_means.append(total_rms[idx].mean() if idx.sum() else -np.inf)

    intended = int(np.argmax(rms_means))
    # tie-breaker
    if np.isfinite(rms_means).all():
        # if means very close, prefer larger size
        if abs(rms_means[0] - rms_means[1]) < 1e-6:
            intended = int(np.argmax(sizes))

    keep_mask = (labels == intended)

    # Safety: if intended cluster is tiny, keep all (avoid over-filtering)
    if keep_mask.sum() < max(3, len(segs) // 4):
        return np.ones(len(segs), dtype=bool)

    return keep_mask

# ========= MAIN =========

def main():
    xls = pd.ExcelFile(EXCEL_FILE)
    all_seg_rows = []
    all_win_rows = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet)
        emg_cols = [c for c in df.columns if str(c).lower().startswith("emg")]
        X = df[emg_cols].to_numpy(dtype=float)
        segs, idxs, time_ms, dt = segment_sheet(df)

        if not segs:
            print(f"[{sheet}] No segments found.")
            continue

        # Optional: KMeans filter (keep intended, drop other)
        if USE_KMEANS:
            keep_mask = kmeans_filter_segments(X, segs, dt)
        else:
            keep_mask = np.ones(len(segs), dtype=bool)

        kept = [segs[i] for i in range(len(segs)) if keep_mask[i]]

        # Write segment labels (start/end)
        seg_id = 1
        for (s, e) in kept:
            start = float(time_ms[s]); end = float(time_ms[e-1])
            all_seg_rows.append({
                "sheet": sheet,
                "seg_id": seg_id,
                "start_ms": round(start, 1),
                "end_ms": round(end, 1),
                "duration_ms": round(end - start, 1),
                "label": sheet  # sheet name as movement label
            })
            seg_id += 1

        # Optional window-level labels
        if WRITE_WINDOWS:
            for (i0, i1) in idxs:
                ws, we = float(time_ms[i0]), float(time_ms[i1 - 1])
                # label 'Movement Active' if overlaps any kept seg
                lab = "Rest"
                for (s, e) in kept:
                    if not (i1 <= s or i0 >= e):
                        lab = sheet  # mark with movement label
                        break
                all_win_rows.append({
                    "sheet": sheet,
                    "window_start_ms": round(ws, 1),
                    "window_end_ms": round(we, 1),
                    "label": lab
                })

        print(f"[{sheet}] segments detected: {len(segs)}, kept: {len(kept)} (dt≈{dt:.1f} ms)")

    # Save CSVs
    pd.DataFrame(all_seg_rows).to_csv("segments_labeled.csv", index=False)
    if WRITE_WINDOWS:
        pd.DataFrame(all_win_rows).to_csv("window_labels.csv", index=False)

    print("\nDone.")
    print(" - segments_labeled.csv written")
    if WRITE_WINDOWS:
        print(" - window_labels.csv written")

if __name__ == "__main__":
    main()
