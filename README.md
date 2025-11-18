# EMG / Movement Detection Pipelines

Multi-channel EMG-like movement detection on Excel workbooks. The main script (`emg_pipeline_no_envelope.py`) high‑pass filters each channel, rectifies, builds per-channel RMS thresholds from a rest window, and detects pulses via channel consensus.

## Repository Map
- `emg_pipeline_no_envelope.py` — no-envelope pipeline with RMS + consensus and plotting.
- `emg_pipeline_final.py`, `emg_pipeline_final_with_gating.py`, `emg_pipeline_rms.py`, `emg_pipeline_per_channel.py` — variations of the pipeline.
- `multi_sheet_labeling_robust.py`, `multi_sheet_movement_labeling_pipeline_python.py` — labeling helpers for Excel workbooks.
- `Upsampling/`, `Labeling/`, `outputs_*` — data utilities and past outputs.
- `requirements.txt` — Python dependencies.

## Setup
1. Python 3.10+ recommended.
2. (Optional) `python -m venv emg_venv`
   - Windows: `emg_venv\Scripts\activate`
   - Unix: `source emg_venv/bin/activate`
3. `pip install -r requirements.txt`

## Input Format
- Input: an Excel workbook; each sheet is processed independently.
- Each sheet must have one time column (auto-detected: `Time`, `time`, `t`, `Zaman`, `Timestamp`, or the first column) and remaining columns are channels (`emg_pipeline_no_envelope.py:136-164`).
- Time is auto-interpreted as ms (if ~1 step and >100 max) or seconds (otherwise), then converted to seconds.

## How the No-Envelope Pipeline Works
- Baseline: selects rest window (`--rest_start`, `--rest_end`, defaults 0–3 s; fallback to first ~20%/≤5 s if too short) and computes mean/std/median/MAD (`emg_pipeline_no_envelope.py:175-209`).
- Dead channels: flags near-zero variance/MAD channels and zeros them (`emg_pipeline_no_envelope.py:199-229`).
- Preprocess: 0.5 Hz high-pass, then abs() as activity (`emg_pipeline_no_envelope.py:258-270`).
- Normalize: z = (act − median) / MAD; optional gating sets |z| < `gate_eps` to 0 for logic/plots only (`emg_pipeline_no_envelope.py:232-251`).
- Features: sliding-window RMS and mean z (`window_ms`, `overlap_ms`) plus window centers (`emg_pipeline_no_envelope.py:277-305`).
- Thresholds: per-channel RMS thresholds from rest windows via median + k*MAD, capped by percentile (`per_channel_k`, `per_channel_perc`) and ignoring dead channels (`emg_pipeline_no_envelope.py:312-348`).
- Activity: binary per-channel activity → fraction of active channels → pulses via hysteresis (`consensus_min_fraction`, `pulse_on_frac`, `pulse_off_frac`, `pulse_min_duration_ms`, `pulse_merge_gap_ms`; see `emg_pipeline_no_envelope.py:348-477`).
- Outputs: per-sheet CSVs + plots (`emg_pipeline_no_envelope.py:753-801`).

## Running the No-Envelope Pipeline
```bash
python emg_pipeline_no_envelope.py \
  --in_xlsx path/to/input.xlsx \
  --out_dir outputs_no_env/run1 \
  --rest_start 0 --rest_end 3 \
  --window_ms 200 --overlap_ms 100 \
  --per_channel_k 2.0 --per_channel_perc 85 \
  --gate_eps 0.3 \
  --consensus_min_fraction 0.4 \
  --pulse_on_frac 0.4 --pulse_off_frac 0.3 \
  --pulse_min_duration_ms 150 --pulse_merge_gap_ms 150
