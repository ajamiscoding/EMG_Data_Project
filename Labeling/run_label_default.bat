@echo off
set "PY=python"
set "IN=30Temmuz_Ampute_polyphase.xlsx"
set "OUT=labels_out"

%PY% "%~dp0label_emg_v5.py" ^
  --in_xlsx "%~dp0%IN%" ^
  --out_dir "%~dp0%OUT%" ^
  --force_mode envelope ^
  --rms_win_ms 300 ^
  --hop_fraction 0.5 ^
  --label_median_win 7 ^
  --min_active_windows 3 ^
  --sample_median_win_ms 150 ^
  --min_on_ms 260 ^
  --min_off_ms 280 ^
  --merge_gap_ms 400 ^
  --peak_min_z 1.2 ^
  --channels_at_peak_min 3 ^
  --env_highpass_hz 0.5 ^
  --env_lowpass_hz 6 ^
  --k_top 4 ^
  --n_clusters 2 ^
  --notch_freq 50 ^
  --bp_low 20 ^
  --bp_high 450 ^
  --time_column "Time" ^
  --channel_prefix "emg"

pause
