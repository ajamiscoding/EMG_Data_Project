import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from fractions import Fraction

# --- CONFIG ---
input_file = "Cuneyt_Yilmaz.xlsx"
output_file = "Cuneyt_Yilmaz_polyphase_1ms.xlsx"
time_step = 1  # milliseconds (ms)

# Read all sheets
sheets = pd.read_excel(input_file, sheet_name=None)
processed_sheets = {}

for sheet_name, df in sheets.items():
    # Drop missing rows
    df = df.dropna()

    # Make sure first col is time (ms). Round to avoid float-dup issues, then average duplicates
    tcol = df.columns[0]
    df[tcol] = df[tcol].astype(float).round(6)
    df = df.groupby(tcol, as_index=False).mean()

    # Sort by time
    df = df.sort_values(by=tcol)

    # Extract time (ms) and channels
    time = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    channel_names = df.columns[1:]

    if len(time) < 3:
        # Not enough samples to resample — just pass through
        processed_sheets[sheet_name] = df.rename(columns={tcol: "Time"})
        continue

    # Estimate original uniform dt (ms) by median (more robust than min)
    diffs = np.diff(time)
    orig_dt = np.median(diffs)
    if orig_dt <= 0:
        raise ValueError(f"Non-positive dt detected in sheet '{sheet_name}'.")

    # Build a strictly uniform original time grid over the data span
    t0, t1 = time[0], time[-1]
    t_uniform = np.arange(t0, t1, orig_dt)
    if t_uniform[-1] != t1:
        t_uniform = np.append(t_uniform, t1)

    # Interpolate all channels to uniform grid (required by resample_poly)
    X_uniform = np.empty((len(t_uniform), X.shape[1]), dtype=float)
    for i in range(X.shape[1]):
        f = interp1d(time, X[:, i], kind="linear", fill_value="extrapolate", assume_sorted=True)
        X_uniform[:, i] = f(t_uniform)

    # Compute sampling rates (Hz): time is in ms -> fs = 1000 / dt_ms
    orig_fs = 1000.0 / orig_dt
    target_fs = 1000.0 / float(time_step)

    # Rational approximation of ratio; cap denominator to keep filters practical
    ratio = Fraction(target_fs / orig_fs).limit_denominator(1000)
    L, M = ratio.numerator, ratio.denominator

    # Polyphase resampling on the uniform data
    X_resamp = np.vstack([
        resample_poly(X_uniform[:, i], L, M)
        for i in range(X_uniform.shape[1])
    ]).T  # shape: (N_resamp, n_channels)

    # Build the corresponding resampled time grid (ms)
    # Length scales by ~L/M over the original duration
    t_resamp = np.linspace(t_uniform[0], t_uniform[-1], num=X_resamp.shape[0])

    # Final desired grid (ms) for output, aligned to the same start time
    new_time = np.arange(t0, t1, time_step)
    if new_time[-1] != t1:
        new_time = np.append(new_time, t1)

    # Interpolate resampled data onto the final grid
    new_df = pd.DataFrame({"Time": new_time})
    for i, col in enumerate(channel_names):
        f2 = interp1d(t_resamp, X_resamp[:, i], kind="linear", fill_value="extrapolate", assume_sorted=True)
        new_df[col] = f2(new_time)

    processed_sheets[sheet_name] = new_df

# Save all processed sheets
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in processed_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Polyphase resampling completed. Saved to '{output_file}'")
