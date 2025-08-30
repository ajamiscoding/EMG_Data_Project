import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- CONFIG ---
input_file = "30Temmuz_Ampute.xlsx"
output_file = "30Temmuz_Ampute_cubic.xlsx"
time_step = 10  # 10 ms

# Read all sheets
sheets = pd.read_excel(input_file, sheet_name=None)
processed_sheets = {}

for sheet_name, df in sheets.items():
    # Drop missing values
    df = df.dropna()

    # Group by time to remove duplicates, take mean of EMG values
    df = df.groupby(df.columns[0], as_index=False).mean()

    # Sort by time
    df = df.sort_values(by=df.columns[0])

    # Extract time and channel data
    time = df.iloc[:, 0].values
    channels = df.iloc[:, 1:].values
    channel_names = df.columns[1:]

    # Create new time grid
    max_time = time.max()
    new_time = np.arange(0, max_time, time_step)
    if new_time[-1] != max_time:
        new_time = np.append(new_time, max_time)

    # Prepare new dataframe
    new_df = pd.DataFrame({"Time": new_time})

    # Cubic interpolation for each channel
    for i, col_name in enumerate(channel_names):
        interp_func = interp1d(time, channels[:, i], kind='cubic', fill_value="extrapolate")
        new_df[col_name] = interp_func(new_time)

    processed_sheets[sheet_name] = new_df

# Save all processed sheets
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in processed_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"cubic interpolation completed. Saved to '{output_file}'")
