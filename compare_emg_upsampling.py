#!/usr/bin/env python3
"""
compare_emg_upsampling.py (v3)

Enhancements per user feedback:
- Saving: if you choose to save plots, they are saved in the current working directory (no path prompt).
- Time window: explicitly ask for both start and end; press Enter to skip either and keep full range.
- Separate figures: shown sequentially (blocking) in the order of files; x-limits are matched across figures for comparability.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively compare EMG upsampling results across three Excel files."
    )
    parser.add_argument(
        "--files",
        nargs=3,
        metavar=("CUBIC", "LINEAR", "POLYPHASE"),
        help="Paths to the three Excel files in the order (cubic, linear, polyphase).",
    )
    return parser.parse_args()


def default_files():
    candidates = [
        Path("30Temmuz_Ampute_cubic.xlsx"),
        Path("30Temmuz_Ampute_linear.xlsx"),
        Path("30Temmuz_Ampute_polyphase.xlsx"),
    ]
    if all(p.exists() for p in candidates):
        return [str(p) for p in candidates]
    return None


def load_workbooks(file_paths):
    workbooks = {}
    for fp in file_paths:
        label = Path(fp).stem  # e.g., 30Temmuz_Ampute_cubic
        sheets = pd.read_excel(fp, sheet_name=None)
        workbooks[label] = sheets
    return workbooks


def get_common_sheets(workbooks):
    sheet_sets = [set(wb.keys()) for wb in workbooks.values()]
    common = set.intersection(*sheet_sets)
    return sorted(common)


def get_emg_columns(df):
    cols = list(df.columns)
    time_like = [c for c in cols if c.strip().lower() == "time"]
    emg_cols = [c for c in cols if c not in time_like]
    return time_like[0] if time_like else None, emg_cols


def prompt_choice(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        choice = input("Enter number or name (q to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            else:
                print("Invalid number. Try again.")
        else:
            if choice in options:
                return choice
            print("Invalid name. Try again.")


def prompt_yes_no(prompt, default=None):
    if default is None:
        suffix = " [y/n]: "
    elif default is True:
        suffix = " [Y/n]: "
    else:
        suffix = " [y/N]: "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter y or n.")


def clean_time_series(series):
    s = series.copy()
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() >= s.notna().sum() * 0.9:
        return sn.dropna(), "numeric"
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().sum() >= s.notna().sum() * 0.9:
        return dt.dropna(), "datetime"
    return s.dropna(), "other"


def get_time_window(kind):
    """Ask explicitly for start and end; empty input means no bound."""
    if kind == "numeric":
        start_str = input("Enter START (numeric) or press Enter for no lower bound: ").strip()
        end_str = input("Enter END (numeric)   or press Enter for no upper bound: ").strip()
        start = float(start_str) if start_str else None
        end = float(end_str) if end_str else None
        if start is not None and end is not None and start > end:
            start, end = end, start
        return start, end
    elif kind == "datetime":
        start_str = input("Enter START datetime (e.g., 2024-07-01 00:00:00) or Enter to skip: ").strip()
        end_str = input("Enter END   datetime (e.g., 2024-07-01 00:10:00) or Enter to skip: ").strip()
        start = pd.to_datetime(start_str, errors="coerce") if start_str else None
        end = pd.to_datetime(end_str, errors="coerce") if end_str else None
        if start is not None and end is not None and start > end:
            start, end = end, start
        return start, end
    else:
        return None, None


def apply_window(df, start, end):
    if start is not None:
        df = df[df["Time"] >= start]
    if end is not None:
        df = df[df["Time"] <= end]
    return df


def main():
    args = parse_args()
    files = args.files if args.files else default_files()
    if not files:
        print("Error: Please pass three files with --files or place the three default files in the current directory.")
        sys.exit(1)

    for f in files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    workbooks = load_workbooks(files)
    labels = list(workbooks.keys())  # keep file order

    common_sheets = get_common_sheets(workbooks)
    if not common_sheets:
        print("No common sheets across the provided files.")
        sys.exit(1)

    print("\nLoaded files:")
    for i, lab in enumerate(labels, 1):
        print(f"  {i}. {lab}")
    print("\nCommon sheets across all files:")
    for s in common_sheets:
        print(f"  - {s}")

    while True:
        print("\n--- Plot Builder ---")
        sheet = prompt_choice("Choose a sheet:", common_sheets)
        if sheet is None:
            print("Exiting.")
            break

        # Determine common EMG columns
        emg_sets = []
        time_names = []
        for lab in labels:
            df = workbooks[lab][sheet]
            tcol, emg_cols = get_emg_columns(df)
            if tcol is None:
                print(f"Error: No 'Time' column found in sheet '{sheet}' of {lab}.")
                return
            emg_sets.append(set(emg_cols))
            time_names.append(tcol)
        common_emg = sorted(list(set.intersection(*emg_sets)))
        if not common_emg:
            print(f"No common EMG columns across files for sheet '{sheet}'.")
            continue

        col = prompt_choice("Choose an EMG column:", common_emg)
        if col is None:
            print("Exiting.")
            break

        # Plot mode
        mode = prompt_choice("Plot mode:", ["Overlay in one figure", "Separate figures"])
        if mode is None:
            print("Exiting.")
            break
        overlay = (mode == "Overlay in one figure")

        # Save or not (save in current working directory)
        want_save = prompt_yes_no("Save plot(s) to disk? (saved in current working directory)")
        save_dir = Path.cwd() if want_save else None

        # Build aligned datasets per file
        datasets = []
        time_kinds = []
        for lab in labels:
            df = workbooks[lab][sheet].copy()
            tcol, _ = get_emg_columns(df)
            t, tkind = clean_time_series(df[tcol])
            y = pd.to_numeric(df[col], errors="coerce")
            aligned = pd.concat([t.rename("Time"), y.rename(col)], axis=1).dropna()
            datasets.append((lab, aligned))
            time_kinds.append(tkind)

        # Optional time window (only if all time kinds match)
        if len(set(time_kinds)) == 1 and prompt_yes_no("Restrict to a time window?"):
            start, end = get_time_window(time_kinds[0])
            new_datasets = []
            for lab, aligned in datasets:
                new_datasets.append((lab, apply_window(aligned, start, end)))
            datasets = new_datasets
        elif len(set(time_kinds)) != 1:
            print("Note: Time column types differ across files; skipping common window filter.")

        # Determine common x-limits for comparability (both overlay and separate)
        # Only if we have at least one datapoint in any dataset
        xmins, xmaxs = [], []
        for _, aligned in datasets:
            if not aligned.empty:
                xmins.append(aligned["Time"].iloc[0])
                xmaxs.append(aligned["Time"].iloc[-1])
        xlim = None
        if xmins and xmaxs:
            # Use global min and max across datasets
            xlim = (min(xmins), max(xmaxs))

        # Plotting
        if overlay:
            plt.figure()
            for lab, aligned in datasets:
                if aligned.empty:
                    print(f"Warning: No data to plot for {lab} after filtering.")
                    continue
                plt.plot(aligned["Time"].values, aligned[col].values, label=lab)
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.title(f"{sheet} — {col} (Comparison)")
            plt.legend()
            plt.tight_layout()
            if xlim is not None:
                try:
                    plt.xlim(xlim)
                except Exception:
                    pass
            if save_dir:
                safe_sheet = "".join(c if c.isalnum() or c in "-_ " else "_" for c in sheet)
                out = save_dir / f"{safe_sheet}__{col}__overlay.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved: {out}")
            plt.show(block=True)
        else:
            # Separate figures, sequential, with matching xlim where possible
            for lab, aligned in datasets:
                plt.figure()
                if aligned.empty:
                    print(f"Warning: No data to plot for {lab} after filtering.")
                else:
                    plt.plot(aligned["Time"].values, aligned[col].values, label=lab)
                plt.xlabel("Time")
                plt.ylabel(col)
                plt.title(f"{sheet} — {col} ({lab})")
                plt.legend()
                plt.tight_layout()
                if xlim is not None:
                    try:
                        plt.xlim(xlim)
                    except Exception:
                        pass
                if save_dir:
                    safe_sheet = "".join(c if c.isalnum() or c in "-_ " else "_" for c in sheet)
                    safe_lab = "".join(c if c.isalnum() or c in "-_ " else "_" for c in lab)
                    out = save_dir / f"{safe_sheet}__{col}__{safe_lab}.png"
                    plt.savefig(out, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out}")
                # Show each figure and wait until closed before continuing
                print(f"Showing figure for {lab}. Close the window to continue...")
                plt.show(block=True)

        again = input("Plot another? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
