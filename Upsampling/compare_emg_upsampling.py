#!/usr/bin/env python3
"""
compare_emg_upsampling.py (v6)

Changes:
- Automatically include ORIGINAL file named "30Temmuz_Ampute.xlsx" if it exists in the current directory.
- Do NOT align to the original time grid (no prompt). All series are plotted on their native time bases.
- No resampling prompts. (Pure visual comparison unless you apply a time window.)
- Saving goes to current working directory when you choose to save.
- Separate-figure mode shows windows sequentially; common x-limits applied for comparability.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Compare ORIGINAL and upsampled EMG files (no alignment).")
    p.add_argument(
        "--files",
        nargs=3,
        metavar=("CUBIC", "LINEAR", "POLYPHASE"),
        help="Paths to the three upsampled Excel files in the order (cubic, linear, polyphase).",
    )
    # --original kept but optional; will default to 30Temmuz_Ampute.xlsx if present in CWD
    p.add_argument("--original", type=str, default=None, help="Path to ORIGINAL file (defaults to 30Temmuz_Ampute.xlsx in CWD).")
    return p.parse_args()


def default_files():
    candidates = [
        Path("30Temmuz_Ampute_cubic.xlsx"),
        Path("30Temmuz_Ampute_linear.xlsx"),
        Path("30Temmuz_Ampute_polyphase.xlsx"),
    ]
    if all(p.exists() for p in candidates):
        return [str(p) for p in candidates]
    return None


def load_workbook(fp):
    return pd.read_excel(fp, sheet_name=None)


def get_common_sheets(workbooks):
    sheet_sets = [set(wb.keys()) for wb in workbooks.values()]
    common = set.intersection(*sheet_sets) if sheet_sets else set()
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
        print("Error: Provide three upsampled files with --files or place the defaults in the current directory.")
        sys.exit(1)

    labels = [Path(fp).stem for fp in files]
    workbooks = {lab: load_workbook(fp) for lab, fp in zip(labels, files)}

    # Auto-include original: default to "30Temmuz_Ampute.xlsx" in CWD if not explicitly provided
    original_path = args.original or "30Temmuz_Ampute.xlsx"
    if Path(original_path).exists():
        original_label = Path(original_path).stem
        # Put ORIGINAL first for clarity in legends/titles
        labels = [original_label] + labels
        workbooks[original_label] = load_workbook(original_path)
        print(f"Included ORIGINAL file: {original_path}")
    else:
        original_label = None
        print("Note: ORIGINAL file not found in CWD (30Temmuz_Ampute.xlsx). Proceeding without it.")

    print("\nLoaded files:")
    for i, lab in enumerate(labels, 1):
        print(f"  {i}. {lab}")

    common_sheets = get_common_sheets(workbooks)
    if not common_sheets:
        print("No common sheets across the provided files.")
        sys.exit(1)
    print("\nCommon sheets across all files:")
    for s in common_sheets:
        print(f"  - {s}")

    while True:
        print("\n--- Plot Builder ---")
        sheet = prompt_choice("Choose a sheet:", common_sheets)
        if sheet is None:
            print("Exiting.")
            break

        # EMG columns common to all files for this sheet
        emg_sets = []
        time_kinds = []
        for lab in labels:
            df = workbooks[lab][sheet]
            tcol, emg_cols = get_emg_columns(df)
            if tcol is None:
                print(f"Error: No 'Time' column found in sheet '{sheet}' of {lab}.")
                return
            emg_sets.append(set(emg_cols))
            _, k = clean_time_series(df[tcol])
            time_kinds.append(k)
        common_emg = sorted(list(set.intersection(*emg_sets)))
        if not common_emg:
            print(f"No common EMG columns across files for sheet '{sheet}'.")
            continue

        col = prompt_choice("Choose an EMG column:", common_emg)
        if col is None:
            print("Exiting.")
            break

        # Build datasets on native time bases (no alignment)
        datasets = []
        for lab in labels:
            df = workbooks[lab][sheet].copy()
            tcol, _ = get_emg_columns(df)
            t, _ = clean_time_series(df[tcol])
            y = pd.to_numeric(df[col], errors="coerce")
            aligned = pd.concat([t.rename("Time"), y.rename(col)], axis=1).dropna()
            datasets.append((lab, aligned))

        # Time window (use the original's time type if present, else first file)
        if original_label is not None:
            rep_df = workbooks[original_label][sheet]
        else:
            rep_df = workbooks[labels[0]][sheet]
        rep_tcol, _ = get_emg_columns(rep_df)
        _, rep_kind = clean_time_series(rep_df[rep_tcol])
        if prompt_yes_no("Restrict to a time window?", default=False):
            start, end = get_time_window(rep_kind)
            datasets = [(lab, apply_window(df, start, end)) for (lab, df) in datasets]

        # Plot mode
        mode = prompt_choice("Plot mode:", ["Overlay in one figure", "Separate figures"])
        if mode is None:
            print("Exiting.")
            break
        overlay = (mode == "Overlay in one figure")

        # Save or not (current working directory)
        want_save = prompt_yes_no("Save plot(s) to disk in the current directory?")

        # Common x-limits for comparability
        xmins, xmaxs = [], []
        for _, aligned in datasets:
            if not aligned.empty:
                xmins.append(aligned["Time"].iloc[0])
                xmaxs.append(aligned["Time"].iloc[-1])
        xlim = (min(xmins), max(xmaxs)) if xmins and xmaxs else None

        sheet_safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in sheet)
        if overlay:
            plt.figure()
            for lab, aligned in datasets:
                if aligned.empty:
                    print(f"Warning: No data to plot for {lab} after filtering.")
                    continue
                plt.plot(aligned["Time"].values, aligned[col].values, label=lab)
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.title(f"{sheet} — {col} (Comparison, native time bases)")
            plt.legend()
            plt.tight_layout()
            if xlim is not None:
                try:
                    plt.xlim(xlim)
                except Exception:
                    pass
            if want_save:
                out = Path.cwd() / "saved_plots" / f"{sheet_safe}__{col}__overlay.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved: {out}")
            plt.show(block=True)
        else:
            for lab, aligned in datasets:
                plt.figure()
                if aligned.empty:
                    print(f"Warning: No data to plot for {lab} after filtering.")
                else:
                    plt.plot(aligned["Time"].values, aligned[col].values, label=lab)
                plt.xlabel("Time")
                plt.ylabel(col)
                plt.title(f"{sheet} — {col} ({lab}, native time)")
                plt.legend()
                plt.tight_layout()
                if xlim is not None:
                    try:
                        plt.xlim(xlim)
                    except Exception:
                        pass
                if want_save:
                    lab_safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in lab)
                    out = Path.cwd() / "saved_plots" / f"{sheet_safe}__{col}__{lab_safe}.png"
                    plt.savefig(out, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out}")
                print(f"Showing figure for {lab}. Close the window to continue...")
                plt.show(block=True)

        again = input("Plot another? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
