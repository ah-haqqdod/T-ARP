#!/usr/bin/env python3
"""
Summarise metric Parquet files (same format as the plotting script):
  - First column: x-axis (e.g., rank)
  - Other columns: method names, each cell is a scalar or a list/array of samples.

For each input metric, produce:
  1) A CSV file with per‑row mean and standard deviation (0 for single sample).
  2) A LaTeX table where cells are either a single value or 'mean ± std'.

The script can either scan a directory for files containing keywords
('ssim', 'psnr', 'relative_error', 'execution_time') or accept explicit file paths.
"""

import argparse
import glob
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Helper: ensure directory exists
def _ensure_save_dir(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)


# ----------------------------------------------------------------------
# Convert DataFrame to mean/std summary (CSV)
def expand_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame where method columns contain scalars or lists/arrays
    into a DataFrame with {method}_mean and {method}_std per row.
    """
    if df.empty:
        return df

    x_col = df.columns[0]
    result = pd.DataFrame({x_col: df.iloc[:, 0].values})

    for col in df.columns[1:]:
        means = []
        stds = []
        for cell in df[col].values:
            if cell is None:
                means.append(np.nan)
                stds.append(np.nan)
                continue

            # Force cell to be a 1D numpy array
            if isinstance(cell, (list, np.ndarray)):
                arr = np.asarray(cell).ravel()
            else:
                arr = np.array([cell])

            if arr.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(arr))
                stds.append(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)

        result[f"{col}_mean"] = means
        result[f"{col}_std"] = stds
    return result


# ----------------------------------------------------------------------
# LaTeX table generation (with automatic mean±std or single value)
def create_latex_tables(
    dfs: Dict[str, pd.DataFrame],
    save_dir: str = "latex_tables",
    float_format: str = "%.3f",
) -> None:
    """
    For each metric DataFrame, produce a LaTeX table where each cell
    contains either a single value (if one sample) or 'mean ± std' (if multiple).

    Parameters
    ----------
    dfs : dict
        Keys: metric name ('ssim', 'psnr', 'relative_error', 'execution_time')
        Values: corresponding DataFrame (same format as plotting script)
    save_dir : str
        Directory where .tex files will be saved.
    float_format : str
        Format string for floating point numbers.
    """
    _ensure_save_dir(save_dir)

    def format_cell(cell, fmt=float_format):
        if cell is None:
            return "--"
        if isinstance(cell, (list, np.ndarray)):
            arr = np.asarray(cell).ravel()
        else:
            arr = np.array([cell])

        if arr.size == 0:
            return "--"
        if arr.size == 1:
            return fmt % arr[0]
        else:
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            return r"%s $\pm$ %s" % (fmt % mean, fmt % std)

    for name, df in dfs.items():
        if df is None or df.empty:
            continue

        x_col = df.columns[0]
        # Build formatted DataFrame
        formatted_data = {x_col: df[x_col].values}
        for col in df.columns[1:]:
            formatted_data[col] = [format_cell(cell) for cell in df[col].values]

        formatted_df = pd.DataFrame(formatted_data)

        # Generate LaTeX table
        latex_str = formatted_df.to_latex(
            index=False,
            escape=False,
            column_format="l" + "c" * (len(formatted_df.columns) - 1),
            caption=f"{name.upper()} values (mean ± std where multiple samples exist)",
            label=f"tab:{name.lower()}",
            float_format=float_format,
        )
        out_path = os.path.join(save_dir, f"{name}_table.tex")
        with open(out_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {out_path}")


# ----------------------------------------------------------------------
# Main summarisation function (CSV + LaTeX)
def create_summary_and_latex(
    ssim_df: Optional[pd.DataFrame] = None,
    psnr_df: Optional[pd.DataFrame] = None,
    relative_error_df: Optional[pd.DataFrame] = None,
    execution_time_df: Optional[pd.DataFrame] = None,
    save_dir_csv: str = "summary_tables",
    save_dir_latex: str = "latex_tables",
) -> None:
    """
    For each provided metric DataFrame:
      1) Compute per‑row mean and std and save as CSV.
      2) Generate a LaTeX table with single values or 'mean ± std'.
    """
    # CSV summaries
    _ensure_save_dir(save_dir_csv)

    if ssim_df is not None:
        out = expand_method_columns(ssim_df)
        out.to_csv(os.path.join(save_dir_csv, "ssim_summary.csv"), index=False)
        print(f"Saved SSIM summary to {save_dir_csv}/ssim_summary.csv")
    if psnr_df is not None:
        out = expand_method_columns(psnr_df)
        out.to_csv(os.path.join(save_dir_csv, "psnr_summary.csv"), index=False)
        print(f"Saved PSNR summary to {save_dir_csv}/psnr_summary.csv")
    if relative_error_df is not None:
        out = expand_method_columns(relative_error_df)
        out.to_csv(
            os.path.join(save_dir_csv, "relative_error_summary.csv"), index=False
        )
        print(
            f"Saved relative error summary to {save_dir_csv}/relative_error_summary.csv"
        )
    if execution_time_df is not None:
        out = expand_method_columns(execution_time_df)
        out.to_csv(
            os.path.join(save_dir_csv, "execution_time_summary.csv"), index=False
        )
        print(
            f"Saved execution time summary to {save_dir_csv}/execution_time_summary.csv"
        )

    # LaTeX tables
    dfs = {
        "ssim": ssim_df,
        "psnr": psnr_df,
        "relative_error": relative_error_df,
        "execution_time": execution_time_df,
    }
    create_latex_tables(dfs, save_dir_latex)

    print(
        f"All summaries saved in '{save_dir_csv}' and LaTeX tables in '{save_dir_latex}'"
    )


# ----------------------------------------------------------------------
# Automatic file discovery in a directory
def find_metric_files(directory: str) -> Dict[str, Optional[str]]:
    """
    Scan `directory` for Parquet files containing specific keywords.
    Returns a dict with keys: ssim, psnr, relative_error, execution_time
    and values: file path or None if not found.
    """
    files = glob.glob(os.path.join(directory, "*.parquet"))
    found = {
        "ssim": None,
        "psnr": None,
        "relative_error": None,
        "execution_time": None,
    }
    for f in files:
        base = os.path.basename(f).lower()
        if "ssim" in base and found["ssim"] is None:
            found["ssim"] = f
        elif "psnr" in base and found["psnr"] is None:
            found["psnr"] = f
        elif "relative_error" in base and found["relative_error"] is None:
            found["relative_error"] = f
        elif "execution_time" in base and found["execution_time"] is None:
            found["execution_time"] = f
    return found


# ----------------------------------------------------------------------
# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summary CSV and LaTeX tables from a directory of metric Parquet files."
    )
    # Accepts either flag and maps the result to args.load_dir
    parser.add_argument(
        "--load-dir",
        dest="load_dir",
        required=True,
        help="Directory containing the metric Parquet files (scanned for keywords: ssim, psnr, relative_error, execution_time)",
    )
    parser.add_argument(
        "--save-dir-csv",
        default=None,
        help="Directory for CSV summaries (default: same as load/metrics directory)",
    )
    parser.add_argument(
        "--save-dir-latex",
        default=None,
        help="Directory for LaTeX tables (default: same as load/metrics directory)",
    )

    args = parser.parse_args()

    # Determine resolution directories
    base_dir = args.load_dir
    save_csv = args.save_dir_csv if args.save_dir_csv is not None else base_dir
    save_latex = args.save_dir_latex if args.save_dir_latex is not None else base_dir

    # Define target metrics and find matching files in the directory
    metric_keys = ["ssim", "psnr", "relative_error", "execution_time"]
    metric_paths = {key: None for key in metric_keys}

    if os.path.isdir(base_dir):
        for filename in os.listdir(base_dir):
            if filename.endswith(".parquet"):
                for key in metric_keys:
                    if key in filename.lower():
                        metric_paths[key] = os.path.join(base_dir, filename)

    # Cleanly load dataframes into a dictionary
    dfs = {
        key: pd.read_parquet(path) if path else None
        for key, path in metric_paths.items()
    }

    # Generate outputs
    create_summary_and_latex(
        ssim_df=dfs["ssim"],
        psnr_df=dfs["psnr"],
        relative_error_df=dfs["relative_error"],
        execution_time_df=dfs["execution_time"],
        save_dir_csv=save_csv,
        save_dir_latex=save_latex,
    )
