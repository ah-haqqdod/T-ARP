#!/usr/bin/env python3
"""
Centralized management for saving and loading metrics (SSIM, PSNR, relative error,
execution time) using Parquet format. Each metric file has:

  - First column: x-axis (e.g., rank) – scalar per row
  - Remaining columns: method names, each cell contains either:
        * a scalar (single deterministic value)  →  plotted as a line
        * a list/vector of length ≥ 2 (multiple seeds) → plotted as mean ± std band

Plotting functions automatically detect the data type and choose the appropriate
visualisation.
"""

import os
from typing import Optional, Tuple

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
#  High‑quality plot settings
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = (8, 5)
# ----------------------------------------------------------------------


# ======================================================================
#  Plotting helpers
# ======================================================================


def _ensure_save_dir(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if not os.access(save_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {save_dir}")


def _prepare_plot_data(
    values: np.ndarray, x_axis: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Convert 1D or 2D values to mean, std, and align x_axis.
    Returns (mean, std, x_aligned). std is None for 1D input.
    """
    values = np.asarray(values)
    x_axis = np.asarray(x_axis, dtype=np.int32)

    if values.ndim == 1:
        mean = values
        std = None
        n_points = len(values)
    elif values.ndim == 2:
        mean = np.mean(values, axis=1)
        # mean = np.mean(values)
        # std = np.std(values, axis=0, ddof=1)
        std = None
        n_points = values.shape[1]
    else:
        raise ValueError(f"Unsupported array dimension: {values.ndim}")

    # if len(x_axis) > n_points:
    #     x_axis = x_axis[:n_points]
    #     print(f"Warning: x_axis trimmed to {n_points} points")
    # elif len(x_axis) < n_points:
    #     raise ValueError(f"x_axis length ({len(x_axis)}) < data points ({n_points})")

    return mean, std, x_axis


def _extract_method_arrays(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame method columns into numpy arrays.

    - Column of scalars → 1D array (shape = n_points)
    - Column of length‑1 lists → 1D array (scalar per point)
    - Column of length ≥2 lists → 2D array (shape = n_points × n_seeds)
    """
    method_cols = df.columns[1:]
    method_arrays = {}

    for col in method_cols:
        col_data = df[col].values  # 1D array of objects

        # Find first non‑None element to decide type
        first_valid = next((x for x in col_data if x is not None), None)
        if first_valid is None:
            continue

        if isinstance(first_valid, (list, np.ndarray)):
            # Column of lists
            # Determine the maximum list length
            lengths = [
                len(x) if isinstance(x, (list, np.ndarray)) else 1 for x in col_data
            ]
            max_len = max(lengths)

            if max_len == 1:
                # Single‑element lists: convert to 1D array of scalars
                arr = np.array(
                    [x[0] if isinstance(x, (list, np.ndarray)) else x for x in col_data]
                )
            else:
                # Multi‑element lists: stack into 2D array
                try:
                    arr = np.vstack(col_data)  # shape (n_points, n_seeds)
                except ValueError:
                    # Fallback for inconsistent lengths (should not happen)
                    arr = np.array([np.asarray(row) for row in col_data], dtype=object)
        else:
            # Column of scalars
            arr = col_data.astype(float)

        method_arrays[col] = arr

    return method_arrays


def _plot_with_errorbar(
    x: np.ndarray,
    mean: np.ndarray,
    std: Optional[np.ndarray],
    label: str,
    marker: str,
    color: Optional[str] = None,
) -> None:
    """Line plot with optional shaded error band."""
    line = plt.plot(x, mean, label=label, marker=marker, markersize=4, color=color)
    plt.xticks(x)
    if std is not None:
        color_used = line[0].get_color()
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color_used,
            linewidth=0,
        )


# ======================================================================
#  Public plotting functions (one per metric)
# ======================================================================


def plot_ssim(df: Optional[pd.DataFrame], save_dir: str = "figures") -> None:
    """Plot SSIM from a DataFrame (first column = x‑axis)."""
    if df is None:
        return
    _ensure_save_dir(save_dir)
    plt.figure()

    x_axis = df.iloc[:, 0].values
    x_label = df.columns[0]
    method_arrays = _extract_method_arrays(df)

    for method, values in method_arrays.items():
        mean, std, x_plot = _prepare_plot_data(values, x_axis)
        _plot_with_errorbar(x_plot, mean, std, method, marker="o")

    plt.xlabel(x_label)
    plt.ylabel("SSIM")
    plt.title("Structural Similarity Index (SSIM) – Higher is better")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "ssim_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"SSIM plot saved to {save_path}")
    # save_path = os.path.join(save_dir, "ssim_plot.tex")
    # matplot2tikz.save(save_path)
    # plt.close()


def plot_psnr(df: Optional[pd.DataFrame], save_dir: str = "figures") -> None:
    """Plot PSNR from a DataFrame."""
    if df is None:
        return
    _ensure_save_dir(save_dir)
    plt.figure()

    x_axis = df.iloc[:, 0].values
    x_label = df.columns[0]
    method_arrays = _extract_method_arrays(df)

    for method, values in method_arrays.items():
        mean, std, x_plot = _prepare_plot_data(values, x_axis)
        _plot_with_errorbar(x_plot, mean, std, method, marker="s")

    plt.xlabel(x_label)
    plt.ylabel("PSNR (dB)")
    plt.title("Peak Signal‑to‑Noise Ratio (PSNR) – Higher is better")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "psnr_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"PSNR plot saved to {save_path}")
    # save_path = os.path.join(save_dir, "psnr_plot.tex")
    # matplot2tikz.save(save_path)
    # plt.close()


def plot_relative_error(df: Optional[pd.DataFrame], save_dir: str = "figures") -> None:
    """Plot relative error from a DataFrame."""
    if df is None:
        return
    _ensure_save_dir(save_dir)
    plt.figure()

    x_axis = df.iloc[:, 0].values
    x_label = df.columns[0]
    method_arrays = _extract_method_arrays(df)

    for method, values in method_arrays.items():
        mean, std, x_plot = _prepare_plot_data(values, x_axis)
        _plot_with_errorbar(x_plot, mean, std, method, marker="^")

    plt.xlabel(x_label)
    # plt.ylabel("Relative Error")
    plt.ylabel("Relative Error (log scale)")
    plt.yscale("log")  # NOTE: remove
    plt.title("Reconstruction Relative Error – Lower is better")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "relative_error_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Relative error plot saved to {save_path}")
    # save_path = os.path.join(save_dir, "relative_error_plot.tex")
    # matplot2tikz.save(save_path)
    # plt.close()


def plot_execution_time(df: Optional[pd.DataFrame], save_dir: str = "figures") -> None:
    """Plot execution time from a DataFrame."""
    if df is None:
        return
    _ensure_save_dir(save_dir)
    plt.figure()

    x_axis = df.iloc[:, 0].values
    x_label = df.columns[0]
    method_arrays = _extract_method_arrays(df)

    for method, values in method_arrays.items():
        mean, std, x_plot = _prepare_plot_data(values, x_axis)
        _plot_with_errorbar(x_plot, mean, std, method, marker="d")

    plt.xlabel(x_label)
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time – Lower is better")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "execution_time_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Execution time plot saved to {save_path}")
    # save_path = os.path.join(save_dir, "execution_time_plot.tex")
    # matplot2tikz.save(save_path)
    # plt.close()


def plot_figures(
    ssim_df: Optional[pd.DataFrame] = None,
    psnr_df: Optional[pd.DataFrame] = None,
    relative_error_df: Optional[pd.DataFrame] = None,
    execution_time_df: Optional[pd.DataFrame] = None,
    save_dir: str = "figures",
) -> None:
    """
    Generate all available plots from DataFrames returned by `load_metrics()`.
    """
    _ensure_save_dir(save_dir)

    if ssim_df is not None:
        plot_ssim(ssim_df, save_dir)
    if psnr_df is not None:
        plot_psnr(psnr_df, save_dir)
    if relative_error_df is not None:
        plot_relative_error(relative_error_df, save_dir)
    if execution_time_df is not None:
        plot_execution_time(execution_time_df, save_dir)

    print(f"All requested plots saved in '{save_dir}'")
