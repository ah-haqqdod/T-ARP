# # Centralized management for saving and loading metrics. Metrics are predefined and currently are the same for both benchmarks: kodak and yuv.
# # Supported metrics: SSIM, PSNR, relative error and execution time.

# Centralized management for saving and loading metrics.
# Supported metrics: SSIM, PSNR, relative error, execution time.

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Schema (Parquet format):
# - One file per metric (e.g., ssim.parquet)
# - Columns:
#     0th column: x-axis (e.g., rank) – scalar per row
#     1st to nth column: each method; each cell contains a vector (list) of results
#                         across random seeds (same length for all rows & methods)


def save_metrics(
    save_path: str,
    x_axis: np.ndarray,
    x_label: str,
    ssim_map: Optional[dict] = None,
    psnr_map: Optional[dict] = None,
    relative_error_map: Optional[dict] = None,
    execution_time_map: Optional[dict] = None,
):
    """Save metrics to disk using Parquet format."""
    metrics_path = os.path.join(save_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    def save_one(metric_map: Optional[dict], filename: str):
        if metric_map is None:
            return
        # Build DataFrame: first column = x_axis (scalar per row)
        data = {x_label: x_axis.tolist()}
        for method, values in metric_map.items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            data[method] = values
        df = pd.DataFrame(data)
        df.to_parquet(os.path.join(metrics_path, filename), index=False)

    save_one(ssim_map, "ssim.parquet")
    save_one(psnr_map, "psnr.parquet")
    save_one(relative_error_map, "relative_error.parquet")
    save_one(execution_time_map, "execution_time.parquet")


def load_metrics(
    save_path: str,
) -> Tuple[
    Optional[pd.DataFrame],  # ssim
    Optional[pd.DataFrame],  # psnr
    Optional[pd.DataFrame],  # relative_error
    Optional[pd.DataFrame],  # execution_time
]:
    """Load metric DataFrames from disk (Parquet format).

    Each DataFrame has the x‑axis as the first column (name stored in the column header)
    and one column per method, each containing a list of seed‑wise results per row.
    Returns a tuple of (ssim_df, psnr_df, relative_error_df, execution_time_df),
    where missing files become None.
    """
    # metrics_path = os.path.join(save_path, "metrics")
    metrics_path = save_path

    def load_one(filename: str) -> Optional[pd.DataFrame]:
        path = os.path.join(metrics_path, filename)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    return (
        load_one("ssim.parquet"),
        load_one("psnr.parquet"),
        load_one("relative_error.parquet"),
        load_one("execution_time.parquet"),
    )
