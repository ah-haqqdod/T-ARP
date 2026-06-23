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


# ---
# import os
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd

# # Data schema
# # columns: 0-th column is the x-axis (e.g., rank)
# #          1-st to n-th column is the y-axis as a vector (e.g., metric value)
# # file name as the metric name


# # save metrics
# def save_metrics(
#     save_path: str,
#     x_axis: np.ndarray,
#     x_label: str,
#     ssim_map: Optional[dict] = None,
#     psnr_map: Optional[dict] = None,
#     relative_error_map: Optional[dict] = None,
#     execution_time_map: Optional[dict] = None,
# ):
#     """Save the metrics to disk."""
#     # create metrics directory if it doesn't exist
#     metrics_path = os.path.join(save_path, "metrics")
#     os.makedirs(metrics_path, exist_ok=True)
#     # save metrics to disk

#     np.savez(os.path.join(metrics_path, "x_axis.npy"), x_axis=x_axis, x_label=x_label)

#     if ssim_map is not None:
#         np.savez(
#             os.path.join(metrics_path, "ssim_map.npz"),
#             **ssim_map,
#         )
#     if psnr_map is not None:
#         np.savez(
#             os.path.join(metrics_path, "psnr_map.npz"),
#             **psnr_map,
#         )
#     if relative_error_map is not None:
#         np.savez(
#             os.path.join(metrics_path, "relative_error_map.npz"),
#             **relative_error_map,
#         )
#     if execution_time_map is not None:
#         np.savez(
#             os.path.join(metrics_path, "execution_time_map.npz"),
#             **execution_time_map,
#         )


# # load metrics
# def load_metrics(
#     save_path: str,
# ) -> Tuple[
#     np.ndarray, str, Optional[dict], Optional[dict], Optional[dict], Optional[dict]
# ]:
#     """Load the metrics from disk."""
#     metrics_path = os.path.join(save_path, "metrics")

#     x_axis = None
#     x_label = None
#     ssim_map = None
#     psnr_map = None
#     relative_error_map = None
#     execution_time_map = None

#     if os.path.exists(os.path.join(metrics_path, "x_axis.npz")):
#         x_axis = np.load(os.path.join(metrics_path, "x_axis.npz"))
#         x_axis = x_axis["x_axis"]
#         x_label = x_axis["x_label"]
#     else:
#         raise FileNotFoundError(f"x_axis.npz not found in {metrics_path}")

#     if os.path.exists(os.path.join(metrics_path, "ssim_map.npz")):
#         ssim_map = np.load(os.path.join(metrics_path, "ssim_map.npz"))

#     if os.path.exists(os.path.join(metrics_path, "psnr_map.npz")):
#         psnr_map = np.load(os.path.join(metrics_path, "psnr_map.npz"))

#     if os.path.exists(os.path.join(metrics_path, "relative_error_map.npz")):
#         relative_error_map = np.load(
#             os.path.join(metrics_path, "relative_error_map.npz")
#         )

#     if os.path.exists(os.path.join(metrics_path, "execution_time_map.npz")):
#         execution_time_map = np.load(
#             os.path.join(metrics_path, "execution_time_map.npz")
#         )

#     if all(
#         m is None for m in [ssim_map, psnr_map, relative_error_map, execution_time_map]
#     ):
#         raise FileNotFoundError(
#             f"No relevant metric .npz files found in {metrics_path}"
#         )

#     return x_axis, x_label, ssim_map, psnr_map, relative_error_map, execution_time_map
