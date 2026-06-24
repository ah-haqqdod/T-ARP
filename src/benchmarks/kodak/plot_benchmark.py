import os

import pandas as pd
from standalone_benchmark import SAVE_PATH

from t_arp.benchmark import load_metrics, plot_figures

SAVE_DIR = SAVE_PATH

ssim_df, psnr_df, relative_error_df, execution_time_df = load_metrics(
    os.path.join(SAVE_DIR, "metrics")
)

plot_figures(
    ssim_df=ssim_df,
    psnr_df=psnr_df,
    relative_error_df=relative_error_df,
    execution_time_df=execution_time_df,
    save_dir=os.path.join(SAVE_DIR, "figures"),
)
