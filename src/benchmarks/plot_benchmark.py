import argparse
import os

from plot_statistics import plot_figures

from t_arp.benchmark import load_metrics


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Plot metrics from a specified directory."
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        type=str,
        required=True,
        help="Root directory containing the 'metrics' subfolder.",
    )
    args = parser.parse_args()

    # Define paths based on the provided argument
    metrics_path = os.path.join(args.save_dir, "metrics")
    figures_path = os.path.join(args.save_dir, "figures")

    # Load metrics
    ssim_df, psnr_df, relative_error_df, execution_time_df = load_metrics(metrics_path)

    # Plot figures
    plot_figures(
        ssim_df=ssim_df,
        psnr_df=psnr_df,
        relative_error_df=relative_error_df,
        execution_time_df=execution_time_df,
        save_dir=figures_path,
    )


if __name__ == "__main__":
    main()
