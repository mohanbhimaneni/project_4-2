from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PSNR/SSIM line comparisons with and without ROI from watermark metrics CSV.",
    )
    parser.add_argument(
        "--input-csv",
        default="d:/Project/SecureDICOM/testing/siim_v6_watermark_metrics.csv",
        help="Path to metrics CSV.",
    )
    parser.add_argument(
        "--output-png",
        default="d:/Project/SecureDICOM/testing/siim_v6_watermark_metrics_plot.png",
        help="Path to output plot image.",
    )
    parser.add_argument(
        "--roi-better-only",
        action="store_true",
        help="Ignore rows where PSNR without ROI is greater than PSNR with ROI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = [
        "PSNR without ROI",
        "SSIM without ROI",
        "PSNR with ROI",
        "SSIM with ROI",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    if args.roi_better_only:
        df = df[df["PSNR with ROI"] >= df["PSNR without ROI"]].reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows left after applying --roi-better-only filter.")

    x = range(1, len(df) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, df["PSNR without ROI"], label="PSNR without ROI", linewidth=1.5)
    axes[0].plot(x, df["PSNR with ROI"], label="PSNR with ROI", linewidth=1.5)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, df["SSIM without ROI"], label="SSIM without ROI", linewidth=1.5)
    axes[1].plot(x, df["SSIM with ROI"], label="SSIM with ROI", linewidth=1.5)
    axes[1].set_ylabel("SSIM")
    axes[1].set_xlabel("Image Index")
    axes[1].set_title("SSIM Comparison")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Watermark Quality Metrics With vs Without ROI", fontsize=14, y=0.98)
    fig.tight_layout()

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"Rows plotted: {len(df)}")
    print(f"Plot saved: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())