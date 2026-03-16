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

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # PSNR plot
    fig_psnr, ax_psnr = plt.subplots(figsize=(14, 5))
    ax_psnr.plot(x, df["PSNR without ROI"], label="Mohamed et. al", linewidth=1.5)
    ax_psnr.plot(x, df["PSNR with ROI"], label="Proposed", linewidth=1.5)
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_xlabel("Image Index")
    ax_psnr.set_title("PSNR comparison")
    ax_psnr.grid(alpha=0.3)
    ax_psnr.legend()
    fig_psnr.tight_layout()
    psnr_png = output_png.with_stem(output_png.stem + "_psnr")
    fig_psnr.savefig(psnr_png, dpi=180)
    plt.close(fig_psnr)

    # SSIM plot
    fig_ssim, ax_ssim = plt.subplots(figsize=(14, 5))
    ax_ssim.plot(x, df["SSIM without ROI"], label="Mohamed et. al", linewidth=1.5)
    ax_ssim.plot(x, df["SSIM with ROI"], label="Proposed", linewidth=1.5)
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_xlabel("Image Index")
    ax_ssim.set_title("SSIM comparison")
    ax_ssim.grid(alpha=0.3)
    ax_ssim.legend()
    fig_ssim.tight_layout()
    ssim_png = output_png.with_stem(output_png.stem + "_ssim")
    fig_ssim.savefig(ssim_png, dpi=180)
    plt.close(fig_ssim)

    print(f"Rows plotted: {len(df)}")
    print(f"PSNR plot saved: {psnr_png}")
    print(f"SSIM plot saved: {ssim_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())