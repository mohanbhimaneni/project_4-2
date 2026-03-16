from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot robust watermark extraction confidence as a line chart.",
    )
    parser.add_argument(
        "--input-csv",
        default="d:/Project/SecureDICOM/testing/api_metrics/robust_extraction_accuracy.csv",
        help="Path to robust_extraction_accuracy.csv.",
    )
    parser.add_argument(
        "--output-png",
        default="d:/Project/SecureDICOM/testing/extraction_confidence_plot.png",
        help="Path to output plot image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if "confidence" not in df.columns:
        raise ValueError("CSV is missing 'confidence' column.")

    df = df[df["status"] == "success"].reset_index(drop=True)
    if df.empty:
        raise ValueError("No successful rows found in CSV.")

    x = range(1, len(df) + 1)
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, df["confidence"], linewidth=1.5, color="steelblue", label="Extraction Confidence")
    ax.axhline(y=df["confidence"].mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean = {df['confidence'].mean():.4f}")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Confidence")
    ax.set_title("Robust Watermark Extraction Confidence")
    ax.set_ylim(0.8, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"Rows plotted : {len(df)}")
    print(f"Mean confidence: {df['confidence'].mean():.4f}")
    print(f"Plot saved     : {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
