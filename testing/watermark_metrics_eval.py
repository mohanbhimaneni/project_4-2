from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "Source"
TRAINING_DIR = PROJECT_ROOT / "training"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from backend.api import _extract_roi_mask_and_overlay, _load_model_if_needed, _select_working_image  # type: ignore
from image_utils import normalize_image, psnr, ssim  # type: ignore
from watermarking import FragileConfig, embed_fragile_watermark, embed_robust_watermark, payload_to_bits  # type: ignore


def iter_dicom_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() in {".dcm", ".dicom"}:
        yield root
        return

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}:
            yield path


def build_payload_hex(image_path: Path, max_bits: int = 256) -> str:
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
    return digest[: max_bits // 4]


def compute_metrics(original: np.ndarray, watermarked: np.ndarray) -> tuple[float, float]:
    original_norm = normalize_image(original)
    watermarked_norm = normalize_image(watermarked)
    return psnr(original_norm, watermarked_norm), ssim(original_norm, watermarked_norm)


def evaluate_image(image_path: Path) -> dict[str, object]:
    dcm = pydicom.dcmread(str(image_path))
    image_2d, _, _ = _select_working_image(dcm.pixel_array)

    roi_mask, _, _, _ = _extract_roi_mask_and_overlay(str(image_path))
    no_roi_mask = np.zeros_like(roi_mask, dtype=np.uint8)

    payload_hex = build_payload_hex(image_path)

    without_roi_robust = embed_robust_watermark(
        image=image_2d,
        roi_mask=no_roi_mask,
        payload_bits=payload_to_bits(payload_hex),
        strength=1.0,
    )
    without_roi_watermarked = embed_fragile_watermark(
        image=without_roi_robust.astype(image_2d.dtype, copy=False),
        roi_mask=no_roi_mask,
        config=FragileConfig(),
    )

    with_roi_robust = embed_robust_watermark(
        image=image_2d,
        roi_mask=roi_mask.astype(np.uint8),
        payload_bits=payload_to_bits(payload_hex),
        strength=1.0,
    )
    with_roi_watermarked = embed_fragile_watermark(
        image=with_roi_robust.astype(image_2d.dtype, copy=False),
        roi_mask=roi_mask.astype(np.uint8),
        config=FragileConfig(),
    )

    psnr_without_roi, ssim_without_roi = compute_metrics(image_2d, without_roi_watermarked)
    psnr_with_roi, ssim_with_roi = compute_metrics(image_2d, with_roi_watermarked)

    return {
        "image id": str(uuid.uuid4()),
        "image path": str(image_path.resolve()),
        "PSNR without ROI": round(psnr_without_roi, 6),
        "SSIM without ROI": round(ssim_without_roi, 6),
        "PSNR with ROI": round(psnr_with_roi, 6),
        "SSIM with ROI": round(ssim_with_roi, 6),
    }


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image id",
                "image path",
                "PSNR without ROI",
                "SSIM without ROI",
                "PSNR with ROI",
                "SSIM with ROI",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate watermark quality metrics with and without ROI masking for DICOM images.",
    )
    parser.add_argument(
        "input_path",
        help="Path to a DICOM file or directory to scan recursively.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "testing" / "watermark_metrics_results.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to process. 0 means no limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        return 1

    _load_model_if_needed()

    rows: list[dict[str, object]] = []
    processed = 0

    for dicom_path in iter_dicom_files(input_path):
        if args.limit and processed >= args.limit:
            break

        try:
            row = evaluate_image(dicom_path)
            rows.append(row)
            processed += 1
            print(f"Processed {processed}: {dicom_path}")
        except Exception as exc:
            print(f"Skipping {dicom_path}: {exc}", file=sys.stderr)

    output_csv = Path(args.output_csv)
    write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())