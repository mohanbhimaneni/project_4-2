import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Source"
BACKEND_DIR = SOURCE_DIR / "backend"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.wm_common import _extract_roi_mask_and_overlay  # type: ignore
from watermarking.robust_watermark import (  # type: ignore
    bits_to_payload,
    embed_robust_watermark,
    extract_robust_watermark,
    payload_to_bits,
)


def collect_dicom_files(dataset_dir: Path, pattern: str) -> List[Path]:
    return sorted(p for p in dataset_dir.glob(pattern) if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate robust watermark extraction confidence on local DICOM dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "dataset" / "siim-medical-images" / "versions" / "6" / "dicom_dir"),
    )
    parser.add_argument("--glob", default="*.dcm")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("robust_extraction_accuracy.csv")),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dicom_files = collect_dicom_files(dataset_dir, args.glob)
    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in {dataset_dir} matching {args.glob}")

    rows: List[Dict[str, Any]] = []

    for dcm_path in dicom_files:
        try:
            roi_mask_u8, _, image_2d, roi_model = _extract_roi_mask_and_overlay(str(dcm_path))
            roi_mask = roi_mask_u8.astype(bool)

            payload_hex = hashlib.sha256(str(dcm_path).encode("utf-8")).hexdigest()[:64]
            payload_bits = payload_to_bits(payload_hex)

            watermarked = embed_robust_watermark(
                image=image_2d,
                roi_mask=roi_mask,
                payload_bits=payload_bits,
                strength=1.0,
            )

            extracted_bits, confidence = extract_robust_watermark(
                image=watermarked,
                roi_mask=roi_mask,
                payload_length=len(payload_bits),
            )

            extracted_hex = bits_to_payload(extracted_bits, bit_length=len(payload_bits))
            compare_len = min(len(payload_bits), len(extracted_bits))
            bit_correct = sum(
                1 for idx in range(compare_len) if extracted_bits[idx] == payload_bits[idx]
            )
            bit_accuracy = float(bit_correct / compare_len) if compare_len > 0 else 0.0

            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "dicom_path": str(dcm_path),
                    "roi_model_used": roi_model,
                    "payload_hex_length": len(payload_hex),
                    "extracted_hex_length": len(extracted_hex),
                    "hex_match": extracted_hex == payload_hex,
                    "bit_accuracy": bit_accuracy,
                    "confidence": float(confidence),
                    "status": "success",
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "dicom_path": str(dcm_path),
                    "roi_model_used": "",
                    "payload_hex_length": "",
                    "extracted_hex_length": "",
                    "hex_match": "",
                    "bit_accuracy": "",
                    "confidence": "",
                    "status": "error",
                    "message": str(exc),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "dicom_path",
        "roi_model_used",
        "payload_hex_length",
        "extracted_hex_length",
        "hex_match",
        "bit_accuracy",
        "confidence",
        "status",
        "message",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
