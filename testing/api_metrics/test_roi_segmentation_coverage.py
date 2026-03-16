import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ROI coverage ratio and export CSV.")
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "dataset" / "siim-medical-images" / "versions" / "6" / "dicom_dir"),
    )
    parser.add_argument("--glob", default="*.dcm", help="Glob inside --dicom-dir, default *.dcm")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("roi_segmentation_coverage.csv")),
    )
    args = parser.parse_args()

    dicom_files = sorted(Path(args.dataset_dir).glob(args.glob))
    if not dicom_files:
        raise RuntimeError("No DICOM files found in dataset directory")

    rows: List[Dict[str, Any]] = []
    for dicom_path in dicom_files:
        try:
            roi_mask_u8, _, _, roi_model = _extract_roi_mask_and_overlay(str(dicom_path))
            roi_mask = roi_mask_u8.astype(bool)
            roi_pixels = int(roi_mask.sum())
            total_pixels = int(roi_mask.size)
            ratio = float(roi_pixels / total_pixels) if total_pixels > 0 else 0.0

            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "dicom_path": str(dicom_path),
                    "roi_pixels": roi_pixels,
                    "total_pixels": total_pixels,
                    "roi_coverage_fraction": f"{roi_pixels}/{total_pixels}",
                    "roi_coverage_ratio": ratio,
                    "roi_model_used": roi_model,
                    "status": "success",
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "dicom_path": str(dicom_path),
                    "roi_pixels": "",
                    "total_pixels": "",
                    "roi_coverage_fraction": "",
                    "roi_coverage_ratio": "",
                    "roi_model_used": "",
                    "status": "error",
                    "message": str(exc),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "dicom_path",
        "roi_pixels",
        "total_pixels",
        "roi_coverage_fraction",
        "roi_coverage_ratio",
        "roi_model_used",
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
