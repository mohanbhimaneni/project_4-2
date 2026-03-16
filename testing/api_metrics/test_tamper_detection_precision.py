import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Source"
BACKEND_DIR = SOURCE_DIR / "backend"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.wm_common import _extract_roi_mask_and_overlay  # type: ignore
from watermarking.fragile_watermark import (  # type: ignore
    FragileConfig,
    detect_tamper_map,
    embed_fragile_watermark,
)


def expected_attack_mask(roi_mask_bool: np.ndarray) -> np.ndarray:
    non_roi_positions = np.argwhere(~roi_mask_bool)
    if non_roi_positions.size == 0:
        raise RuntimeError("No non-ROI region available for expected attack mask")

    row, col = non_roi_positions[len(non_roi_positions) // 2]
    r0, r1 = max(0, row - 8), min(roi_mask_bool.shape[0], row + 8)
    c0, c1 = max(0, col - 8), min(roi_mask_bool.shape[1], col + 8)

    gt = np.zeros_like(roi_mask_bool, dtype=bool)
    gt[r0:r1, c0:c1] = True
    return gt


def apply_attack(image_2d: np.ndarray, roi_mask_bool: np.ndarray, attack_type: str) -> np.ndarray:
    attacked = image_2d.copy()
    non_roi_positions = np.argwhere(~roi_mask_bool)
    if non_roi_positions.size == 0:
        raise RuntimeError("No non-ROI region available for attack")

    row, col = non_roi_positions[len(non_roi_positions) // 2]
    r0, r1 = max(0, row - 8), min(attacked.shape[0], row + 8)
    c0, c1 = max(0, col - 8), min(attacked.shape[1], col + 8)

    if attack_type == "zero":
        attacked[r0:r1, c0:c1] = 0
    else:
        noise = np.random.normal(0, 35, size=(r1 - r0, c1 - c0))
        region = attacked[r0:r1, c0:c1].astype(np.float32) + noise
        region = np.clip(region, float(image_2d.min()), float(image_2d.max()))
        attacked[r0:r1, c0:c1] = region.astype(image_2d.dtype)

    return attacked


def precision_score(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())

    denom = tp + fp
    precision = float(tp / denom) if denom > 0 else 0.0
    return tp, fp, fn, precision


def collect_dicom_files(dicom_paths: List[str], dicom_dir: str, pattern: str) -> List[Path]:
    files: List[Path] = []
    for p in dicom_paths:
        path = Path(p)
        if path.exists() and path.is_file():
            files.append(path)
    if dicom_dir:
        files.extend(sorted(Path(dicom_dir).glob(pattern)))
    unique = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tamper detection precision and export CSV.")
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "dataset" / "siim-medical-images" / "versions" / "6" / "dicom_dir"),
    )
    parser.add_argument("--glob", default="*.dcm", help="Glob inside --dicom-dir, default *.dcm")
    parser.add_argument("--attack-types", default="noise,zero", help="Comma-separated attack types")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("tamper_detection_precision.csv")),
    )
    args = parser.parse_args()

    dicom_files = sorted(Path(args.dataset_dir).glob(args.glob))
    if not dicom_files:
        raise RuntimeError("No DICOM files found in dataset directory")

    attack_types = [a.strip().lower() for a in args.attack_types.split(",") if a.strip()]
    if not attack_types:
        raise RuntimeError("No attack types provided")

    rows: List[Dict[str, Any]] = []
    for dicom_path in dicom_files:
        try:
            roi_mask_u8, _, image_2d, roi_model = _extract_roi_mask_and_overlay(str(dicom_path))
            roi_mask = roi_mask_u8.astype(bool)

            watermarked = embed_fragile_watermark(
                image=image_2d,
                roi_mask=roi_mask,
                config=FragileConfig(),
            )

            gt_mask = expected_attack_mask(roi_mask)

            for attack_type in attack_types:
                attacked = apply_attack(watermarked, roi_mask, attack_type)
                tamper_map = detect_tamper_map(
                    image=attacked,
                    roi_mask=roi_mask,
                    config=FragileConfig(),
                ).astype(bool)

                tp, fp, fn, precision = precision_score(tamper_map, gt_mask)
                rows.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "dicom_path": str(dicom_path),
                        "roi_model_used": roi_model,
                        "attack_type": attack_type,
                        "tp_pixels": tp,
                        "fp_pixels": fp,
                        "fn_pixels": fn,
                        "precision": precision,
                        "status": "success",
                        "error": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "dicom_path": str(dicom_path),
                    "roi_model_used": "",
                    "attack_type": "",
                    "tp_pixels": "",
                    "fp_pixels": "",
                    "fn_pixels": "",
                    "precision": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "dicom_path",
        "roi_model_used",
        "attack_type",
        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
        "precision",
        "status",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
