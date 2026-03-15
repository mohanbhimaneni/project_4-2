from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
from flask import g, jsonify, request
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SOURCE_DIR.parent
TRAINING = PROJECT_ROOT / "training"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(TRAINING))

from image_utils import normalize_image  # noqa: E402

try:
    from .database_api import create_audit_log_entry, is_study_shared_with_user
except ImportError:
    from database_api import create_audit_log_entry, is_study_shared_with_user  # type: ignore

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"dcm", "dicom"}
CHECKPOINT_PATH = Path(
    os.environ.get(
        "SECUREDICOM_CHECKPOINT",
        str(TRAINING / "checkpoints" / "midi_b_modality_vit_tiny_next" / "best.pt"),
    )
)
CHEST_CHECKPOINT_PATH = Path(
    os.environ.get(
        "SECUREDICOM_CHEST_CHECKPOINT",
        str(TRAINING / "checkpoints" / "rider_vit_seg" / "best.pt"),
    )
)
MODEL_NAME = "vit_tiny_patch16_224"
DEVICE = "cpu"

_cached_model = None
_cached_device = None


def _is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _select_working_image(pixels: np.ndarray) -> tuple[np.ndarray, int | None, str]:
    arr = np.asarray(pixels)

    if arr.ndim == 2:
        return arr, None, "2d"

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            gray = np.mean(arr[..., :3], axis=-1).astype(arr.dtype)
            return gray, None, "color2d"
        frame_idx = arr.shape[0] // 2
        return arr[frame_idx], frame_idx, "volume3d"

    if arr.ndim == 4:
        frame_idx = arr.shape[0] // 2
        frame = arr[frame_idx]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            gray = np.mean(frame[..., :3], axis=-1).astype(arr.dtype)
            return gray, frame_idx, "volume4d_color"
        return frame, frame_idx, "volume4d"

    raise ValueError(f"Unsupported pixel array dimensions: {arr.shape}")


def _write_watermarked_back(
    original_pixels: np.ndarray,
    watermarked_2d: np.ndarray,
    frame_idx: int | None,
    mode: str,
) -> np.ndarray:
    arr = np.array(original_pixels, copy=True)
    wm = watermarked_2d.astype(arr.dtype, copy=False)

    if mode == "2d":
        return wm
    if mode == "volume3d" and frame_idx is not None:
        arr[frame_idx] = wm
        return arr

    raise ValueError(f"Unsupported DICOM mode for writeback: {mode}")


def _load_model_if_needed() -> None:
    global _cached_model, _cached_device

    if _cached_model is not None:
        return

    from roi import _get_vit_model  # type: ignore

    _cached_model, _cached_device = _get_vit_model(
        model_name=MODEL_NAME,
        checkpoint=str(CHECKPOINT_PATH),
        device=DEVICE,
    )

    if CHEST_CHECKPOINT_PATH != CHECKPOINT_PATH and CHEST_CHECKPOINT_PATH.exists():
        _get_vit_model(
            model_name=MODEL_NAME,
            checkpoint=str(CHEST_CHECKPOINT_PATH),
            device=DEVICE,
        )


def _is_chest_like_study(dcm: Any) -> bool:
    text_parts = [
        str(getattr(dcm, "BodyPartExamined", "") or ""),
        str(getattr(dcm, "StudyDescription", "") or ""),
        str(getattr(dcm, "SeriesDescription", "") or ""),
        str(getattr(dcm, "ProtocolName", "") or ""),
    ]
    haystack = " ".join(text_parts).lower()
    keywords = ("chest", "thorax", "lung", "pulmonary", "cxr")
    return any(keyword in haystack for keyword in keywords)


def _roi_quality_score(mask: np.ndarray) -> float:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return -1e9

    h, w = mask_bool.shape
    coverage = float(mask_bool.mean())

    rr, cc = np.where(mask_bool)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    max_dist = float(np.sqrt(cy ** 2 + cx ** 2)) + 1e-6
    center_score = 1.0 - float(np.mean(dist) / max_dist)

    target_coverage = 0.30
    coverage_score = 1.0 - min(abs(coverage - target_coverage) / target_coverage, 1.2)
    return 0.65 * center_score + 0.35 * coverage_score


def _extract_roi_mask_with_checkpoint(image_2d: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    from roi import _cluster_roi, _get_patch_embeddings, _normalize_image  # type: ignore

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ROI checkpoint not found: {checkpoint_path}")

    image_norm = _normalize_image(image_2d)
    patch_features = _get_patch_embeddings(
        image_norm,
        model_name=MODEL_NAME,
        checkpoint=str(checkpoint_path),
        device=DEVICE,
    )
    roi_mask = _cluster_roi(patch_features, image_norm)
    if roi_mask is None:
        raise RuntimeError("ROI extraction failed")
    return roi_mask.astype(np.uint8)


def _extract_roi_mask_and_overlay(dicom_path: str) -> tuple[np.ndarray, Image.Image, np.ndarray, str]:

    dcm = pydicom.dcmread(dicom_path)
    pixels = dcm.pixel_array
    image_2d, _, _ = _select_working_image(pixels)

    chest_like = _is_chest_like_study(dcm)
    candidate_order = [
        ("general", CHECKPOINT_PATH),
        ("chest", CHEST_CHECKPOINT_PATH),
    ]

    seen_paths: set[str] = set()
    candidates: list[tuple[str, Path, np.ndarray, float, float]] = []

    for candidate_name, candidate_path in candidate_order:
        resolved = str(candidate_path.resolve()) if candidate_path.exists() else str(candidate_path)
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        try:
            candidate_mask = _extract_roi_mask_with_checkpoint(image_2d, candidate_path)
            candidate_score = _roi_quality_score(candidate_mask)
            candidate_coverage = float(candidate_mask.astype(bool).mean())
            candidates.append((candidate_name, candidate_path, candidate_mask, candidate_score, candidate_coverage))
        except Exception as exc:
            logger.warning("ROI candidate '%s' failed: %s", candidate_name, exc)

    if not candidates:
        raise RuntimeError("ROI extraction failed for all configured checkpoints")

    valid_by_coverage = [candidate for candidate in candidates if 0.05 <= candidate[4] <= 0.80]

    if valid_by_coverage:
        selected_name, selected_checkpoint, roi_mask, selected_score, selected_coverage = max(
            valid_by_coverage,
            key=lambda x: (x[4], x[3]),
        )
    else:
        selected_name, selected_checkpoint, roi_mask, selected_score, selected_coverage = max(
            candidates,
            key=lambda x: x[3],
        )

    logger.info(
        "ROI selector: chest_like=%s selected=%s checkpoint=%s score=%.4f coverage=%.4f",
        chest_like,
        selected_name,
        selected_checkpoint,
        selected_score,
        selected_coverage,
    )

    base_uint8 = (normalize_image(image_2d) * 255.0).astype(np.uint8)
    rgb = np.stack([base_uint8, base_uint8, base_uint8], axis=-1)

    roi_bool = roi_mask.astype(bool)
    overlay = rgb.copy()
    overlay[roi_bool] = [255, 80, 80]
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)

    return roi_mask.astype(np.uint8), Image.fromarray(blended, mode="RGB"), image_2d, selected_name


def _count_non_roi_patches(roi_mask: np.ndarray, patch_size: int = 8, patch_stride: int = 8) -> int:
    h, w = roi_mask.shape
    total = 0
    for i in range(0, h - patch_size + 1, patch_stride):
        for j in range(0, w - patch_size + 1, patch_stride):
            patch_roi = roi_mask[i : i + patch_size, j : j + patch_size]
            roi_coverage = patch_roi.sum() / (patch_size * patch_size)
            if roi_coverage < 1.0:
                total += 1
    return total


def _png_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _array_to_base64_png(image_2d: np.ndarray) -> str:
    image_uint8 = (normalize_image(image_2d) * 255.0).astype(np.uint8)
    img = Image.fromarray(image_uint8, mode="L")
    return _png_to_base64(img)


def _mask_to_base64_png(mask: np.ndarray) -> str:
    mask_uint8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    return _png_to_base64(Image.fromarray(mask_uint8, mode="L"))


def _decode_data_url_png(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image data format")
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw)).convert("L")


def _overlay_from_image_and_mask(image_2d: np.ndarray, roi_mask: np.ndarray) -> Image.Image:
    base_uint8 = (normalize_image(image_2d) * 255.0).astype(np.uint8)
    rgb = np.stack([base_uint8, base_uint8, base_uint8], axis=-1)
    overlay = rgb.copy()
    overlay[roi_mask.astype(bool)] = [255, 80, 80]
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _original_preview_bytes(original_dcm_path: str) -> bytes:
    dcm = pydicom.dcmread(original_dcm_path)
    image_2d, _, _ = _select_working_image(dcm.pixel_array)
    image_uint8 = (normalize_image(image_2d) * 255.0).astype(np.uint8)
    img = Image.fromarray(image_uint8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _study_dir(storage_root: Path, study_id: str) -> Path:
    study_path = storage_root / "studies" / study_id
    study_path.mkdir(parents=True, exist_ok=True)
    return study_path


def _check_study_access(study: dict[str, Any], write: bool = False) -> Optional[tuple[Any, int]]:
    user = g.current_user
    if user["role"] == "ADMIN":
        return None
    if not write and int(study.get("is_public") or 0) == 1:
        return None
    if not write and is_study_shared_with_user(study_id=study["id"], user_id=user["id"]):
        return None
    if study["owner_user_id"] != user["id"]:
        return jsonify({"status": "error", "message": "Study does not belong to current user"}), 403
    if write and user["role"] not in {"DOCTOR", "ADMIN"}:
        return jsonify({"status": "error", "message": "Write operation requires DOCTOR role"}), 403
    return None


def _log_audit(
    action: str,
    outcome: str = "SUCCESS",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    """Write an audit log entry. Silently swallows errors so auditing never breaks the API."""
    try:
        user = g.get("current_user")
        create_audit_log_entry(
            actor_user_id=user["id"] if user else None,
            actor_email=user["email"] if user else None,
            actor_role=user["role"] if user else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=request.remote_addr,
            user_agent=(request.headers.get("User-Agent") or "")[:255],
            outcome=outcome,
            detail=json.dumps(detail) if detail else None,
        )
    except Exception:
        logger.exception("Failed to write audit log for action=%s", action)
