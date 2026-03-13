from __future__ import annotations

import base64
import hashlib
import logging
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
from flask import Flask, g, jsonify, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SOURCE_DIR.parent
TRAINING = PROJECT_ROOT / "training"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(TRAINING))

from dicom_utils import DicomValidationError, get_dicom_metadata, validate_dicom_file
from image_utils import normalize_image, psnr
from watermarking import (
    FragileConfig,
    bits_to_payload,
    detect_tamper_map,
    embed_fragile_watermark,
    embed_robust_watermark,
    extract_robust_watermark,
    payload_to_bits,
    tamper_stats,
)

try:
    from .database_api import (
        close_db,
        create_study,
        db_api,
        get_study,
        init_db,
        list_studies_for_user,
        update_study_fields,
    )
    from .rbac_api import rbac_api, require_auth, require_roles
except ImportError:
    from database_api import (  # type: ignore
        close_db,
        create_study,
        db_api,
        get_study,
        init_db,
        list_studies_for_user,
        update_study_fields,
    )
    from rbac_api import rbac_api, require_auth, require_roles  # type: ignore

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
    if study["owner_user_id"] != user["id"]:
        return jsonify({"status": "error", "message": "Study does not belong to current user"}), 403
    if write and user["role"] not in {"DOCTOR", "ADMIN"}:
        return jsonify({"status": "error", "message": "Write operation requires DOCTOR role"}), 403
    return None


def create_app() -> Flask:
    app = Flask(__name__)

    storage_root = CURRENT_DIR / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    app.config["SECRET_KEY"] = os.environ.get("SECUREDICOM_SECRET", "securedicom-dev-secret")
    app.config["TOKEN_MAX_AGE_SECONDS"] = int(os.environ.get("SECUREDICOM_TOKEN_TTL", "86400"))
    app.config["DATABASE_PATH"] = str(storage_root / "securedicom.db")
    app.config["STORAGE_ROOT"] = str(storage_root)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    init_db(Path(app.config["DATABASE_PATH"]))

    app.register_blueprint(rbac_api)
    app.register_blueprint(db_api)

    @app.teardown_appcontext
    def _teardown(_: Any) -> None:
        close_db()

    @app.after_request
    def _cors(resp: Any) -> Any:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return resp

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok", "service": "securedicom-backend"}), 200

    @app.route("/wm/roi", methods=["POST"])
    @require_roles("DOCTOR", "ADMIN")
    def roi_process() -> tuple[Any, int]:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "file is required"}), 400

        dicom_file = request.files["file"]
        if not dicom_file.filename or not _is_allowed_file(dicom_file.filename):
            return jsonify({"status": "error", "message": "Only .dcm/.dicom files are allowed"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            dicom_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            validate_dicom_file(tmp_path)
            _load_model_if_needed()

            owner_id = g.current_user["id"]
            study = create_study(owner_user_id=owner_id, original_filename=secure_filename(dicom_file.filename), original_dcm_path="PENDING")
            study_path = _study_dir(Path(app.config["STORAGE_ROOT"]), study["id"])

            original_dcm = study_path / "original.dcm"
            shutil.move(tmp_path, str(original_dcm))

            roi_mask, overlay_img, _, roi_model_used = _extract_roi_mask_and_overlay(str(original_dcm))

            roi_mask_path = study_path / "roi_mask.npy"
            overlay_path = study_path / "roi_overlay.png"

            np.save(roi_mask_path, roi_mask)
            overlay_img.save(overlay_path, format="PNG")

            update_study_fields(
                study_id=study["id"],
                fields={
                    "original_dcm_path": str(original_dcm),
                    "roi_mask_path": str(roi_mask_path),
                    "roi_overlay_png_path": str(overlay_path),
                },
            )

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study["id"],
                        "roi_model_used": roi_model_used,
                        "roi_overlay_image": _png_to_base64(overlay_img),
                        "dicom_metadata": get_dicom_metadata(str(original_dcm)),
                    }
                ),
                200,
            )
        except DicomValidationError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            logger.exception("ROI process failed")
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.route("/wm/embed", methods=["POST"])
    @require_roles("DOCTOR", "ADMIN")
    def embed_watermarks() -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        study_id = (body.get("study_id") or "").strip()
        strength = float(body.get("strength", 1.0))
        visibility = str(body.get("visibility") or "PRIVATE").strip().upper()

        if not study_id:
            return jsonify({"status": "error", "message": "study_id is required"}), 400
        if not 0.5 <= strength <= 2.0:
            return jsonify({"status": "error", "message": "strength must be in [0.5, 2.0]"}), 400
        if visibility not in {"PRIVATE", "PUBLIC"}:
            return jsonify({"status": "error", "message": "visibility must be PRIVATE or PUBLIC"}), 400

        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=True)
        if denied:
            return denied

        if not study.get("original_dcm_path") or not study.get("roi_mask_path"):
            return jsonify({"status": "error", "message": "ROI must be generated first"}), 400

        try:
            dcm = pydicom.dcmread(study["original_dcm_path"])
            pixels = dcm.pixel_array
            image_2d, frame_idx, mode = _select_working_image(pixels)

            roi_mask = np.load(study["roi_mask_path"]).astype(bool)

            include_patient_id = bool(body.get("include_patient_id", True))
            include_doctor_id = bool(body.get("include_doctor_id", False))
            include_timestamp = bool(body.get("include_timestamp", False))
            include_hospital_name = bool(body.get("include_hospital_name", False))

            payload_parts: list[str] = []

            if include_patient_id:
                patient_id = body.get("patient_id") or getattr(dcm, "PatientID", "UNKNOWN")
                payload_parts.append(f"PATIENT_ID={patient_id}")

            patient_name = body.get("patient_name")
            if patient_name:
                payload_parts.append(f"PATIENT_NAME={patient_name}")

            if include_doctor_id:
                doctor_id = body.get("doctor_id") or g.current_user.get("id")
                payload_parts.append(f"DOCTOR_ID={doctor_id}")

            if include_timestamp:
                payload_parts.append(f"TIMESTAMP={datetime.now(UTC).isoformat()}")

            if include_hospital_name:
                hospital_name = body.get("hospital_name") or getattr(dcm, "InstitutionName", "UNKNOWN")
                payload_parts.append(f"HOSPITAL_NAME={hospital_name}")

            custom_metadata = body.get("custom_metadata")
            if custom_metadata:
                payload_parts.append(f"CUSTOM_METADATA={custom_metadata}")

            custom_watermark_text = body.get("custom_watermark_text")
            if custom_watermark_text:
                payload_parts.append(f"CUSTOM_WM_TEXT={custom_watermark_text}")

            if not payload_parts:
                payload_parts.append(f"PATIENT_ID={getattr(dcm, 'PatientID', 'UNKNOWN')}")

            payload_text = ";".join(payload_parts)
            payload_hex_full = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()

            non_roi_patches = _count_non_roi_patches(roi_mask)
            max_payload_bits = max(0, non_roi_patches // 3)
            payload_bit_length = min(256, (max_payload_bits // 4) * 4)

            if payload_bit_length < 8:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": (
                                "Not enough non-ROI capacity for robust watermark "
                                f"(available_patches={non_roi_patches}, required_min_patches=24)"
                            ),
                        }
                    ),
                    400,
                )

            payload_hex = payload_hex_full[: payload_bit_length // 4]

            robust_image = embed_robust_watermark(
                image=image_2d,
                roi_mask=roi_mask,
                payload_bits=payload_to_bits(payload_hex),
                strength=strength,
            )

            fragile_image = embed_fragile_watermark(
                image=robust_image.astype(image_2d.dtype),
                roi_mask=roi_mask,
                config=FragileConfig(),
            )

            watermarked_pixels = _write_watermarked_back(
                original_pixels=pixels,
                watermarked_2d=fragile_image,
                frame_idx=frame_idx,
                mode=mode,
            )
            dcm.PixelData = watermarked_pixels.tobytes()

            study_path = _study_dir(Path(app.config["STORAGE_ROOT"]), study_id)
            wm_dcm_path = study_path / "watermarked.dcm"
            wm_png_path = study_path / "watermarked.png"

            dcm.save_as(str(wm_dcm_path))

            wm_preview = Image.fromarray((normalize_image(fragile_image) * 255.0).astype(np.uint8), mode="L")
            wm_preview.save(wm_png_path, format="PNG")

            update_study_fields(
                study_id=study_id,
                fields={
                    "is_public": 1 if visibility == "PUBLIC" else 0,
                    "watermarked_dcm_path": str(wm_dcm_path),
                    "watermarked_png_path": str(wm_png_path),
                    "robust_payload_text": payload_text,
                    "robust_payload_hex": payload_hex,
                    "robust_verified": 0,
                },
            )

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study_id,
                        "payload_text": payload_text,
                        "payload_hex": payload_hex,
                        "payload_bits": payload_bit_length,
                        "available_non_roi_patches": non_roi_patches,
                        "visibility": visibility,
                        "psnr_db": round(psnr(normalize_image(image_2d), normalize_image(fragile_image)), 3),
                        "watermarked_png": _png_to_base64(wm_preview),
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.exception("Embed failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/wm/verify-robust", methods=["POST"])
    @require_auth
    def verify_robust() -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        study_id = (body.get("study_id") or "").strip()
        if not study_id:
            return jsonify({"status": "error", "message": "study_id is required"}), 400

        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=False)
        if denied:
            return denied

        if not study.get("watermarked_dcm_path") or not Path(study["watermarked_dcm_path"]).exists():
            return jsonify({"status": "error", "message": "No watermarked DICOM found for study"}), 400

        try:
            dcm = pydicom.dcmread(study["watermarked_dcm_path"])
            image_2d, _, _ = _select_working_image(dcm.pixel_array)
            roi_mask = np.load(study["roi_mask_path"]).astype(bool)

            expected_hex = study.get("robust_payload_hex")
            payload_length = len(expected_hex) * 4 if expected_hex else 256

            extracted_bits, confidence = extract_robust_watermark(
                image=image_2d,
                roi_mask=roi_mask,
                payload_length=payload_length,
            )

            extracted_hex = bits_to_payload(extracted_bits[:payload_length], bit_length=payload_length)
            verified = bool(expected_hex and extracted_hex == expected_hex)

            update_study_fields(study_id, {"robust_verified": int(verified)})

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study_id,
                        "extracted_payload_hex": extracted_hex,
                        "payload_text": study.get("robust_payload_text"),
                        "verified": verified,
                        "confidence": round(float(confidence), 4),
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.exception("Verify failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/wm/simulate-attack", methods=["POST"])
    @require_roles("DOCTOR", "ADMIN")
    def simulate_attack() -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        study_id = (body.get("study_id") or "").strip()
        attack_type = (body.get("attack_type") or "noise").strip().lower()

        if not study_id:
            return jsonify({"status": "error", "message": "study_id is required"}), 400

        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=True)
        if denied:
            return denied

        if not study.get("watermarked_dcm_path"):
            return jsonify({"status": "error", "message": "Embed watermark first"}), 400

        try:
            dcm = pydicom.dcmread(study["watermarked_dcm_path"])
            pixels = dcm.pixel_array
            image_2d, _, _ = _select_working_image(pixels)
            roi_mask = np.load(study["roi_mask_path"]).astype(bool)

            attacked = image_2d.copy()
            non_roi_positions = np.argwhere(~roi_mask)
            if non_roi_positions.size == 0:
                return jsonify({"status": "error", "message": "No non-ROI region available for attack"}), 400

            row, col = non_roi_positions[len(non_roi_positions) // 2]
            r0, r1 = max(0, row - 8), min(attacked.shape[0], row + 8)
            c0, c1 = max(0, col - 8), min(attacked.shape[1], col + 8)

            if attack_type == "zero":
                attacked[r0:r1, c0:c1] = 0
            else:
                noise = np.random.normal(0, 35, size=(r1 - r0, c1 - c0))
                attacked[r0:r1, c0:c1] = attacked[r0:r1, c0:c1].astype(np.float32) + noise
                attacked = np.clip(attacked, image_2d.min(), image_2d.max()).astype(image_2d.dtype)

            tamper_map = detect_tamper_map(image=attacked, roi_mask=roi_mask, config=FragileConfig())
            stats = tamper_stats(tamper_map, block_size=FragileConfig().block_size)

            tamper_img = (tamper_map.astype(np.uint8) * 255)
            attacked_b64 = _array_to_base64_png(attacked)
            tamper_b64 = _png_to_base64(Image.fromarray(tamper_img, mode="L"))

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study_id,
                        "attack_type": attack_type,
                        "attacked_png": attacked_b64,
                        "tamper_map_png": tamper_b64,
                        "tamper_stats": stats,
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.exception("Attack simulation failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/wm/download/<study_id>", methods=["GET"])
    @require_auth
    def download_watermarked_dcm(study_id: str) -> Any:
        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=False)
        if denied:
            return denied

        dcm_path = study.get("watermarked_dcm_path")
        if not dcm_path or not Path(dcm_path).exists():
            return jsonify({"status": "error", "message": "No watermarked DICOM available"}), 404

        filename = Path(dcm_path).name
        return send_file(dcm_path, as_attachment=True, download_name=filename, mimetype="application/dicom")

    @app.route("/wm/studies", methods=["GET"])
    @require_auth
    def studies_list() -> tuple[Any, int]:
        user = g.current_user
        scope = (request.args.get("scope") or "mine").strip().lower()
        if scope not in {"mine", "public", "mixed", "all"}:
            return jsonify({"status": "error", "message": "invalid scope"}), 400
        if scope == "all" and user["role"] != "ADMIN":
            scope = "mixed"
        rows = list_studies_for_user(user_id=user["id"], role=user["role"], scope=scope)
        return jsonify({"status": "success", "studies": rows}), 200

    @app.route("/wm/studies/<study_id>", methods=["GET"])
    @require_auth
    def studies_get(study_id: str) -> tuple[Any, int]:
        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=False)
        if denied:
            return denied

        study_response = dict(study)
        original_path = study.get("original_dcm_path")
        watermarked_path = study.get("watermarked_dcm_path")

        if original_path and watermarked_path and Path(original_path).exists() and Path(watermarked_path).exists():
            try:
                original_dcm = pydicom.dcmread(original_path)
                watermarked_dcm = pydicom.dcmread(watermarked_path)
                original_2d, _, _ = _select_working_image(original_dcm.pixel_array)
                watermarked_2d, _, _ = _select_working_image(watermarked_dcm.pixel_array)
                study_response["psnr_db"] = round(psnr(normalize_image(original_2d), normalize_image(watermarked_2d)), 3)
            except Exception as exc:
                logger.warning("Failed to compute PSNR for study %s: %s", study_id, exc)

        return jsonify({"status": "success", "study": study_response}), 200

    @app.route("/wm/studies/<study_id>/preview", methods=["GET"])
    @require_auth
    def studies_preview(study_id: str) -> Any:
        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=False)
        if denied:
            return denied

        kind = (request.args.get("kind") or "watermarked").strip().lower()
        if kind == "roi":
            image_path = study.get("roi_overlay_png_path")
        else:
            image_path = study.get("watermarked_png_path")

        if not image_path or not Path(image_path).exists():
            return jsonify({"status": "error", "message": "preview not available"}), 404

        return send_file(image_path, mimetype="image/png")

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5001, debug=True)
