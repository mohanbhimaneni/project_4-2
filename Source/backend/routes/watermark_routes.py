from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, g, jsonify, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

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
    from ..database_api import create_study, get_study, update_study_fields
    from ..rbac_api import require_auth, require_roles
    from ..storage_layer import (
        secure_read_bytes,
        secure_read_dicom,
        secure_read_numpy,
        secure_write_bytes,
        secure_write_numpy,
    )
    from ..wm_common import (
        _array_to_base64_png,
        _check_study_access,
        _count_non_roi_patches,
        _decode_data_url_png,
        _extract_roi_mask_and_overlay,
        _is_allowed_file,
        _load_model_if_needed,
        _log_audit,
        _mask_to_base64_png,
        _overlay_from_image_and_mask,
        _png_to_base64,
        _select_working_image,
        _study_dir,
        _write_watermarked_back,
    )
except ImportError:
    from database_api import create_study, get_study, update_study_fields  # type: ignore
    from rbac_api import require_auth, require_roles  # type: ignore
    from storage_layer import (  # type: ignore
        secure_read_bytes,
        secure_read_dicom,
        secure_read_numpy,
        secure_write_bytes,
        secure_write_numpy,
    )
    from wm_common import (  # type: ignore
        _array_to_base64_png,
        _check_study_access,
        _count_non_roi_patches,
        _decode_data_url_png,
        _extract_roi_mask_and_overlay,
        _is_allowed_file,
        _load_model_if_needed,
        _log_audit,
        _mask_to_base64_png,
        _overlay_from_image_and_mask,
        _png_to_base64,
        _select_working_image,
        _study_dir,
        _write_watermarked_back,
    )

logger = logging.getLogger(__name__)


def register_watermark_routes(app: Flask) -> None:
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
            study = create_study(
                owner_user_id=owner_id,
                original_filename=secure_filename(dicom_file.filename),
                original_dcm_path="PENDING",
            )
            study_path = _study_dir(Path(app.config["STORAGE_ROOT"]), study["id"])
            secret = str(app.config["SECRET_KEY"])

            original_dcm = study_path / "original.dcm.sdc"
            raw_dcm = Path(tmp_path).read_bytes()
            original_meta = secure_write_bytes(original_dcm, raw_dcm, secret)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as dec_tmp:
                dec_tmp.write(raw_dcm)
                dec_tmp_path = dec_tmp.name

            try:
                roi_mask, overlay_img, image_2d, roi_model_used = _extract_roi_mask_and_overlay(dec_tmp_path)
                dicom_metadata = get_dicom_metadata(dec_tmp_path)
            finally:
                if os.path.exists(dec_tmp_path):
                    os.remove(dec_tmp_path)

            roi_mask_path = study_path / "roi_mask.npy.sdc"
            overlay_path = study_path / "roi_overlay.png"

            roi_mask_meta = secure_write_numpy(roi_mask_path, roi_mask, secret)
            overlay_img.save(overlay_path, format="PNG")
            overlay_meta = {
                "algorithm": "plain-png",
                "stored_bytes": overlay_path.stat().st_size,
                "encrypted": False,
            }

            update_study_fields(
                study_id=study["id"],
                fields={
                    "original_dcm_path": str(original_dcm),
                    "roi_mask_path": str(roi_mask_path),
                    "roi_overlay_png_path": str(overlay_path),
                },
            )

            _log_audit(
                "DICOM_UPLOAD",
                "SUCCESS",
                resource_type="study",
                resource_id=study["id"],
                detail={
                    "filename": secure_filename(dicom_file.filename),
                    "study_id": study["id"],
                    "roi_model_used": roi_model_used,
                    "storage": {
                        "original": original_meta,
                        "roi_mask": roi_mask_meta,
                        "roi_overlay": overlay_meta,
                    },
                },
            )

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study["id"],
                        "roi_model_used": roi_model_used,
                        "roi_overlay_image": _png_to_base64(overlay_img),
                        "original_image": _array_to_base64_png(image_2d),
                        "roi_mask_image": _mask_to_base64_png(roi_mask),
                        "dicom_metadata": dicom_metadata,
                    }
                ),
                200,
            )
        except DicomValidationError as exc:
            _log_audit(
                "DICOM_UPLOAD",
                "FAILURE",
                detail={"reason": str(exc), "filename": secure_filename(dicom_file.filename)},
            )
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            logger.exception("ROI process failed")
            _log_audit("DICOM_UPLOAD", "FAILURE", detail={"reason": str(exc)})
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
            secret = str(app.config["SECRET_KEY"])
            dcm = secure_read_dicom(study["original_dcm_path"], secret)
            pixels = dcm.pixel_array
            image_2d, frame_idx, mode = _select_working_image(pixels)

            roi_mask = secure_read_numpy(study["roi_mask_path"], secret).astype(bool)

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
            wm_dcm_path = study_path / "watermarked.dcm.sdc"
            wm_png_path = study_path / "watermarked.png"

            dcm_buf = BytesIO()
            dcm.save_as(dcm_buf)
            wm_dcm_meta = secure_write_bytes(wm_dcm_path, dcm_buf.getvalue(), secret)

            wm_preview = Image.fromarray((normalize_image(fragile_image) * 255.0).astype(np.uint8), mode="L")
            wm_preview.save(wm_png_path, format="PNG")
            wm_png_meta = {
                "algorithm": "plain-png",
                "stored_bytes": wm_png_path.stat().st_size,
                "encrypted": False,
            }

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

            psnr_val = round(psnr(normalize_image(image_2d), normalize_image(fragile_image)), 3)

            _log_audit(
                "WATERMARK_EMBEDDED",
                "SUCCESS",
                resource_type="study",
                resource_id=study_id,
                detail={
                    "study_id": study_id,
                    "payload_text": payload_text,
                    "payload_hex": payload_hex,
                    "payload_bits": payload_bit_length,
                    "visibility": visibility,
                    "psnr_db": psnr_val,
                    "storage": {"watermarked_dcm": wm_dcm_meta, "watermarked_png": wm_png_meta},
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
                        "psnr_db": psnr_val,
                        "watermarked_png": _png_to_base64(wm_preview),
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.exception("Embed failed")
            _log_audit(
                "WATERMARK_EMBEDDED",
                "FAILURE",
                resource_type="study",
                resource_id=study_id,
                detail={"reason": str(exc)},
            )
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/wm/studies/<study_id>/roi-mask", methods=["PUT"])
    @require_roles("DOCTOR", "ADMIN")
    def update_roi_mask(study_id: str) -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        mask_png = (body.get("mask_png") or "").strip()
        if not mask_png:
            return jsonify({"status": "error", "message": "mask_png is required"}), 400

        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=True)
        if denied:
            return denied

        if not study.get("original_dcm_path") or not Path(study["original_dcm_path"]).exists():
            return jsonify({"status": "error", "message": "Original DICOM not available"}), 400

        if not study.get("roi_mask_path"):
            return jsonify({"status": "error", "message": "ROI must be generated first"}), 400

        try:
            secret = str(app.config["SECRET_KEY"])
            dcm = secure_read_dicom(study["original_dcm_path"], secret)
            image_2d, _, _ = _select_working_image(dcm.pixel_array)

            mask_img = _decode_data_url_png(mask_png)
            expected_h, expected_w = image_2d.shape
            if mask_img.size != (expected_w, expected_h):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Mask dimensions mismatch. Expected {expected_w}x{expected_h}",
                        }
                    ),
                    400,
                )

            roi_mask = (np.array(mask_img, dtype=np.uint8) >= 128).astype(np.uint8)
            if not np.any(roi_mask):
                return jsonify({"status": "error", "message": "ROI mask cannot be empty"}), 400

            roi_mask_path = Path(study["roi_mask_path"])
            study_path = _study_dir(Path(app.config["STORAGE_ROOT"]), study_id)
            overlay_path = study_path / "roi_overlay.png"

            roi_mask_meta = secure_write_numpy(roi_mask_path, roi_mask, secret)

            overlay_img = _overlay_from_image_and_mask(image_2d=image_2d, roi_mask=roi_mask)
            overlay_img.save(overlay_path, format="PNG")
            overlay_meta = {
                "algorithm": "plain-png",
                "stored_bytes": overlay_path.stat().st_size,
                "encrypted": False,
            }

            update_study_fields(
                study_id=study_id,
                fields={
                    "roi_mask_path": str(roi_mask_path),
                    "roi_overlay_png_path": str(overlay_path),
                },
            )

            _log_audit(
                "ROI_MASK_UPDATED",
                "SUCCESS",
                resource_type="study",
                resource_id=study_id,
                detail={"storage": {"roi_mask": roi_mask_meta, "roi_overlay": overlay_meta}},
            )

            return (
                jsonify(
                    {
                        "status": "success",
                        "study_id": study_id,
                        "roi_overlay_image": _png_to_base64(overlay_img),
                        "roi_mask_image": _mask_to_base64_png(roi_mask),
                    }
                ),
                200,
            )
        except Exception as exc:
            logger.exception("ROI mask update failed")
            _log_audit(
                "ROI_MASK_UPDATED",
                "FAILURE",
                resource_type="study",
                resource_id=study_id,
                detail={"reason": str(exc)},
            )
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
            secret = str(app.config["SECRET_KEY"])
            dcm = secure_read_dicom(study["watermarked_dcm_path"], secret)
            image_2d, _, _ = _select_working_image(dcm.pixel_array)
            roi_mask = secure_read_numpy(study["roi_mask_path"], secret).astype(bool)

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

            _log_audit(
                "WATERMARK_VERIFIED",
                "SUCCESS",
                resource_type="study",
                resource_id=study_id,
                detail={"verified": verified, "confidence": round(float(confidence), 4)},
            )

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
            _log_audit(
                "WATERMARK_VERIFIED",
                "FAILURE",
                resource_type="study",
                resource_id=study_id,
                detail={"reason": str(exc)},
            )
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
            secret = str(app.config["SECRET_KEY"])
            dcm = secure_read_dicom(study["watermarked_dcm_path"], secret)
            pixels = dcm.pixel_array
            image_2d, _, _ = _select_working_image(pixels)
            roi_mask = secure_read_numpy(study["roi_mask_path"], secret).astype(bool)

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

            _log_audit(
                "ATTACK_SIMULATED",
                "SUCCESS",
                resource_type="study",
                resource_id=study_id,
                detail={"attack_type": attack_type, "tamper_stats": stats},
            )

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
            _log_audit(
                "ATTACK_SIMULATED",
                "FAILURE",
                resource_type="study",
                resource_id=study_id,
                detail={"reason": str(exc)},
            )
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

        secret = str(app.config["SECRET_KEY"])
        dcm_raw = secure_read_bytes(dcm_path, secret)

        _log_audit("STUDY_DOWNLOADED", "SUCCESS", resource_type="study", resource_id=study_id)
        filename = "watermarked.dcm"
        return send_file(
            BytesIO(dcm_raw),
            as_attachment=True,
            download_name=filename,
            mimetype="application/dicom",
        )
