from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, g, jsonify, request, send_file
from PIL import Image

from image_utils import normalize_image, psnr

try:
    from ..database_api import (
        get_study,
        get_user_by_email,
        list_studies_for_user,
        share_study_with_user,
    )
    from ..rbac_api import require_auth, require_roles
    from ..storage_layer import secure_read_bytes, secure_read_dicom
    from ..wm_common import (
        _check_study_access,
        _log_audit,
        _select_working_image,
    )
except ImportError:
    from database_api import get_study, get_user_by_email, list_studies_for_user, share_study_with_user  # type: ignore
    from rbac_api import require_auth, require_roles  # type: ignore
    from storage_layer import secure_read_bytes, secure_read_dicom  # type: ignore
    from wm_common import _check_study_access, _log_audit, _select_working_image  # type: ignore

logger = logging.getLogger(__name__)


def _png_bytes_from_dicom(path: str, secret: str) -> bytes:
    dcm = secure_read_dicom(path, secret)
    image_2d, _, _ = _select_working_image(dcm.pixel_array)
    image_uint8 = (normalize_image(image_2d) * 255.0).astype("uint8")
    img = Image.fromarray(image_uint8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def register_study_routes(app: Flask) -> None:
    @app.route("/wm/studies", methods=["GET"])
    @require_auth
    def studies_list() -> tuple[Any, int]:
        user = g.current_user
        scope = (request.args.get("scope") or "mine").strip().lower()
        if scope not in {"mine", "public", "shared", "mixed", "all"}:
            return jsonify({"status": "error", "message": "invalid scope"}), 400
        if scope == "all" and user["role"] != "ADMIN":
            scope = "mixed"
        rows = list_studies_for_user(user_id=user["id"], role=user["role"], scope=scope)
        return jsonify({"status": "success", "studies": rows}), 200

    @app.route("/wm/studies/<study_id>/share", methods=["POST"])
    @require_roles("DOCTOR", "ADMIN")
    def share_study(study_id: str) -> tuple[Any, int]:
        from flask import g

        body = request.get_json(silent=True) or {}
        recipient_email = (body.get("recipient_email") or "").strip().lower()
        if "@" not in recipient_email:
            return jsonify({"status": "error", "message": "recipient_email must be a valid email"}), 400

        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404

        denied = _check_study_access(study, write=True)
        if denied:
            return denied

        recipient = get_user_by_email(recipient_email)
        if not recipient:
            return jsonify({"status": "error", "message": "Recipient user not found"}), 404
        if recipient["id"] == g.current_user["id"]:
            return jsonify({"status": "error", "message": "Cannot share a study with yourself"}), 400

        created = share_study_with_user(
            study_id=study_id,
            recipient_user_id=recipient["id"],
            shared_by_user_id=g.current_user["id"],
        )

        _log_audit(
            "STUDY_SHARED", "SUCCESS", resource_type="study", resource_id=study_id,
            detail={"recipient_email": recipient_email, "already_shared": not created},
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "study_id": study_id,
                    "recipient_email": recipient["email"],
                    "shared": True,
                    "already_shared": not created,
                }
            ),
            200,
        )

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
                secret = str(app.config["SECRET_KEY"])
                original_dcm = secure_read_dicom(original_path, secret)
                watermarked_dcm = secure_read_dicom(watermarked_path, secret)
                original_2d, _, _ = _select_working_image(original_dcm.pixel_array)
                watermarked_2d, _, _ = _select_working_image(watermarked_dcm.pixel_array)
                study_response["psnr_db"] = round(psnr(normalize_image(original_2d), normalize_image(watermarked_2d)), 3)
            except Exception as exc:
                logger.warning("Failed to compute PSNR for study %s: %s", study_id, exc)

        _log_audit("STUDY_VIEWED", "SUCCESS", resource_type="study", resource_id=study_id)
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
        secret = str(app.config["SECRET_KEY"])

        if kind == "roi":
            image_path = study.get("roi_overlay_png_path")
            if not image_path or not Path(image_path).exists():
                return jsonify({"status": "error", "message": "preview not available"}), 404
            if str(image_path).endswith(".sdc"):
                png_raw = secure_read_bytes(image_path, secret)
                return send_file(BytesIO(png_raw), mimetype="image/png")
            return send_file(image_path, mimetype="image/png")

        if kind == "original":
            original_path = study.get("original_dcm_path")
            if not original_path or not Path(original_path).exists():
                return jsonify({"status": "error", "message": "original image not available"}), 404
            return send_file(BytesIO(_png_bytes_from_dicom(original_path, secret)), mimetype="image/png")

        image_path = study.get("watermarked_png_path")
        if not image_path or not Path(image_path).exists():
            return jsonify({"status": "error", "message": "preview not available"}), 404
        if str(image_path).endswith(".sdc"):
            png_raw = secure_read_bytes(image_path, secret)
            return send_file(BytesIO(png_raw), mimetype="image/png")
        return send_file(image_path, mimetype="image/png")

    @app.route("/wm/public/studies", methods=["GET"])
    def public_studies_list() -> tuple[Any, int]:
        rows = list_studies_for_user(user_id="", role="GUEST", scope="public")
        return jsonify({"status": "success", "studies": rows}), 200

    @app.route("/wm/public/studies/<study_id>/preview", methods=["GET"])
    def public_studies_preview(study_id: str) -> Any:
        study = get_study(study_id)
        if not study:
            return jsonify({"status": "error", "message": "study not found"}), 404
        if int(study.get("is_public") or 0) != 1:
            return jsonify({"status": "error", "message": "study is not public"}), 403

        kind = (request.args.get("kind") or "watermarked").strip().lower()
        secret = str(app.config["SECRET_KEY"])

        if kind == "roi":
            image_path = study.get("roi_overlay_png_path")
            if not image_path or not Path(image_path).exists():
                return jsonify({"status": "error", "message": "preview not available"}), 404
            if str(image_path).endswith(".sdc"):
                png_raw = secure_read_bytes(image_path, secret)
                return send_file(BytesIO(png_raw), mimetype="image/png")
            return send_file(image_path, mimetype="image/png")

        if kind == "original":
            original_path = study.get("original_dcm_path")
            if not original_path or not Path(original_path).exists():
                return jsonify({"status": "error", "message": "original image not available"}), 404
            return send_file(BytesIO(_png_bytes_from_dicom(original_path, secret)), mimetype="image/png")

        image_path = study.get("watermarked_png_path")
        if not image_path or not Path(image_path).exists():
            return jsonify({"status": "error", "message": "preview not available"}), 404
        if str(image_path).endswith(".sdc"):
            png_raw = secure_read_bytes(image_path, secret)
            return send_file(BytesIO(png_raw), mimetype="image/png")
        return send_file(image_path, mimetype="image/png")
