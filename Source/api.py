"""REST API for DICOM ROI extraction and watermarking using fine-tuned ViT model.

Endpoints:
  POST /roi/process - Upload a DICOM file and receive ROI-overlaid PNG image
  POST /embed - Embed watermark in DICOM file
  POST /verify - Extract and verify watermark from image
  GET /health - Health check endpoint
  GET / - API documentation
"""

import os
import sys
import tempfile
import logging
import time
from pathlib import Path
from io import BytesIO
import traceback
from datetime import datetime

import numpy as np
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

# Add project_understanding to Python path to import roi.py
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "project_understanding"))

# Import local utilities
from dicom_utils import validate_dicom_file, get_dicom_metadata, DicomValidationError
from image_utils import normalize_image, psnr, get_roi_statistics, get_roi_coverage, array_to_pil_image
from watermarking import (
    embed_robust_watermark,
    extract_robust_watermark,
    payload_to_bits,
    bits_to_payload,
    WatermarkException,
)
from watermarking.watermark_patterns import create_ownership_payload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# === Configuration ===
ALLOWED_EXTENSIONS = {'dcm', 'dicom'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CHECKPOINT_PATH = project_root / "project_understanding" / "checkpoints" / "midi_b_modality_vit_tiny_next" / "best.pt"
MODEL_NAME = "vit_tiny_patch16_224"
DEVICE = "cpu"  # Use "cpu" for CPU, "cuda" for GPU
ALPHA = 0.5  # Overlay transparency
MAX_PROCESSING_TIME = 30  # seconds

# === Flask App Setup ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model cache (loaded once on first request)
_cached_model = None
_cached_device = None
_model_load_time = None


def _select_working_image(pixels: np.ndarray) -> tuple[np.ndarray, int | None, str]:
    """Select a 2D working image from DICOM pixel array.

    Returns:
        (image_2d, frame_index, mode)
    """
    arr = np.asarray(pixels)

    if arr.ndim == 2:
        return arr, None, "2d"

    if arr.ndim == 3:
        # 2D color image
        if arr.shape[-1] in (3, 4):
            gray = np.mean(arr[..., :3], axis=-1).astype(arr.dtype)
            return gray, None, "color2d"

        # Multi-frame grayscale volume: (frames, H, W)
        frame_idx = arr.shape[0] // 2
        return arr[frame_idx], frame_idx, "volume3d"

    if arr.ndim == 4:
        # Multi-frame color volume: (frames, H, W, C)
        frame_idx = arr.shape[0] // 2
        frame = arr[frame_idx]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            gray = np.mean(frame[..., :3], axis=-1).astype(frame.dtype)
            return gray, frame_idx, "volume4d_color"
        return frame, frame_idx, "volume4d"

    raise ValueError(f"Unsupported pixel array dimensions: {arr.shape}")


def _write_watermarked_back(original_pixels: np.ndarray, watermarked_2d: np.ndarray,
                            frame_idx: int | None, mode: str) -> np.ndarray:
    """Write watermarked 2D result back into original DICOM pixel structure."""
    arr = np.array(original_pixels, copy=True)
    wm = watermarked_2d.astype(arr.dtype, copy=False)

    if mode == "2d":
        return wm

    if mode == "volume3d" and frame_idx is not None:
        arr[frame_idx] = wm
        return arr

    raise ValueError(
        f"DICOM return_format='dicom' not supported for pixel mode '{mode}'. Use return_format='json'."
    )


def _load_model_if_needed():
    """Load and cache the ViT model on first API request."""
    global _cached_model, _cached_device, _model_load_time
    
    if _cached_model is None:
        logger.info(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
        try:
            import sys
            sys.path.insert(0, str(project_root / "project_understanding"))
            from roi import _get_vit_model  # type: ignore
            
            # _get_vit_model returns a tuple (model, device)
            _cached_model, _cached_device = _get_vit_model(
                model_name=MODEL_NAME,
                checkpoint=str(CHECKPOINT_PATH),
                device=DEVICE
            )
            _model_load_time = datetime.now()
            logger.info(f"Model loaded successfully on device: {_cached_device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def _create_error_response(error_code: str, message: str, details: str = None, http_code: int = 400) -> tuple:
    """Create standardized error response."""
    response = {
        "status": "error",
        "error_code": error_code,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    if details:
        response["details"] = details
    return jsonify(response), http_code


def _create_success_response(data: dict, http_code: int = 200) -> tuple:
    """Create standardized success response."""
    data["status"] = "success"
    data["timestamp"] = datetime.now().isoformat()
    return jsonify(data), http_code


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status."""
    try:
        deep_check = request.args.get('deep_check', '0') == '1'

        if _cached_model is not None:
            model_status = "loaded"
        elif deep_check:
            try:
                _load_model_if_needed()
                model_status = "loaded"
            except Exception as e:
                model_status = f"error: {str(e)}"
        else:
            model_status = "not_loaded"
        
        return _create_success_response({
            "service": "DICOM ROI Extraction API",
            "version": "1.0",
            "model": MODEL_NAME,
            "device": _cached_device if _cached_device else DEVICE,
            "model_status": model_status,
            "deep_check": deep_check,
            "model_load_time": _model_load_time.isoformat() if _model_load_time else None,
            "checkpoint": str(CHECKPOINT_PATH),
        })[0]
    except Exception as e:
        return _create_error_response(
            "HEALTH_CHECK_FAILED",
            f"Health check failed: {str(e)}",
            http_code=500
        )[0]


@app.route('/roi/process', methods=['POST'])
def process_roi():
    """
    Process DICOM file and return ROI-overlaid PNG image with metadata.
    
    Expected input:
      - FILE: multipart/form-data with key 'file' containing DICOM file
      - Optional PARAMS:
        - alpha: overlay transparency (0.0-1.0, default 0.5)
        - device: cpu or cuda (default from config)
        - return_format: 'image' (PNG only) or 'json' (PNG + metadata), default 'json'
    
    Returns:
      - Success (200): PNG image or JSON with image + metadata
      - Error (400): JSON with error details
    """
    request_start_time = time.time()
    tmp_dicom_path = None
    
    try:
        # === FILE UPLOAD VALIDATION ===
        if 'file' not in request.files:
            return _create_error_response(
                "NO_FILE",
                "No file part in the request. Use form key 'file'.",
                http_code=400
            )
        
        file = request.files['file']
        
        if file.filename == '':
            return _create_error_response(
                "EMPTY_FILENAME",
                "No file selected for upload.",
                http_code=400
            )
        
        if not allowed_file(file.filename):
            return _create_error_response(
                "INVALID_FILE_TYPE",
                f"File type not allowed. Expected: {', '.join(ALLOWED_EXTENSIONS)}",
                details=f"Got: {Path(file.filename).suffix}",
                http_code=400
            )
        
        # === REQUEST PARAMETER VALIDATION ===
        alpha = request.form.get('alpha', ALPHA, type=float)
        if not (0.0 <= alpha <= 1.0):
            return _create_error_response(
                "INVALID_ALPHA",
                "Parameter 'alpha' must be between 0.0 and 1.0",
                http_code=400
            )
        
        device = request.form.get('device', DEVICE)
        if device not in ['cpu', 'cuda']:
            return _create_error_response(
                "INVALID_DEVICE",
                "Parameter 'device' must be 'cpu' or 'cuda'",
                http_code=400
            )
        
        return_format = request.form.get('return_format', 'json').lower()
        if return_format not in ['image', 'json']:
            return_format = 'json'
        
        # === SAVE AND VALIDATE DICOM FILE ===
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            file.save(tmp_file.name)
            tmp_dicom_path = tmp_file.name
        
        # Validate DICOM file
        try:
            validate_dicom_file(tmp_dicom_path)
        except DicomValidationError as e:
            return _create_error_response(
                "INVALID_DICOM",
                f"DICOM validation failed: {str(e)}",
                http_code=400
            )
        
        # Extract metadata
        try:
            dicom_metadata = get_dicom_metadata(tmp_dicom_path)
        except Exception as e:
            logger.warning(f"Could not extract full DICOM metadata: {str(e)}")
            dicom_metadata = {"error": str(e)}
        
        # === LOAD MODEL ===
        try:
            _load_model_if_needed()
        except Exception as e:
            return _create_error_response(
                "MODEL_LOAD_FAILED",
                f"Failed to load ROI extraction model: {str(e)}",
                http_code=500
            )
        
        # === ROI EXTRACTION ===
        logger.info(f"Processing DICOM: {file.filename}")
        try:
            from roi import extract_roi_from_dicom_file  # type: ignore
            
            # Call ROI extraction with cached model
            roi_image = extract_roi_from_dicom_file(
                dicom_path=tmp_dicom_path,
                model=_cached_model,
                model_name=MODEL_NAME,
                device=_cached_device,
                alpha=alpha
            )
            
            if roi_image is None:
                return _create_error_response(
                    "ROI_EXTRACTION_FAILED",
                    "Failed to extract ROI from DICOM file.",
                    details="Check logs for more information.",
                    http_code=400
                )
        
        except Exception as e:
            logger.error(f"ROI extraction error: {str(e)}")
            logger.error(traceback.format_exc())
            return _create_error_response(
                "ROI_EXTRACTION_ERROR",
                f"Error during ROI extraction: {str(e)}",
                http_code=500
            )
        
        # === CALCULATE QUALITY METRICS ===
        processing_time = time.time() - request_start_time
        
        roi_coverage = None
        try:
            # Rebuild ROI mask for real coverage metric
            from roi import _load_dicom_pixels, _normalize_image, _get_patch_embeddings, _cluster_roi  # type: ignore

            roi_image_array = _load_dicom_pixels(tmp_dicom_path)
            roi_norm = _normalize_image(roi_image_array)
            roi_patch_feats = _get_patch_embeddings(
                roi_norm,
                model_name=MODEL_NAME,
                checkpoint=str(CHECKPOINT_PATH),
                device=_cached_device,
            )
            roi_mask_raw = _cluster_roi(roi_patch_feats, roi_norm)
            if roi_mask_raw is not None:
                roi_coverage = round(get_roi_coverage((roi_mask_raw == 1).astype(np.uint8)), 3)
        except Exception as e:
            logger.warning(f"Could not calculate ROI coverage metric: {str(e)}")
        
        # === PREPARE RESPONSE ===
        logger.info(f"Successfully processed DICOM: {file.filename} ({processing_time:.2f}s)")
        
        if return_format == 'image':
            # Return PNG only
            img_bytes = BytesIO()
            roi_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return send_file(
                img_bytes,
                mimetype='image/png',
                as_attachment=True,
                download_name=f"{Path(file.filename).stem}_roi.png"
            )
        else:
            # Return JSON + base64 image
            img_bytes = BytesIO()
            roi_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            import base64
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            response_data = {
                "filename": file.filename,
                "image": f"data:image/png;base64,{img_base64}",
                "dimensions": {
                    "width": dicom_metadata.get("width"),
                    "height": dicom_metadata.get("height"),
                    "bit_depth": dicom_metadata.get("bit_depth"),
                },
                "dicom_metadata": dicom_metadata,
                "processing": {
                    "model": MODEL_NAME,
                    "device": _cached_device,
                    "alpha": alpha,
                    "processing_time_seconds": round(processing_time, 3),
                    "roi_coverage_percent": roi_coverage,
                },
            }
            
            return _create_success_response(response_data)[0]
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        return _create_error_response(
            "INTERNAL_ERROR",
            "Internal server error",
            details=str(e),
            http_code=500
        )
    
    finally:
        # Clean up temporary DICOM file
        if tmp_dicom_path and os.path.exists(tmp_dicom_path):
            try:
                os.remove(tmp_dicom_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {str(e)}")


@app.route('/embed', methods=['POST'])
def embed_watermark():
    """
    Embed watermark in DICOM file.
    
    Expected input:
      - FILE: DICOM file
      - payload: Hex string watermark payload (64 chars = 256 bits) or 'auto'
      - patient_id: Patient identifier (for auto payload generation)
      - strength: Watermark embedding strength (0.5-2.0, default 1.0)
      - return_format: 'json' or 'dicom'
    
    Returns JSON with watermark info or DICOM file.
    """
    request_start_time = time.time()
    tmp_dicom_path = None
    
    try:
        # File validation
        if 'file' not in request.files or request.files['file'].filename == '':
            return _create_error_response("NO_FILE", "No file provided", http_code=400)
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return _create_error_response("INVALID_FILE_TYPE", "Expected .dcm or .dicom file", http_code=400)
        
        # Parameter validation
        payload_param = request.form.get('payload', 'auto').lower()
        patient_id = request.form.get('patient_id', 'UNKNOWN')
        strength = request.form.get('strength', 1.0, type=float)
        return_format = request.form.get('return_format', 'json').lower()
        
        # Generate or validate payload
        if payload_param == 'auto':
            try:
                payload_hex = create_ownership_payload(patient_id)
            except Exception as e:
                return _create_error_response("PAYLOAD_GENERATION_FAILED", str(e), http_code=400)
        else:
            payload_hex = payload_param
            if len(payload_hex) != 64:
                return _create_error_response("INVALID_PAYLOAD", "Payload must be 64 hex chars", http_code=400)
        
        if not (0.5 <= strength <= 2.0):
            return _create_error_response("INVALID_STRENGTH", "Strength must be 0.5-2.0", http_code=400)
        
        # Save and validate DICOM
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            file.save(tmp_file.name)
            tmp_dicom_path = tmp_file.name
        
        try:
            validate_dicom_file(tmp_dicom_path)
        except DicomValidationError as e:
            return _create_error_response("INVALID_DICOM", str(e), http_code=400)
        
        dicom_metadata = get_dicom_metadata(tmp_dicom_path)
        
        # Load model
        try:
            _load_model_if_needed()
        except Exception as e:
            return _create_error_response("MODEL_LOAD_FAILED", str(e), http_code=500)
        
        # Extract ROI and image
        try:
            from roi import _load_dicom_pixels, _normalize_image, _get_patch_embeddings, _cluster_roi # type: ignore
            import pydicom
            
            dcm = pydicom.dcmread(tmp_dicom_path)
            pixel_array = dcm.pixel_array
            image_array, frame_idx, pixel_mode = _select_working_image(pixel_array)
            
            # Extract ROI mask using the same clustering approach as roi.py
            # This gives us the actual binary ROI mask (1 = ROI, 0 = background)
            img_norm = _normalize_image(image_array)
            patch_feats = _get_patch_embeddings(
                img_norm,
                model_name=MODEL_NAME,
                checkpoint=str(CHECKPOINT_PATH),
                device=_cached_device
            )
            roi_mask_raw = _cluster_roi(patch_feats, img_norm)
            
            if roi_mask_raw is None:
                return _create_error_response("ROI_EXTRACTION_FAILED", "Could not cluster ROI", http_code=400)
            
            # _cluster_roi returns binary ROI where 1 = ROI, 0 = non-ROI
            # Robust watermark expects True = ROI (protected), False = non-ROI (embeddable)
            roi_mask = (roi_mask_raw == 1).astype(bool)
            
        except Exception as e:
            logger.error(f"ROI extraction error: {str(e)}")
            return _create_error_response("ROI_EXTRACTION_ERROR", str(e), http_code=500)
        
        # Embed watermark
        try:
            payload_bits = payload_to_bits(payload_hex)
            watermarked = embed_robust_watermark(
                image=image_array,
                roi_mask=roi_mask,
                payload_bits=payload_bits,
                strength=strength
            )
            psnr_val = psnr(normalize_image(image_array), normalize_image(watermarked))
            
        except WatermarkException as e:
            return _create_error_response("WATERMARK_EMBEDDING_FAILED", str(e), http_code=400)
        except Exception as e:
            logger.error(f"Watermarking error: {str(e)}")
            return _create_error_response("WATERMARK_ERROR", str(e), http_code=500)
        
        processing_time = time.time() - request_start_time
        response_data = {
            "filename": file.filename,
            "watermark_payload": payload_hex,
            "embedding": {
                "strength": strength,
                "psnr_db": round(psnr_val, 2),
                "processing_time_seconds": round(processing_time, 3),
            },
            "dicom_metadata": dicom_metadata,
        }
        
        if return_format == 'dicom':
            try:
                watermarked_pixels = _write_watermarked_back(pixel_array, watermarked, frame_idx, pixel_mode)
            except ValueError as e:
                return _create_error_response("UNSUPPORTED_DICOM_PIXELS", str(e), http_code=400)

            dcm.PixelData = watermarked_pixels.tobytes()
            watermarked_dcm = BytesIO()
            dcm.save_as(watermarked_dcm)
            watermarked_dcm.seek(0)
            return send_file(watermarked_dcm, mimetype='application/dicom', as_attachment=True,
                           download_name=f"{Path(file.filename).stem}_watermarked.dcm")
        else:
            # Convert watermarked image to uint8 preview with robust normalization
            # (important for medical integer ranges like HU where clipping causes binary-looking previews)
            watermarked_norm = normalize_image(watermarked)
            watermarked_uint8 = np.clip(watermarked_norm * 255.0, 0, 255).astype(np.uint8)

            from PIL import Image
            watermarked_pil = Image.fromarray(watermarked_uint8, mode='L')
            img_bytes = BytesIO()
            watermarked_pil.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            import base64
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            response_data["watermarked_image_preview"] = f"data:image/png;base64,{img_base64}"
            
            return _create_success_response(response_data)[0]
    
    except Exception as e:
        logger.error(f"Unhandled exception in /embed: {str(e)}")
        logger.error(traceback.format_exc())
        return _create_error_response("INTERNAL_ERROR", str(e), http_code=500)
    
    finally:
        if tmp_dicom_path and os.path.exists(tmp_dicom_path):
            try:
                os.remove(tmp_dicom_path)
            except Exception:
                pass


@app.route('/verify', methods=['POST'])
def verify_watermark():
    """
    Extract and verify watermark from DICOM or image file.
    
    Expected input:
      - FILE: DICOM or image file
      - expected_payload: Optional hex payload to compare against
    
    Returns JSON with extracted payload and verification result.
    """
    request_start_time = time.time()
    tmp_dicom_path = None
    
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return _create_error_response("NO_FILE", "No file provided", http_code=400)
        
        file = request.files['file']
        expected_payload = request.form.get('expected_payload', '').lower()
        
        # Load model
        try:
            _load_model_if_needed()
        except Exception as e:
            return _create_error_response("MODEL_LOAD_FAILED", str(e), http_code=500)
        
        # Load image/DICOM
        try:
            import pydicom
            from PIL import Image
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                file.save(tmp_file.name)
                tmp_dicom_path = tmp_file.name
            
            try:
                dcm = pydicom.dcmread(tmp_dicom_path)
                pixel_array = dcm.pixel_array
                image_array, _, _ = _select_working_image(pixel_array)
                is_dicom = True
            except Exception:
                img = Image.open(tmp_dicom_path)
                image_array = np.array(img.convert('L'))
                is_dicom = False
            
            if is_dicom:
                # Rebuild ROI mask using the SAME clustering flow as /embed.
                # This is required so extraction reads the same non-ROI patch set used at embed time.
                from roi import _normalize_image, _get_patch_embeddings, _cluster_roi  # type: ignore

                img_norm = _normalize_image(image_array)
                patch_feats = _get_patch_embeddings(
                    img_norm,
                    model_name=MODEL_NAME,
                    checkpoint=str(CHECKPOINT_PATH),
                    device=_cached_device
                )
                roi_mask_raw = _cluster_roi(patch_feats, img_norm)

                if roi_mask_raw is None:
                    return _create_error_response("ROI_EXTRACTION_FAILED", "Could not cluster ROI", http_code=400)

                roi_mask = (roi_mask_raw == 1).astype(bool)
            else:
                # Non-DICOM fallback: no model-based ROI extraction available.
                # Use full-image extraction region for best-effort verification.
                roi_mask = np.zeros_like(image_array, dtype=bool)
                logger.warning("Verifying non-DICOM input without ROI clustering; using full-image extraction mask")
            
        except Exception as e:
            logger.error(f"Image loading error: {str(e)}")
            return _create_error_response("IMAGE_LOADING_FAILED", str(e), http_code=400)
        
        # Extract watermark
        try:
            extracted_bits, confidence = extract_robust_watermark(
                image=image_array,
                roi_mask=roi_mask,
                payload_length=256
            )
            
            if not extracted_bits:
                return _create_error_response("WATERMARK_EXTRACTION_FAILED", "No watermark detected", http_code=400)
            
            extracted_payload = bits_to_payload(extracted_bits[:256])
            
        except Exception as e:
            logger.error(f"Watermark extraction error: {str(e)}")
            return _create_error_response("WATERMARK_EXTRACTION_ERROR", str(e), http_code=500)
        
        processing_time = time.time() - request_start_time
        response_data = {
            "filename": file.filename,
            "extracted_payload": extracted_payload,
            "extraction": {
                "confidence": round(confidence, 3),
                "processing_time_seconds": round(processing_time, 3),
            },
        }
        
        if expected_payload:
            response_data["verification"] = {
                "status": "verified" if expected_payload == extracted_payload else "mismatch",
                "match": expected_payload == extracted_payload,
            }
        
        logger.info(f"Watermark extracted (confidence={confidence:.3f})")
        return _create_success_response(response_data)[0]
    
    except Exception as e:
        logger.error(f"Unhandled exception in /verify: {str(e)}")
        logger.error(traceback.format_exc())
        return _create_error_response("INTERNAL_ERROR", str(e), http_code=500)
    
    finally:
        if tmp_dicom_path and os.path.exists(tmp_dicom_path):
            try:
                os.remove(tmp_dicom_path)
            except Exception:
                pass


@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint."""
    return _create_success_response({
        "service": "DICOM ROI Extraction and Watermarking API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check and model status",
            "POST /roi/process": "Process DICOM and return ROI image (Phase 1A)",
            "POST /embed": "Embed watermark in DICOM file (Phase 1B)",
            "POST /verify": "Extract and verify watermark from image (Phase 1B)",
        },
        "roi_process_usage": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "parameters": {
                "file": {
                    "description": "DICOM file (.dcm or .dicom) - REQUIRED",
                    "type": "binary"
                },
                "alpha": {
                    "description": "Overlay transparency (0.0-1.0)",
                    "type": "float",
                    "default": 0.5
                },
                "device": {
                    "description": "Compute device (cpu or cuda)",
                    "type": "string",
                    "default": "cpu"
                },
                "return_format": {
                    "description": "Response format: 'image' (PNG only) or 'json' (JSON + base64 image)",
                    "type": "string",
                    "default": "json"
                }
            },
            "example_curl": 'curl -F "file=@input.dcm" http://localhost:5000/roi/process',
            "example_curl_image_only": 'curl -F "file=@input.dcm" -F "return_format=image" http://localhost:5000/roi/process -o output.png',
        },
        "response_format": {
            "success": {
                "status": "success",
                "filename": "string",
                "image": "data:image/png;base64,<PNG_BASE64>",
                "dimensions": {"width": 512, "height": 512, "bit_depth": 16},
                "dicom_metadata": {
                    "modality": "CT",
                    "patient_id": "xxx",
                    "study_date": "xxx"
                },
                "processing": {
                    "model": "vit_tiny_patch16_224",
                    "device": "cpu",
                    "alpha": 0.5,
                    "processing_time_seconds": 2.34
                },
                "timestamp": "2026-02-24T12:34:56.789000"
            },
            "error": {
                "status": "error",
                "error_code": "INVALID_DICOM",
                "message": "DICOM validation failed: ...",
                "details": "...",
                "timestamp": "2026-02-24T12:34:56.789000"
            }
        }
    })[0]


def main():
    """Start the API server."""
    global DEVICE, CHECKPOINT_PATH, MODEL_NAME
    import argparse
    
    parser = argparse.ArgumentParser(description='DICOM ROI Extraction API')
    parser.add_argument('--host', default='0.0.0.0', help='API host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='API port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Compute device')
    parser.add_argument('--checkpoint', default=str(CHECKPOINT_PATH), help='Path to checkpoint')
    parser.add_argument('--model-name', default=MODEL_NAME, help='ViT model name')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()
    
    # Update global configuration
    DEVICE = args.device
    CHECKPOINT_PATH = Path(args.checkpoint)
    MODEL_NAME = args.model_name
    
    # Configure logging
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("Starting DICOM ROI Extraction API")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"API documentation available at http://{args.host}:{args.port}/")
    logger.info("=" * 60)
    
    # Verify checkpoint exists
    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
