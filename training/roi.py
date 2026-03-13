# ROI extraction: ViT patch embeddings + k-means clustering
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
    import timm
    from sklearn.cluster import KMeans
    _vit_ok = True
except Exception as e:
    _vit_ok = False
    print("Missing deps for ViT ROI extraction:", e)

try:
    import pydicom
    _dicom_ok = True
except Exception as e:
    _dicom_ok = False
    print("Missing deps for DICOM read:", e)


_MODEL_CACHE = {}


def _normalize_image(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def _get_vit_model(model_name="vit_base_patch16_224", checkpoint=None, device=None):
    if not _vit_ok:
        raise RuntimeError("ViT dependencies missing. Install torch, timm, scikit-learn.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (model_name, str(checkpoint) if checkpoint else "", device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], device

    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    model = model.to(device).eval()
    _MODEL_CACHE[cache_key] = model
    return model, device


def _get_patch_embeddings(img_norm, model_name="vit_base_patch16_224", checkpoint=None, device=None):
    # img_norm: HxW in [0, 1]
    h, w = img_norm.shape
    # ViT expects 224x224 and 3 channels
    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    img_t = F.interpolate(img_t, size=(224, 224), mode="bilinear", align_corners=False)
    img_t = img_t.repeat(1, 3, 1, 1)

    model, device = _get_vit_model(model_name=model_name, checkpoint=checkpoint, device=device)
    img_t = img_t.to(device)

    with torch.inference_mode():
        feats = model.forward_features(img_t)

    if feats.ndim == 3 and feats.shape[1] > 1:
        feats = feats[:, 1:, :]

    num_patches = feats.shape[1]
    grid = int(np.sqrt(num_patches))
    feats = feats.reshape(1, grid, grid, -1)
    return feats.squeeze(0).cpu().numpy()


def _cluster_roi(patch_feats, img_norm, k=2):
    h, w = img_norm.shape
    ph, pw, d = patch_feats.shape
    feats_flat = patch_feats.reshape(-1, d)

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(feats_flat).reshape(ph, pw)

    scale_y = int(np.ceil(h / ph))
    scale_x = int(np.ceil(w / pw))
    mask = np.kron(labels, np.ones((scale_y, scale_x), dtype=np.int32))
    mask = mask[:h, :w]

    # Choose ROI cluster with preference for central, compact regions.
    # This avoids selecting large border/background-dominant regions.
    h_idx, w_idx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((h_idx - cy) ** 2 + (w_idx - cx) ** 2)
    max_dist = dist.max() + 1e-6
    center_weight = 1.0 - (dist / max_dist)

    # Border mask for border-touch penalty
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    best_label = 0
    best_score = -1e9
    for label in range(k):
        m = (mask == label)
        if not np.any(m):
            continue

        area_frac = float(m.mean())
        border_frac = float(m[border].mean())
        center_score = float(center_weight[m].mean())
        intensity = float(img_norm[m].mean())

        # Prefer central, moderate-size, low-border-touch regions.
        # Small intensity term helps disambiguate ties.
        score = (
            1.5 * center_score
            - 1.2 * border_frac
            - 0.7 * area_frac
            + 0.1 * intensity
        )

        if score > best_score:
            best_score = score
            best_label = label

    roi = (mask == best_label).astype(np.uint8)

    try:
        from scipy.ndimage import (
            binary_opening,
            binary_closing,
            binary_fill_holes,
            binary_erosion,
            label,
        )

        roi_bool = roi.astype(bool)

        # Remove speckles, smooth boundaries, and fill internal holes.
        roi_bool = binary_opening(roi_bool, iterations=1)
        roi_bool = binary_closing(roi_bool, iterations=1)
        roi_bool = binary_fill_holes(roi_bool)

        # Keep only significant central connected components (up to 2).
        lbl, n_comp = label(roi_bool)
        if n_comp > 0:
            min_area = max(64, int(0.003 * h * w))
            comp_info = []
            for comp_id in range(1, n_comp + 1):
                comp = (lbl == comp_id)
                area = int(comp.sum())
                if area < min_area:
                    continue
                comp_center = float(center_weight[comp].mean())
                comp_info.append((area, comp_center, comp_id))

            if comp_info:
                # Prefer large and central components, keep top 2 (e.g., bilateral anatomy).
                comp_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
                keep_ids = {cid for _, _, cid in comp_info[:2]}
                roi_bool = np.isin(lbl, list(keep_ids))

        # Slightly shrink ROI to reduce over-coverage.
        roi_bool = binary_erosion(roi_bool, iterations=1)
        roi = roi_bool.astype(np.uint8)
    except Exception:
        pass

    return roi


def _load_dicom_pixels(dcm_path):
    if not _dicom_ok:
        raise RuntimeError("pydicom is required. Install with: python -m pip install pydicom")
    ds = pydicom.dcmread(str(dcm_path))
    pixels = ds.pixel_array

    # Ensure 2D image for ROI model.
    # - 2D: use as-is
    # - 3D volume (frames, H, W): use middle frame
    # - 2D color (H, W, C): convert to grayscale
    # - 4D (frames, H, W, C): middle frame + grayscale
    if pixels.ndim == 2:
        selected = pixels
    elif pixels.ndim == 3:
        if pixels.shape[-1] in (3, 4):
            selected = np.mean(pixels[..., :3], axis=-1)
        else:
            mid_idx = pixels.shape[0] // 2
            selected = pixels[mid_idx]
    elif pixels.ndim == 4:
        mid_idx = pixels.shape[0] // 2
        frame = pixels[mid_idx]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            selected = np.mean(frame[..., :3], axis=-1)
        else:
            selected = frame
    else:
        raise ValueError(f"Unsupported DICOM pixel array dimensions: {pixels.shape}")

    return selected.astype(np.float32)


def extract_roi_from_dicom_file(dicom_path, model=None, model_name="vit_base_patch16_224", device=None, alpha=0.5):
    """
    Extract ROI from a DICOM file and return a PIL Image with overlay.
    
    Args:
        dicom_path: Path to DICOM file
        model: Optional pre-loaded model (for API usage with cached model)
        model_name: ViT model name from timm
        device: Compute device ('cpu' or 'cuda')
        alpha: Overlay transparency (0.0-1.0)
    
    Returns:
        PIL Image with ROI mask overlaid on DICOM, or None if extraction fails
        
    Raises:
        RuntimeError: If required dependencies are missing
        FileNotFoundError: If DICOM file not found
        ValueError: If DICOM has no pixel data
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate input file
        dicom_path = str(dicom_path)
        if not Path(dicom_path).exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        # Load DICOM pixels
        try:
            pixels = _load_dicom_pixels(dicom_path)
        except AttributeError as e:
            if "Pixel Data" in str(e):
                raise ValueError(f"DICOM file has no pixel data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading DICOM pixels: {str(e)}")
            raise
        
        # Validate pixel array
        if pixels is None or pixels.size == 0:
            raise ValueError("Pixel array is empty or None")
        
        if pixels.ndim < 2:
            raise ValueError(f"Invalid pixel array dimensions: {pixels.shape}")
        
        # Normalize image
        img_norm = _normalize_image(pixels)
        
        # Extract patch embeddings
        if model is not None:
            # Use provided/cached model
            if not _vit_ok:
                raise RuntimeError("ViT dependencies missing. Install torch, timm, scikit-learn.")
            
            img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
            img_t = F.interpolate(img_t, size=(224, 224), mode="bilinear", align_corners=False)
            img_t = img_t.repeat(1, 3, 1, 1)
            img_t = img_t.to(device)
            
            with torch.inference_mode():
                try:
                    feats = model.forward_features(img_t)
                except AttributeError as e:
                    logger.error(f"Model attribute error: {str(e)}")
                    logger.error(f"Model type: {type(model)}")
                    raise RuntimeError(f"Model does not have forward_features method: {str(e)}")
            
            if feats.ndim == 3 and feats.shape[1] > 1:
                feats = feats[:, 1:, :]
            
            num_patches = feats.shape[1]
            grid = int(np.sqrt(num_patches))
            feats = feats.reshape(1, grid, grid, -1)
            patch_feats = feats.squeeze(0).cpu().numpy()
        else:
            # Load model from checkpoint
            patch_feats = _get_patch_embeddings(
                img_norm,
                model_name=model_name,
                checkpoint=None,
                device=device
            )
        
        # Cluster and extract ROI
        roi_mask = _cluster_roi(patch_feats, img_norm)
        
        # Validate ROI mask
        if roi_mask is None or roi_mask.size == 0:
            raise ValueError("ROI mask is None or empty")
        
        if roi_mask.shape != pixels.shape[:2]:
            logger.warning(f"ROI mask shape {roi_mask.shape} doesn't match pixels {pixels.shape[:2]}")
        
        # Create PIL image with overlay
        from PIL import Image
        
        # Normalize image to 0-255 for display
        img_display = (img_norm * 255).astype(np.uint8)
        img_rgb = Image.fromarray(img_display).convert('RGB')
        
        # Create overlay
        img_array = np.array(img_rgb)
        roi_colored = np.zeros_like(img_array)
        
        # Apply red overlay to ROI
        roi_colored[roi_mask == 1] = [255, 0, 0]
        
        # Blend images
        alpha_val = min(max(alpha, 0.0), 1.0)
        img_array = (img_array * (1 - alpha_val) + roi_colored * alpha_val).astype(np.uint8)
        
        result_image = Image.fromarray(img_array)
        return result_image
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] Failed to extract ROI: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="ROI extraction for a single DICOM file")
    parser.add_argument("dcm_path", type=str, help="Path to a .dcm file")
    parser.add_argument("--output", type=str, default="output.png", help="Output PNG path")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help=(
            "Overlay transparency for ROI mask in [0, 1]. "
            "0 is fully transparent, 1 is fully opaque."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="ViT model name (timm)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to fine-tuned checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Device for inference (default: auto)",
    )
    args = parser.parse_args()

    dcm_path = Path(args.dcm_path)
    if not dcm_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")

    pixels = _load_dicom_pixels(dcm_path)
    img_norm = _normalize_image(pixels)

    if _vit_ok:
        patch_feats = _get_patch_embeddings(
            img_norm,
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        roi_mask = _cluster_roi(patch_feats, img_norm)

        # Visualize
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_norm, cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_norm, cmap="gray")
        alpha = min(max(args.alpha, 0.0), 1.0)
        plt.imshow(roi_mask, alpha=alpha, cmap="jet")
        plt.title("ROI Mask (Unsupervised)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(args.output, dpi=200, bbox_inches="tight")
        plt.show()
    else:
        print("ROI extraction skipped. Install torch, timm, scikit-learn.")


if __name__ == "__main__":
    main()
