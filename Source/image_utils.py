"""
Image processing utilities: conversions, quality metrics, and normalizations.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def normalize_image(img: np.ndarray, clip_percentile: float = 0.0) -> np.ndarray:
    """
    Normalize image to [0, 1] range using min-max normalization.
    
    Args:
        img: Input image array
        clip_percentile: Optional percentile clipping (e.g., 0.5 clips 0.5% from each tail)
        
    Returns:
        Normalized image as float32 in [0, 1]
    """
    img = img.astype(np.float32)
    
    if clip_percentile > 0:
        lower = np.percentile(img, clip_percentile)
        upper = np.percentile(img, 100 - clip_percentile)
        img = np.clip(img, lower, upper)
    
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < 1e-10:
        return np.zeros_like(img)
    
    img = (img - img_min) / (img_max - img_min)
    return img


def denormalize_image(img: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """
    Reverse normalization to original value range.
    
    Args:
        img: Normalized image in [0, 1]
        original_min: Original minimum value
        original_max: Original maximum value
        
    Returns:
        Denormalized image
    """
    return img * (original_max - original_min) + original_min


def apply_window_level(
    img: np.ndarray,
    center: Optional[float] = None,
    width: Optional[float] = None
) -> np.ndarray:
    """
    Apply window/level transformation (common in medical imaging).
    
    Args:
        img: Input image
        center: Window center value
        width: Window width value
        
    Returns:
        Windowed image in [0, 1]
    """
    if center is None or width is None:
        return normalize_image(img)
    
    img = img.astype(np.float32)
    lower = center - width / 2
    upper = center + width / 2
    
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    
    return np.clip(img, 0, 1)


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    Args:
        original: Original image (uint8 or normalized float)
        modified: Modified image (same type as original)
        
    Returns:
        PSNR in dB
    """
    # Ensure same shape
    if original.shape != modified.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {modified.shape}")
    
    # Convert to float if needed
    original = original.astype(np.float32)
    modified = modified.astype(np.float32)
    
    # MSE
    mse = np.mean((original - modified) ** 2)
    
    if mse == 0:
        return 100.0  # Perfect match
    
    # Determine max value based on dtype
    if original.dtype == np.uint8:
        max_val = 255.0
    else:
        max_val = max(original.max(), modified.max())
    
    psnr_val = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr_val)


def ssim(original: np.ndarray, modified: np.ndarray, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        original: Original image
        modified: Modified image
        window_size: Size of Gaussian window
        
    Returns:
        SSIM score in [-1, 1], where 1 is perfect match
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning 1.0 as fallback")
        return 1.0
    
    # Convert to float tensors
    x = torch.tensor(original, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = torch.tensor(modified, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Normalize to [0, 1] if needed
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        x = (x - x_min) / (x_max - x_min)
    
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 0:
        y = (y - y_min) / (y_max - y_min)
    
    # Mean and variance
    c1, c2 = 0.01, 0.03
    
    mean_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mean_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    
    mean_xx = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2)
    mean_yy = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2)
    mean_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2)
    
    var_x = mean_xx - mean_x ** 2
    var_y = mean_yy - mean_y ** 2
    cov_xy = mean_xy - mean_x * mean_y
    
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / \
               ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2))
    
    return float(ssim_map.mean().item())


def array_to_pil_image(
    img: np.ndarray,
    normalize: bool = True,
    mode: str = 'L'
) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        img: Input numpy array (float or uint8)
        normalize: If True, normalize to [0, 1] first
        mode: PIL image mode ('L' for grayscale, 'RGB' for color)
        
    Returns:
        PIL Image
    """
    if normalize:
        img = normalize_image(img)
    
    # Convert to uint8
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    if mode == 'L':
        return Image.fromarray(img, mode='L')
    elif mode == 'RGB':
        # If grayscale, replicate to 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return Image.fromarray(img, mode='RGB')
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def pil_image_to_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array (float32 in [0, 1]).
    
    Args:
        img: PIL Image
        
    Returns:
        Normalized numpy array
    """
    arr = np.array(img).astype(np.float32)
    
    # Convert RGB to grayscale if needed
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    
    # Normalize
    if arr.max() > 1:
        arr = arr / 255.0
    
    return arr


def blend_images(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Blend two images with specified opacity.
    
    Args:
        base: Base image (uint8 or float in [0, 1])
        overlay: Overlay image (uint8 or float in [0, 1])
        alpha: Opacity of overlay (0=invisible, 1=opaque)
        
    Returns:
        Blended image (same dtype as base)
    """
    base = base.astype(np.float32)
    overlay = overlay.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if base.max() > 1:
        base = base / 255.0
    if overlay.max() > 1:
        overlay = overlay / 255.0
    
    alpha = np.clip(alpha, 0.0, 1.0)
    result = base * (1 - alpha) + overlay * alpha
    
    return result


def create_heatmap(
    mask: np.ndarray,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Convert grayscale mask to color heatmap.
    
    Args:
        mask: Grayscale mask in [0, 1] or [0, 255]
        colormap: Matplotlib colormap name
        
    Returns:
        RGB heatmap as uint8 array
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available, returning red channel")
        mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return np.stack([mask_uint8, np.zeros_like(mask_uint8), np.zeros_like(mask_uint8)], axis=-1)
    
    # Normalize to [0, 1]
    mask = mask.astype(np.float32)
    if mask.max() > 1:
        mask = mask / 255.0
    mask = normalize_image(mask)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(mask)
    
    # Extract RGB channels
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    
    return rgb


def resize_image(img: np.ndarray, target_size: Tuple[int, int], mode: str = 'bilinear') -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        img: Input image
        target_size: Target (height, width)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        
    Returns:
        Resized image
    """
    if not TORCH_AVAILABLE:
        # Fallback to PIL
        pil_img = array_to_pil_image(img)
        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        return pil_image_to_array(pil_img)
    
    # Use torch for consistency
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        img_t,
        size=target_size,
        mode=mode,
        align_corners=False
    )
    return resized.squeeze().numpy()


def get_roi_coverage(roi_mask: np.ndarray) -> float:
    """
    Calculate ROI coverage as percentage of total pixels.
    
    Args:
        roi_mask: Binary ROI mask (0 and 1)
        
    Returns:
        Coverage percentage (0-100)
    """
    if roi_mask.size == 0:
        return 0.0
    
    coverage = (np.sum(roi_mask) / roi_mask.size) * 100
    return float(coverage)


def get_roi_statistics(roi_mask: np.ndarray, image: np.ndarray) -> dict:
    """
    Calculate statistics for ROI region.
    
    Args:
        roi_mask: Binary ROI mask
        image: Original image
        
    Returns:
        Dictionary with statistics
    """
    roi_pixels = image[roi_mask == 1]
    non_roi_pixels = image[roi_mask == 0]
    
    stats = {
        'roi_mean': float(roi_pixels.mean()) if roi_pixels.size > 0 else 0.0,
        'roi_std': float(roi_pixels.std()) if roi_pixels.size > 0 else 0.0,
        'roi_min': float(roi_pixels.min()) if roi_pixels.size > 0 else 0.0,
        'roi_max': float(roi_pixels.max()) if roi_pixels.size > 0 else 0.0,
        'non_roi_mean': float(non_roi_pixels.mean()) if non_roi_pixels.size > 0 else 0.0,
        'non_roi_std': float(non_roi_pixels.std()) if non_roi_pixels.size > 0 else 0.0,
        'coverage_percent': get_roi_coverage(roi_mask),
    }
    
    return stats
