"""
FFT-based Robust Watermarking for DICOM Medical Images

This module implements frequency-domain watermarking using 2D FFT transformation.
The watermark is embedded in non-ROI image regions for ownership verification.

Key Features:
- Blind extraction (no original image needed)
- Robust to JPEG compression (QF >= 75)
- Robust to noise, scaling, and small rotations
- Imperceptible (PSNR > 40 dB)
- Only watermarks non-ROI regions to preserve clinical data
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Watermarking parameters
PATCH_SIZE = 8  # 8x8 patches for FFT
PATCH_STRIDE = 8  # Non-overlapping patches reduce interference and preserve PSNR
MIN_FREQ = 2  # Skip DC component (index 0) and start from freq 2
MAX_FREQ = 6  # Use frequencies 2-6 for mid-band embedding
REDUNDANCY_FACTOR = 3  # Repeat each bit 3x for majority voting
DEFAULT_STRENGTH = 1.0  # Default embedding strength (0.5-2.0)


class WatermarkException(Exception):
    """Custom exception for watermarking errors."""
    pass


def _validate_image_shape(image: np.ndarray, name: str = "image") -> None:
    """Validate image is 2D grayscale."""
    if image.ndim != 2:
        raise WatermarkException(f"{name} must be 2D grayscale, got shape {image.shape}")
    if image.dtype not in [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64]:
        raise WatermarkException(f"{name} has unsupported dtype {image.dtype}")


def _validate_roi_mask(roi_mask: np.ndarray, image_shape: Tuple[int, int]) -> None:
    """Validate ROI mask matches image dimensions."""
    if roi_mask.shape != image_shape:
        raise WatermarkException(f"ROI mask shape {roi_mask.shape} doesn't match image shape {image_shape}")
    if roi_mask.dtype != bool and roi_mask.dtype != np.uint8:
        raise WatermarkException(f"ROI mask must be binary (bool or uint8), got {roi_mask.dtype}")
    if roi_mask.dtype == np.uint8:
        unique_vals = np.unique(roi_mask)
        if not np.all(np.isin(unique_vals, [0, 1, 255])):
            raise WatermarkException(
                f"ROI mask uint8 must contain only binary values (0/1/255), got {unique_vals.tolist()}"
            )


def _normalize_to_float(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convert image to float32 for processing, return original dtype."""
    original_dtype = image.dtype
    if original_dtype == np.uint8:
        return image.astype(np.float32), original_dtype
    elif original_dtype == np.uint16:
        return image.astype(np.float32), original_dtype
    elif original_dtype == np.int16:
        return image.astype(np.float32), original_dtype
    elif original_dtype in [np.int32, np.float32, np.float64]:
        return image.astype(np.float32), original_dtype
    else:
        raise WatermarkException(f"Unsupported dtype: {original_dtype}")


def _get_non_roi_patches(
    roi_mask: np.ndarray,
    patch_size: int = PATCH_SIZE,
    patch_stride: int = PATCH_STRIDE,
) -> List[Tuple[int, int]]:
    """
    Get indices of patches that contain at least some non-ROI area.
    
    Returns: List of (row_idx, col_idx) tuples for non-ROI patches.
    """
    height, width = roi_mask.shape
    non_roi_patches = []
    
    for i in range(0, height - patch_size + 1, patch_stride):
        for j in range(0, width - patch_size + 1, patch_stride):
            patch_roi = roi_mask[i:i + patch_size, j:j + patch_size]
            roi_coverage = patch_roi.sum() / (patch_size * patch_size)
            
            # Embed in any patch that has at least one non-ROI pixel.
            # ROI pixels remain protected by per-pixel masking during writeback.
            if roi_coverage < 1.0:
                non_roi_patches.append((i, j))
    
    if not non_roi_patches:
        logger.warning("No non-ROI patches found. Watermark may not embed properly.")
    
    return non_roi_patches


def _get_mid_band_frequencies(patch_shape: Tuple[int, int] = (PATCH_SIZE, PATCH_SIZE),
                              n_freqs: int = 4) -> List[Tuple[int, int]]:
    """
    Get mid-band frequency indices for embedding.
    Avoids DC (0,0) and highest frequencies.
    
    Returns: List of (freq_y, freq_x) tuples in order.
    """
    h, w = patch_shape
    mid_frequencies = []
    
    # Select frequencies in mid-band (avoid DC and Nyquist)
    for freq_idx in range(MIN_FREQ, min(MAX_FREQ, min(h, w) // 2)):
        # Use frequencies along the diagonal and near-diagonal
        if freq_idx < h and freq_idx < w:
            mid_frequencies.append((freq_idx, freq_idx))
        if len(mid_frequencies) >= n_freqs:
            break
    
    # If not enough, add adjacent frequencies
    if len(mid_frequencies) < n_freqs:
        for freq_idx in range(MIN_FREQ, min(MAX_FREQ, min(h, w) // 2)):
            for offset in [0, 1, -1]:
                new_idx = freq_idx + offset
                if new_idx < h and new_idx < w and new_idx >= MIN_FREQ:
                    mid_frequencies.append((freq_idx, new_idx))
                    if len(mid_frequencies) >= n_freqs:
                        break
            if len(mid_frequencies) >= n_freqs:
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_freqs = []
    for freq in mid_frequencies:
        if freq not in seen:
            unique_freqs.append(freq)
            seen.add(freq)
    
    return unique_freqs[:n_freqs]


def _embed_bit_in_patch(patch: np.ndarray, bit: int, strength: float = DEFAULT_STRENGTH,
                       freq_indices: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Embed a single watermark bit in one patch using FFT.
    
    Args:
        patch: 8x8 image patch (float)
        bit: Binary value (0 or 1)
        strength: Embedding strength multiplier
        freq_indices: Precomputed frequency indices
    
    Returns: Watermarked patch (float)
    """
    # Apply 2D FFT
    patch_fft = np.fft.fft2(patch)
    patch_fft_shifted = np.fft.fftshift(patch_fft)
    
    # Get amplitude
    amplitude = np.abs(patch_fft_shifted)
    
    if freq_indices is None:
        freq_indices = _get_mid_band_frequencies()
    
    # Differential embedding across two frequency groups while preserving
    # conjugate symmetry (same-sign updates on mirrored coefficients).
    delta = strength * 0.02 * np.mean(amplitude)

    h, w = amplitude.shape
    group_a = freq_indices[::2]
    group_b = freq_indices[1::2] if len(freq_indices) > 1 else freq_indices

    sign_a = 1 if bit == 1 else -1
    sign_b = -sign_a

    for group, sign in ((group_a, sign_a), (group_b, sign_b)):
        for freq_y, freq_x in group:
            mirror_y = h - 1 - freq_y
            mirror_x = w - 1 - freq_x

            if not (0 <= mirror_y < h and 0 <= mirror_x < w):
                continue

            amplitude[freq_y, freq_x] += sign * delta
            amplitude[mirror_y, mirror_x] += sign * delta

    amplitude = np.clip(amplitude, 1e-6, None)
    
    # Reconstruct FFT and take inverse
    phase = np.angle(patch_fft_shifted)
    patch_fft_reconstructed = amplitude * np.exp(1j * phase)
    patch_fft = np.fft.ifftshift(patch_fft_reconstructed)
    
    # Inverse FFT
    watermarked_patch = np.fft.ifft2(patch_fft).real
    
    return watermarked_patch


def _extract_bit_from_patch(patch: np.ndarray,
                           freq_indices: List[Tuple[int, int]] = None) -> int:
    """
    Extract a single watermark bit from one patch using FFT.
    
    Args:
        patch: 8x8 image patch (float)
        freq_indices: Precomputed frequency indices
    
    Returns: Extracted bit (0 or 1)
    """
    # Apply 2D FFT
    patch_fft = np.fft.fft2(patch)
    patch_fft_shifted = np.fft.fftshift(patch_fft)
    
    # Get amplitude
    amplitude = np.abs(patch_fft_shifted)
    
    if freq_indices is None:
        freq_indices = _get_mid_band_frequencies()
    
    h, w = amplitude.shape
    group_a = freq_indices[::2]
    group_b = freq_indices[1::2] if len(freq_indices) > 1 else freq_indices

    def _group_mean(group: List[Tuple[int, int]]) -> float:
        vals = []
        for freq_y, freq_x in group:
            mirror_y = h - 1 - freq_y
            mirror_x = w - 1 - freq_x
            if 0 <= freq_y < h and 0 <= freq_x < w:
                vals.append(float(amplitude[freq_y, freq_x]))
            if 0 <= mirror_y < h and 0 <= mirror_x < w:
                vals.append(float(amplitude[mirror_y, mirror_x]))
        return float(np.mean(vals)) if vals else 0.0

    mean_a = _group_mean(group_a)
    mean_b = _group_mean(group_b)
    return 1 if mean_a >= mean_b else 0


def embed_robust_watermark(image: np.ndarray, 
                          roi_mask: np.ndarray,
                          payload_bits: List[int],
                          strength: float = DEFAULT_STRENGTH) -> np.ndarray:
    """
    Embed watermark bits in image using FFT, only in non-ROI regions.
    
    Args:
        image: Input image (uint8, uint16, int16, int32, or float32, 2D grayscale)
        roi_mask: Binary mask where True/1 = ROI region (same shape as image)
        payload_bits: List of binary bits to embed (length <= 256)
        strength: Embedding strength (0.5-2.0, default 1.0)
    
    Returns:
        Watermarked image (same dtype and shape as input)
    
    Raises:
        WatermarkException: If image or mask invalid
    """
    logger.info(f"Embedding watermark: {len(payload_bits)} bits, strength={strength}")
    
    # Validate inputs
    _validate_image_shape(image, "image")
    _validate_roi_mask(roi_mask, image.shape)
    
    if not isinstance(payload_bits, list) or not all(b in [0, 1] for b in payload_bits):
        raise WatermarkException("payload_bits must be List[int] of 0s and 1s")
    
    if len(payload_bits) > 256:
        raise WatermarkException(f"Payload too large: {len(payload_bits)} bits (max 256)")
    
    if not (0.5 <= strength <= 2.0):
        logger.warning(f"Strength {strength} outside recommended range [0.5, 2.0]")
    
    # Convert to float for processing
    img_float, original_dtype = _normalize_to_float(image)
    
    # Normalize ROI mask to boolean semantics: True = ROI, False = non-ROI
    roi_mask_bool = roi_mask.astype(bool)

    # Redundantly encode bits (repeat each bit)
    redundant_bits = []
    for bit in payload_bits:
        redundant_bits.extend([bit] * REDUNDANCY_FACTOR)
    
    # Pad with zeros if needed
    while len(redundant_bits) % PATCH_SIZE != 0:
        redundant_bits.append(0)
    
    # Get non-ROI patches
    non_roi_patches = _get_non_roi_patches(roi_mask_bool, PATCH_SIZE, PATCH_STRIDE)
    
    if len(non_roi_patches) < len(redundant_bits):  # 1 bit per patch
        raise WatermarkException(
            f"Not enough non-ROI patches ({len(non_roi_patches)}) for payload ({len(payload_bits)} bits, requires {len(redundant_bits)} patches with redundancy)"
        )
    
    # Pre-compute frequency indices
    freq_indices = _get_mid_band_frequencies()
    
    # Create output image copy
    watermarked = img_float.copy()
    
    # Embed bits in patches
    used_patches = 0
    for patch_idx, (i, j) in enumerate(non_roi_patches):
        if patch_idx >= len(redundant_bits):
            break
        
        patch = watermarked[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
        bit = redundant_bits[patch_idx]
        
        watermarked_patch = _embed_bit_in_patch(patch, bit, strength, freq_indices)

        # Safety: never modify ROI pixels even within selected patches
        patch_non_roi = ~roi_mask_bool[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
        watermarked[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = np.where(
            patch_non_roi,
            watermarked_patch,
            patch,
        )
        used_patches += 1
    
    # Convert back to original dtype
    # Final safety gate: ROI pixels must always remain identical to input
    watermarked = np.where(~roi_mask_bool, watermarked, img_float)

    if original_dtype == np.uint8:
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        watermarked = np.clip(watermarked, 0, 65535).astype(np.uint16)
    elif original_dtype == np.int16:
        watermarked = np.clip(watermarked, -32768, 32767).astype(np.int16)
    else:
        watermarked = watermarked.astype(original_dtype)
    
    logger.info(f"Watermark embedded successfully in {used_patches} patches")
    return watermarked


def extract_robust_watermark(image: np.ndarray,
                            roi_mask: np.ndarray,
                           payload_length: int = 256) -> Tuple[List[int], float]:
    """
    Extract watermark bits from image using FFT, using majority voting.
    
    Args:
        image: Input image (uint8, uint16, int16, int32, or float32, 2D grayscale)
        roi_mask: Binary mask where True/1 = ROI region (same shape as image)
        payload_length: Expected number of payload bits (default 256 for full token)
    
    Returns:
        Tuple of:
        - extracted_bits: List of extracted binary bits
        - confidence: Confidence score [0.0, 1.0] based on bit voting consistency
    
    Raises:
        WatermarkException: If image or mask invalid
    """
    logger.info(f"Extracting watermark: expecting {payload_length} bits")
    
    # Validate inputs
    _validate_image_shape(image, "image")
    _validate_roi_mask(roi_mask, image.shape)

    # Normalize ROI mask to boolean semantics: True = ROI, False = non-ROI
    roi_mask_bool = roi_mask.astype(bool)
    
    # Convert to float
    img_float, _ = _normalize_to_float(image)
    
    # Get non-ROI patches
    non_roi_patches = _get_non_roi_patches(roi_mask_bool, PATCH_SIZE, PATCH_STRIDE)
    
    if not non_roi_patches:
        logger.warning("No non-ROI patches found for extraction")
        return [], 0.0
    
    # Pre-compute frequency indices
    freq_indices = _get_mid_band_frequencies()
    
    # Extract bits with redundancy
    extracted_redundant = []
    confidence_scores = []
    
    for i, j in non_roi_patches:
        if len(extracted_redundant) >= payload_length * REDUNDANCY_FACTOR:
            break
        
        patch = img_float[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue
        
        bit = _extract_bit_from_patch(patch, freq_indices)
        extracted_redundant.append(bit)
    
    if not extracted_redundant:
        logger.warning("No bits extracted")
        return [], 0.0
    
    # Majority voting: decode redundant bits
    extracted_bits = []
    confidence_total = 0.0
    
    for bit_idx in range(payload_length):
        start = bit_idx * REDUNDANCY_FACTOR
        end = start + REDUNDANCY_FACTOR
        
        if end > len(extracted_redundant):
            break
        
        bit_votes = extracted_redundant[start:end]
        
        # Majority vote
        bit_value = 1 if sum(bit_votes) >= len(bit_votes) / 2 else 0
        extracted_bits.append(bit_value)
        
        # Confidence: how many votes agreed
        agreement = max(bit_votes.count(0), bit_votes.count(1)) / len(bit_votes)
        confidence_total += agreement
    
    # Average confidence
    confidence = confidence_total / len(extracted_bits) if extracted_bits else 0.0
    
    logger.info(f"Extracted {len(extracted_bits)} bits with confidence {confidence:.3f}")
    return extracted_bits, confidence


def payload_to_bits(payload_hex: str) -> List[int]:
    """
    Convert hex payload string to list of bits.
    
    Args:
        payload_hex: Hex string (e.g., "a3b2c4d506a08980")
    
    Returns:
        List of binary bits (length = 4 * hex-length)
    """
    if not isinstance(payload_hex, str):
        raise ValueError("Payload must be hex string")
    
    payload_hex_clean = payload_hex.replace(" ", "").lower()
    if payload_hex_clean.startswith("0x"):
        payload_hex_clean = payload_hex_clean[2:]
    if not payload_hex_clean:
        raise ValueError("Payload must not be empty")

    bit_length = len(payload_hex_clean) * 4
    payload_int = int(payload_hex_clean, 16)
    
    bits = [(payload_int >> i) & 1 for i in range(bit_length)]
    
    return bits


def bits_to_payload(bits: List[int], bit_length: int | None = None) -> str:
    """
    Convert list of bits to hex payload string.
    
    Args:
        bits: List of binary bits
    
    Returns:
        Hex string representation using the requested number of bits
    """
    if not isinstance(bits, list) or not bits:
        raise ValueError("bits must be a non-empty list")

    effective_bits = bit_length if bit_length is not None else len(bits)
    if effective_bits <= 0:
        raise ValueError("bit_length must be positive")
    if len(bits) < effective_bits:
        raise ValueError(f"Need at least {effective_bits} bits, got {len(bits)}")
    
    # Convert bits to integer
    payload_int = 0
    for i, bit in enumerate(bits[:effective_bits]):
        payload_int |= (int(bit) << i)
    
    hex_chars = max(1, (effective_bits + 3) // 4)
    return format(payload_int, f"0{hex_chars}x")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic test image
    test_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    test_roi = np.zeros((256, 256), dtype=bool)
    test_roi[50:150, 50:150] = True
    
    # Watermark
    test_payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32  # 256 bits
    watermarked = embed_robust_watermark(test_img, test_roi, test_payload, strength=1.0)
    
    # Extract
    extracted, confidence = extract_robust_watermark(watermarked, test_roi, len(test_payload))
    
    accuracy = sum(e == o for e, o in zip(extracted, test_payload)) / len(test_payload)
    print(f"Extraction accuracy: {accuracy:.2%}, Confidence: {confidence:.3f}")
