"""
Fragile block-wise watermarking for tamper detection.

Phase 1C implementation:
- Non-overlapping block processing (default 8x8)
- SHA-256 based block digest (truncated bits)
- LSB embedding of digest bits into neighboring eligible block
- Detects tampering and localizes modified blocks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib

import numpy as np


class FragileWatermarkException(Exception):
    """Raised when fragile watermark operations fail."""


@dataclass(frozen=True)
class FragileConfig:
    block_size: int = 8
    lsb_depth: int = 1
    digest_bits: int = 16
    seed: int = 1337


def _validate_inputs(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray],
    config: FragileConfig,
) -> None:
    if image.ndim != 2:
        raise FragileWatermarkException(f"image must be 2D grayscale, got shape {image.shape}")

    if image.dtype not in (np.uint8, np.uint16, np.int16, np.int32):
        raise FragileWatermarkException(f"unsupported dtype for fragile watermark: {image.dtype}")

    if config.block_size < 4:
        raise FragileWatermarkException("block_size must be >= 4")

    if not (1 <= config.lsb_depth <= 3):
        raise FragileWatermarkException("lsb_depth must be in [1, 3]")

    if not (8 <= config.digest_bits <= config.block_size * config.block_size):
        raise FragileWatermarkException(
            f"digest_bits must be in [8, {config.block_size * config.block_size}]"
        )

    if roi_mask is not None and roi_mask.shape != image.shape:
        raise FragileWatermarkException(
            f"roi_mask shape {roi_mask.shape} must match image shape {image.shape}"
        )


def _eligible_block(roi_block: Optional[np.ndarray]) -> bool:
    if roi_block is None:
        return True
    return not np.any(roi_block.astype(bool))


def _carrier_positions(
    block_size: int,
    block_row: int,
    block_col: int,
    seed: int,
    n_carriers: int,
) -> list[Tuple[int, int]]:
    local_seed = (seed * 1315423911 + block_row * 2654435761 + block_col * 2246822519) & 0xFFFFFFFF
    rng = np.random.default_rng(local_seed)
    total = block_size * block_size
    n = min(max(1, n_carriers), total)
    indices = rng.choice(total, size=n, replace=False)
    return [divmod(int(idx), block_size) for idx in indices]


def _iter_eligible_blocks(
    image_shape: Tuple[int, int],
    roi_mask: Optional[np.ndarray],
    block_size: int,
) -> list[Tuple[int, int, int, int, int, int]]:
    h, w = image_shape
    blocks_h = h // block_size
    blocks_w = w // block_size

    blocks = []
    for br in range(blocks_h):
        for bc in range(blocks_w):
            r0, r1 = br * block_size, (br + 1) * block_size
            c0, c1 = bc * block_size, (bc + 1) * block_size
            roi_block = roi_mask[r0:r1, c0:c1] if roi_mask is not None else None
            if _eligible_block(roi_block):
                blocks.append((br, bc, r0, r1, c0, c1))
    return blocks


def _block_digest_bits(block: np.ndarray, digest_bits: int, lsb_depth: int) -> list[int]:
    # Ignore low embedded bits to keep digest stable with respect to payload bits.
    high = np.right_shift(block.astype(np.int64), lsb_depth).astype(np.uint16, copy=False)
    digest = hashlib.sha256(high.tobytes()).digest()

    bits = []
    for byte in digest:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
            if len(bits) >= digest_bits:
                return bits
    return bits[:digest_bits]


def _embed_bits_to_positions(
    block: np.ndarray,
    bits: list[int],
    positions: list[Tuple[int, int]],
    lsb_depth: int,
) -> None:
    for bit, (r, c) in zip(bits, positions):
        block[r, c] = _embed_bit(int(block[r, c]), int(bit), lsb_depth)


def _extract_bits_from_positions(
    block: np.ndarray,
    positions: list[Tuple[int, int]],
    lsb_depth: int,
) -> list[int]:
    return [_extract_bit(int(block[r, c]), lsb_depth) for r, c in positions]


def _embed_bit(value: int, bit: int, lsb_depth: int) -> int:
    mask = (1 << lsb_depth) - 1
    return (int(value) & ~mask) | (bit & mask)


def _extract_bit(value: int, lsb_depth: int) -> int:
    return int(value) & ((1 << lsb_depth) - 1)


def embed_fragile_watermark(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    config: FragileConfig = FragileConfig(),
) -> np.ndarray:
    """Embed block integrity bits into image LSBs.

    Returns image with fragile watermark embedded.
    """
    _validate_inputs(image, roi_mask, config)

    block_size = config.block_size
    out = image.copy()

    eligible_blocks = _iter_eligible_blocks(image.shape, roi_mask, block_size)
    if not eligible_blocks:
        return out

    if len(eligible_blocks) < 2:
        raise FragileWatermarkException("image too small for configured block size")

    base = image.copy()

    for idx, (src_br, src_bc, src_r0, src_r1, src_c0, src_c1) in enumerate(eligible_blocks):
        dst_br, dst_bc, dst_r0, dst_r1, dst_c0, dst_c1 = eligible_blocks[(idx + 1) % len(eligible_blocks)]

        src_block = base[src_r0:src_r1, src_c0:src_c1]
        dst_block = out[dst_r0:dst_r1, dst_c0:dst_c1]

        bits = _block_digest_bits(src_block, config.digest_bits, config.lsb_depth)

        pos_seed = (config.seed ^ (src_br << 16) ^ (src_bc << 1) ^ dst_br ^ (dst_bc << 8)) & 0xFFFFFFFF
        positions = _carrier_positions(
            block_size,
            dst_br,
            dst_bc,
            pos_seed,
            n_carriers=config.digest_bits,
        )

        _embed_bits_to_positions(dst_block, bits, positions, config.lsb_depth)

    return out


def detect_tamper_map(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    config: FragileConfig = FragileConfig(),
) -> np.ndarray:
    """Return per-pixel tamper map (1=tampered, 0=clean) for non-ROI blocks."""
    _validate_inputs(image, roi_mask, config)

    block_size = config.block_size
    tamper = np.zeros_like(image, dtype=np.uint8)

    eligible_blocks = _iter_eligible_blocks(image.shape, roi_mask, block_size)
    if len(eligible_blocks) < 2:
        return tamper

    for idx, (src_br, src_bc, src_r0, src_r1, src_c0, src_c1) in enumerate(eligible_blocks):
        dst_br, dst_bc, dst_r0, dst_r1, dst_c0, dst_c1 = eligible_blocks[(idx + 1) % len(eligible_blocks)]

        src_block = image[src_r0:src_r1, src_c0:src_c1]
        dst_block = image[dst_r0:dst_r1, dst_c0:dst_c1]

        expected_bits = _block_digest_bits(src_block, config.digest_bits, config.lsb_depth)

        pos_seed = (config.seed ^ (src_br << 16) ^ (src_bc << 1) ^ dst_br ^ (dst_bc << 8)) & 0xFFFFFFFF
        positions = _carrier_positions(
            block_size,
            dst_br,
            dst_bc,
            pos_seed,
            n_carriers=config.digest_bits,
        )
        stored_bits = _extract_bits_from_positions(dst_block, positions, config.lsb_depth)

        if expected_bits != stored_bits:
            tamper[src_r0:src_r1, src_c0:src_c1] = 1

    return tamper


def tamper_stats(tamper_map: np.ndarray, block_size: int = 8) -> dict:
    """Summarize tamper density statistics."""
    if tamper_map.ndim != 2:
        raise FragileWatermarkException("tamper_map must be 2D")

    h, w = tamper_map.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    total_blocks = blocks_h * blocks_w

    tampered_blocks = 0
    for br in range(blocks_h):
        for bc in range(blocks_w):
            r0, r1 = br * block_size, (br + 1) * block_size
            c0, c1 = bc * block_size, (bc + 1) * block_size
            if np.any(tamper_map[r0:r1, c0:c1]):
                tampered_blocks += 1

    ratio = float(tampered_blocks / total_blocks) if total_blocks else 0.0
    return {
        "total_blocks": int(total_blocks),
        "tampered_blocks": int(tampered_blocks),
        "tampered_ratio": ratio,
    }
