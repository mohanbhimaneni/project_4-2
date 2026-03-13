"""
Tamper localization helpers for fragile watermarking outputs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .fragile_watermark import detect_tamper_map, tamper_stats, FragileConfig


def localize_tamper_regions(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    config: FragileConfig = FragileConfig(),
) -> dict:
    """Return tamper map + summary for a fragile-watermarked image."""
    tamper_map = detect_tamper_map(image=image, roi_mask=roi_mask, config=config)
    stats = tamper_stats(tamper_map=tamper_map, block_size=config.block_size)

    return {
        "tamper_map": tamper_map,
        "summary": {
            "tampered": stats["tampered_blocks"] > 0,
            **stats,
        },
    }
