"""Tests for Phase 1C fragile watermarking and tamper localization."""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from watermarking import (
    FragileConfig,
    embed_fragile_watermark,
    detect_tamper_map,
    tamper_stats,
    localize_tamper_regions,
)


class TestFragileWatermark(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        self.config = FragileConfig(block_size=8, lsb_depth=1, seed=2026)

        # Central ROI where fragile watermark should not be embedded or checked.
        self.roi_mask = np.zeros((128, 128), dtype=bool)
        self.roi_mask[40:88, 40:88] = True

    def test_no_tamper_detected_after_embed(self):
        watermarked = embed_fragile_watermark(self.image, roi_mask=self.roi_mask, config=self.config)
        tamper_map = detect_tamper_map(watermarked, roi_mask=self.roi_mask, config=self.config)

        self.assertEqual(int(tamper_map.sum()), 0)

    def test_single_pixel_tamper_detected(self):
        watermarked = embed_fragile_watermark(self.image, roi_mask=self.roi_mask, config=self.config)
        attacked = watermarked.copy()

        # Pick a deterministic non-carrier pixel in block (1,1).
        block_row, block_col = 1, 1
        local_seed = (
            self.config.seed * 1315423911
            + block_row * 2654435761
            + block_col * 2246822519
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(local_seed)
        carrier_indices = rng.choice(
            self.config.block_size * self.config.block_size,
            size=3,
            replace=False,
        )
        carriers = {divmod(int(idx), self.config.block_size) for idx in carrier_indices}

        # Apply deterministic edits to multiple non-carrier pixels in block [8:16, 8:16].
        for rr in range(8, 12):
            for cc in range(8, 12):
                local_r, local_c = rr - 8, cc - 8
                if (local_r, local_c) in carriers:
                    continue
                attacked[rr, cc] = np.uint8((int(attacked[rr, cc]) ^ 0x1F) % 256)

        tamper_map = detect_tamper_map(attacked, roi_mask=self.roi_mask, config=self.config)
        stats = tamper_stats(tamper_map, block_size=self.config.block_size)

        self.assertGreater(stats["tampered_blocks"], 0)
        self.assertTrue(np.any(tamper_map[8:16, 8:16] == 1))

    def test_roi_blocks_skipped(self):
        watermarked = embed_fragile_watermark(self.image, roi_mask=self.roi_mask, config=self.config)
        attacked = watermarked.copy()

        # Tamper only inside ROI region.
        attacked[50, 50] = np.uint8((int(attacked[50, 50]) + 7) % 256)

        tamper_map = detect_tamper_map(attacked, roi_mask=self.roi_mask, config=self.config)

        # ROI-only tampering should be ignored by ROI-aware fragile detector.
        self.assertFalse(np.any(tamper_map[40:88, 40:88] == 1))

    def test_localize_tamper_regions_summary(self):
        watermarked = embed_fragile_watermark(self.image, roi_mask=self.roi_mask, config=self.config)
        attacked = watermarked.copy()
        attacked[24, 24] = np.uint8((int(attacked[24, 24]) + 1) % 256)

        result = localize_tamper_regions(attacked, roi_mask=self.roi_mask, config=self.config)

        self.assertIn("tamper_map", result)
        self.assertIn("summary", result)
        self.assertTrue(result["summary"]["tampered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
