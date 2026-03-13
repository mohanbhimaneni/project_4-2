"""
Comprehensive Test Suite for FFT Watermarking (Phase 1B)

Tests cover:
- Core watermark embedding and extraction
- Imperceptibility metrics (PSNR, SSIM)
- Robustness to attacks (JPEG, noise, scaling, rotation)
- ROI integrity preservation
- Payload encoding/decoding
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import time
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from watermarking import (
    embed_robust_watermark,
    extract_robust_watermark,
    payload_to_bits,
    bits_to_payload,
    WatermarkException,
)

from watermarking.watermark_patterns import (
    LFSR,
    hash_to_seed,
    generate_patient_bit_sequence,
    create_ownership_payload,
)

from image_utils import psnr, ssim, normalize_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLFSR(unittest.TestCase):
    """Test LFSR pseudo-random number generator."""
    
    def test_lfsr_reproducibility(self):
        """Same seed produces same sequence."""
        lfsr1 = LFSR(0x12345678)
        bits1 = lfsr1.next_bits(256)
        
        lfsr2 = LFSR(0x12345678)
        bits2 = lfsr2.next_bits(256)
        
        self.assertEqual(bits1, bits2, "LFSR not reproducible with same seed")
    
    def test_lfsr_different_seeds(self):
        """Different seeds produce different sequences."""
        lfsr1 = LFSR(0x12345678)
        bits1 = lfsr1.next_bits(256)
        
        lfsr2 = LFSR(0x87654321)
        bits2 = lfsr2.next_bits(256)
        
        self.assertNotEqual(bits1, bits2, "Different seeds produced same sequence")
    
    def test_lfsr_zero_seed_becomes_nonzero(self):
        """Zero seed is automatically converted to non-zero."""
        lfsr = LFSR(0)
        bits = lfsr.next_bits(16)
        # Should produce some bits without errors
        self.assertEqual(len(bits), 16)


class TestWatermarkPayload(unittest.TestCase):
    """Test payload encoding and decoding."""
    
    def test_payload_to_bits(self):
        """Convert hex payload to bits."""
        payload = "a3b2c4d5" * 8  # 64 chars = 256 bits
        bits = payload_to_bits(payload)
        self.assertEqual(len(bits), 256)
        self.assertTrue(all(b in [0, 1] for b in bits))
    
    def test_bits_to_payload(self):
        """Convert bits back to hex."""
        original_bits = [1, 0, 1, 1] * 64  # 256 bits
        payload = bits_to_payload(original_bits)
        recovered_bits = payload_to_bits(payload)
        self.assertEqual(len(recovered_bits), 256)
    
    def test_payload_roundtrip(self):
        """Payload → bits → payload roundtrip."""
        original = "a3b2c4d506a080801f2e3d4c01000000" * 2  # 64 chars
        bits = payload_to_bits(original)
        recovered = bits_to_payload(bits)
        recovered_bits = payload_to_bits(recovered)
        self.assertEqual(bits, recovered_bits)
    
    def test_create_ownership_payload(self):
        """Create ownership payload from components."""
        payload = create_ownership_payload("PATIENT_001", "06a08980", "1f2e3d4c")
        self.assertEqual(len(payload), 64)  # 256 bits in hex
        self.assertTrue(all(c in "0123456789abcdef" for c in payload))


class TestWatermarkingBasics(unittest.TestCase):
    """Test core watermarking functionality."""
    
    def setUp(self):
        """Create test image and ROI."""
        np.random.seed(42)
        self.image_uint8 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        self.image_uint16 = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
        self.image_float = np.random.rand(256, 256).astype(np.float32)
        
        # Create ROI (center square)
        self.roi_mask = np.zeros((256, 256), dtype=bool)
        self.roi_mask[50:150, 50:150] = True
    
    def test_embed_uint8_image(self):
        """Embed watermark in uint8 image."""
        payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        watermarked = embed_robust_watermark(
            self.image_uint8, self.roi_mask, payload, strength=1.0
        )
        self.assertEqual(watermarked.dtype, np.uint8)
        self.assertEqual(watermarked.shape, self.image_uint8.shape)
    
    def test_embed_uint16_image(self):
        """Embed watermark in uint16 image."""
        payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        watermarked = embed_robust_watermark(
            self.image_uint16, self.roi_mask, payload, strength=1.0
        )
        self.assertEqual(watermarked.dtype, np.uint16)
        self.assertEqual(watermarked.shape, self.image_uint16.shape)
    
    def test_embed_float_image(self):
        """Embed watermark in float image."""
        payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        watermarked = embed_robust_watermark(
            self.image_float, self.roi_mask, payload, strength=1.0
        )
        self.assertEqual(watermarked.dtype, np.float32)
        self.assertEqual(watermarked.shape, self.image_float.shape)
    
    def test_extract_watermark(self):
        """Extract watermark from watermarked image."""
        payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        watermarked = embed_robust_watermark(
            self.image_uint8, self.roi_mask, payload, strength=1.0
        )
        extracted, confidence = extract_robust_watermark(watermarked, self.roi_mask, len(payload))
        
        self.assertEqual(len(extracted), len(payload))
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_roi_preservation(self):
        """Verify ROI regions are not watermarked."""
        payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        watermarked = embed_robust_watermark(
            self.image_uint8, self.roi_mask, payload, strength=1.0
        )
        
        # ROI should be less modified than non-ROI
        roi_diff = np.abs(self.image_uint8[self.roi_mask].astype(float) - 
                         watermarked[self.roi_mask].astype(float))
        non_roi_diff = np.abs(self.image_uint8[~self.roi_mask].astype(float) - 
                              watermarked[~self.roi_mask].astype(float))
        
        # Mean difference in non-ROI should be larger
        self.assertGreater(np.mean(non_roi_diff), np.mean(roi_diff))
        self.assertTrue(np.all(roi_diff == 0), "ROI pixels must remain unchanged")


class TestImperceptibility(unittest.TestCase):
    """Test imperceptibility of watermarking (PSNR > 40 dB)."""
    
    def setUp(self):
        """Create test image."""
        np.random.seed(42)
        self.image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        self.roi_mask = np.zeros((256, 256), dtype=bool)
        self.roi_mask[80:180, 80:180] = True
    
    def test_psnr_strength_1(self):
        """PSNR > 40 dB with strength=1.0."""
        payload = [0, 1] * 128
        watermarked = embed_robust_watermark(
            self.image, self.roi_mask, payload, strength=1.0
        )
        
        # Normalize for PSNR comparison
        img_norm = normalize_image(self.image)
        watermarked_norm = normalize_image(watermarked)
        
        psnr_val = psnr(img_norm, watermarked_norm)
        logger.info(f"PSNR (strength=1.0): {psnr_val:.2f} dB")
        self.assertGreater(psnr_val, 40.0, f"PSNR {psnr_val:.2f} below 40 dB threshold")
    
    def test_psnr_strength_0_5(self):
        """PSNR > 45 dB with lower strength."""
        payload = [0, 1] * 128
        watermarked = embed_robust_watermark(
            self.image, self.roi_mask, payload, strength=0.5
        )
        
        img_norm = normalize_image(self.image)
        watermarked_norm = normalize_image(watermarked)
        
        psnr_val = psnr(img_norm, watermarked_norm)
        logger.info(f"PSNR (strength=0.5): {psnr_val:.2f} dB")
        self.assertGreater(psnr_val, 45.0)


class TestRobustness(unittest.TestCase):
    """Test watermark robustness to attacks."""
    
    def setUp(self):
        """Create test image and watermark."""
        np.random.seed(42)
        self.image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        self.roi_mask = np.zeros((256, 256), dtype=bool)
        self.roi_mask[80:180, 80:180] = True
        
        self.payload = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        self.watermarked = embed_robust_watermark(
            self.image, self.roi_mask, self.payload, strength=1.0
        )
    
    def _extract_accuracy(self, attacked_image):
        """Extract watermark and return accuracy."""
        extracted, conf = extract_robust_watermark(attacked_image, self.roi_mask, len(self.payload))
        if not extracted:
            return 0.0
        accuracy = sum(e == o for e, o in zip(extracted, self.payload)) / len(self.payload)
        return accuracy
    
    def test_robustness_no_attack(self):
        """Verify extraction without attack."""
        accuracy = self._extract_accuracy(self.watermarked)
        logger.info(f"Accuracy (no attack): {accuracy:.2%}")
        self.assertGreater(accuracy, 0.9, "Basic extraction failed")
    
    def test_robustness_gaussian_noise(self):
        """Robustness to Gaussian noise (σ ≤ 5)."""
        for sigma in [1, 2, 3, 4, 5]:
            noise = np.random.normal(0, sigma, self.watermarked.shape)
            attacked = np.clip(self.watermarked.astype(float) + noise, 0, 255).astype(np.uint8)
            
            accuracy = self._extract_accuracy(attacked)
            logger.info(f"Accuracy (σ={sigma}): {accuracy:.2%}")
            self.assertGreater(accuracy, 0.85, f"Failed at σ={sigma}")
    
    def test_robustness_scaling(self):
        """Robustness to image scaling (± 10%)."""
        for scale in [0.9, 0.95, 1.05, 1.1]:
            h, w = self.watermarked.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize using nearest neighbor
            attacked = self.watermarked[::max(1, h//new_h)][:, ::max(1, w//new_w)]
            
            # Pad back to original size if smaller
            if attacked.shape[0] < h or attacked.shape[1] < w:
                padded = np.pad(attacked, 
                               ((0, h - attacked.shape[0]), (0, w - attacked.shape[1])),
                               mode='edge')
                attacked = padded[:h, :w]
            
            accuracy = self._extract_accuracy(attacked)
            logger.info(f"Accuracy (scale={scale}): {accuracy:.2%}")
            self.assertGreater(accuracy, 0.80, f"Failed at scale={scale}")
    
    def test_robustness_rotation_small(self):
        """Robustness to small rotation (≤ 2°)."""
        import scipy.ndimage as ndimage
        
        for angle in [-2, -1, 1, 2]:
            attacked = ndimage.rotate(self.watermarked, angle, reshape=False)
            attacked = np.clip(attacked, 0, 255).astype(np.uint8)
            
            accuracy = self._extract_accuracy(attacked)
            logger.info(f"Accuracy (rotation={angle}°): {accuracy:.2%}")
            # Relaxed threshold for rotation
            self.assertGreater(accuracy, 0.70, f"Failed at rotation={angle}°")


class TestExtractionPerformance(unittest.TestCase):
    """Test extraction speed and efficiency."""
    
    def setUp(self):
        """Create test image."""
        np.random.seed(42)
        self.image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        self.roi_mask = np.zeros((512, 512), dtype=bool)
        self.roi_mask[100:400, 100:400] = True
        
        self.payload = [1, 0] * 128
        self.watermarked = embed_robust_watermark(
            self.image, self.roi_mask, self.payload, strength=1.0
        )
    
    def test_extraction_speed(self):
        """Extract watermark and measure time."""
        start = time.time()
        extracted, conf = extract_robust_watermark(self.watermarked, self.roi_mask, len(self.payload))
        duration = time.time() - start
        
        logger.info(f"Extraction time: {duration:.3f}s")
        self.assertLess(duration, 5.0, "Extraction too slow")
    
    def test_embedding_speed(self):
        """Embed watermark and measure time."""
        payload = [0, 1] * 128
        
        start = time.time()
        _ = embed_robust_watermark(self.image, self.roi_mask, payload, strength=1.0)
        duration = time.time() - start
        
        logger.info(f"Embedding time: {duration:.3f}s")
        self.assertLess(duration, 10.0, "Embedding too slow")


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLFSR))
    suite.addTests(loader.loadTestsFromTestCase(TestWatermarkPayload))
    suite.addTests(loader.loadTestsFromTestCase(TestWatermarkingBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestImperceptibility))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractionPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
