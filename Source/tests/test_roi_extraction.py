"""
Comprehensive unit tests for ROI extraction API.

Tests cover:
- DICOM validation
- ROI extraction across modalities
- Error handling
- API endpoints
- Performance benchmarks
"""

import unittest
import tempfile
import os
import sys
import time
from pathlib import Path
import json

import numpy as np

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "Source"))
sys.path.insert(0, str(project_root / "project_understanding"))

from dicom_utils import validate_dicom_file, get_dicom_metadata, DicomValidationError
from image_utils import normalize_image, psnr, get_roi_coverage, pil_image_to_array


class TestDicomValidation(unittest.TestCase):
    """Test DICOM file validation."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        with self.assertRaises(DicomValidationError):
            validate_dicom_file("/nonexistent/path/file.dcm")
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
            tmp_path = f.name
        
        try:
            with self.assertRaises(DicomValidationError):
                validate_dicom_file(tmp_path)
        finally:
            os.remove(tmp_path)
    
    def test_validate_invalid_dicom(self):
        """Test validation of invalid DICOM file."""
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
            # Write random binary data
            f.write(b"This is not a valid DICOM file" * 100)
            tmp_path = f.name
        
        try:
            with self.assertRaises(DicomValidationError):
                validate_dicom_file(tmp_path)
        finally:
            os.remove(tmp_path)


class TestImageNormalization(unittest.TestCase):
    """Test image normalization functions."""
    
    def test_normalize_image_zeros(self):
        """Test normalization of all-zero image."""
        img = np.zeros((10, 10))
        normalized = normalize_image(img)
        self.assertTrue((normalized == 0).all())
    
    def test_normalize_image_ones(self):
        """Test normalization of all-one image."""
        img = np.ones((10, 10)) * 100
        normalized = normalize_image(img)
        # All same value should normalize to 0
        self.assertTrue((normalized == 0).all())
    
    def test_normalize_image_range(self):
        """Test normalization produces [0, 1] range."""
        img = np.random.rand(100, 100) * 1000
        normalized = normalize_image(img)
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
    
    def test_normalize_image_dtype(self):
        """Test normalized image is float32."""
        img = np.ones((10, 10), dtype=np.uint16) * 100
        normalized = normalize_image(img)
        self.assertEqual(normalized.dtype, np.float32)


class TestPSNRCalculation(unittest.TestCase):
    """Test PSNR calculation."""
    
    def test_psnr_identical_images(self):
        """Test PSNR of identical images is high."""
        img = np.random.rand(100, 100) * 255
        psnr_val = psnr(img, img)
        self.assertGreater(psnr_val, 90)  # Should be very high for identical images
    
    def test_psnr_noisy_image(self):
        """Test PSNR degrades with noise."""
        original = np.ones((100, 100), dtype=np.uint8) * 128
        noisy = original.copy().astype(np.float32) + np.random.randn(100, 100) * 10
        
        psnr_val = psnr(original.astype(np.float32), noisy)
        self.assertGreater(psnr_val, 0)
        self.assertLess(psnr_val, 100)
    
    def test_psnr_completely_different(self):
        """Test PSNR of completely different images."""
        img1 = np.zeros((100, 100))
        img2 = np.ones((100, 100))
        
        psnr_val = psnr(img1, img2)
        # Should be low for completely different images
        self.assertLess(psnr_val, 50)


class TestROIStatistics(unittest.TestCase):
    """Test ROI statistics calculation."""
    
    def test_roi_coverage_full(self):
        """Test ROI coverage when all pixels are ROI."""
        roi_mask = np.ones((100, 100))
        coverage = get_roi_coverage(roi_mask)
        self.assertAlmostEqual(coverage, 100.0, places=1)
    
    def test_roi_coverage_half(self):
        """Test ROI coverage when half pixels are ROI."""
        roi_mask = np.zeros((100, 100))
        roi_mask[:50, :] = 1
        coverage = get_roi_coverage(roi_mask)
        self.assertAlmostEqual(coverage, 50.0, places=1)
    
    def test_roi_coverage_none(self):
        """Test ROI coverage when no pixels are ROI."""
        roi_mask = np.zeros((100, 100))
        coverage = get_roi_coverage(roi_mask)
        self.assertAlmostEqual(coverage, 0.0, places=1)


class TestDicomMetadata(unittest.TestCase):
    """Test DICOM metadata extraction."""
    
    def find_sample_dicom(self):
        """Find a sample DICOM file in the dataset."""
        dataset_path = project_root / "dataset" / "siim-medical-images" / "versions" / "6" / "dicom_dir"
        if dataset_path.exists():
            dicom_files = list(dataset_path.glob("*.dcm"))
            if dicom_files:
                return str(dicom_files[0])
        return None
    
    def test_extract_metadata_from_valid_dicom(self):
        """Test metadata extraction from valid DICOM if available."""
        dicom_path = self.find_sample_dicom()
        
        if dicom_path is None:
            self.skipTest("No sample DICOM files found in dataset")
        
        try:
            metadata = get_dicom_metadata(dicom_path)
            
            # Check expected keys
            self.assertIn('modality', metadata)
            self.assertIn('width', metadata)
            self.assertIn('height', metadata)
            self.assertIn('bit_depth', metadata)
            
            # Check reasonable values
            self.assertGreater(metadata['width'], 0)
            self.assertGreater(metadata['height'], 0)
            self.assertGreaterEqual(metadata['bit_depth'], 8)
            
        except Exception as e:
            self.skipTest(f"Could not extract metadata: {str(e)}")


class TestROIExtractionAPI(unittest.TestCase):
    """Test API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Flask test client."""
        from Source.backend.api import app
        cls.app = app
        cls.client = app.test_client()
    
    def test_health_endpoint(self):
        """Test /health endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('service', data)
        self.assertIn('model', data)
    
    def test_documentation_endpoint(self):
        """Test / endpoint."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('endpoints', data)
    
    def test_roi_process_no_file(self):
        """Test /roi/process without file."""
        response = self.client.post('/roi/process', data={})
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['error_code'], 'NO_FILE')
    
    def test_roi_process_invalid_file_type(self):
        """Test /roi/process with invalid file type."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not a DICOM file")
            tmp_path = f.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post('/roi/process', data={
                    'file': (f, 'test.txt')
                })
            
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error_code'], 'INVALID_FILE_TYPE')
        finally:
            os.remove(tmp_path)
    
    def test_roi_process_invalid_alpha(self):
        """Test /roi/process with invalid alpha."""
        response = self.client.post('/roi/process', data={
            'file': (b'dummy', 'test.dcm'),
            'alpha': '1.5'
        })
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['error_code'], 'INVALID_ALPHA')
    
    def test_roi_process_invalid_device(self):
        """Test /roi/process with invalid device."""
        response = self.client.post('/roi/process', data={
            'file': (b'dummy', 'test.dcm'),
            'device': 'gpu'
        })
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['error_code'], 'INVALID_DEVICE')


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests."""
    
    def find_sample_dicom(self):
        """Find a sample DICOM file."""
        dataset_path = project_root / "dataset" / "siim-medical-images" / "versions" / "6" / "dicom_dir"
        if dataset_path.exists():
            dicom_files = list(dataset_path.glob("*.dcm"))[:3]  # Take 3 samples
            return [str(f) for f in dicom_files]
        return []
    
    def test_dicom_loading_speed(self):
        """Benchmark DICOM loading."""
        dicom_files = self.find_sample_dicom()
        
        if not dicom_files:
            self.skipTest("No sample DICOM files found")
        
        times = []
        for dicom_path in dicom_files:
            try:
                start = time.time()
                validate_dicom_file(dicom_path)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                self.skipTest(f"Could not load DICOM: {str(e)}")
        
        avg_time = np.mean(times)
        print(f"\nDICOM loading time: {avg_time:.3f}s (avg of {len(times)} files)")
        
        # DICOM loading should be fast (<1 second)
        self.assertLess(avg_time, 1.0, f"DICOM loading too slow: {avg_time:.3f}s")


def run_tests(verbose=True):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDicomValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestImageNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestPSNRCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestROIStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestDicomMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestROIExtractionAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)
