"""
DICOM file utilities: validation, header parsing, and modality detection.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

logger = logging.getLogger(__name__)


class DicomValidationError(Exception):
    """Raised when DICOM validation fails."""
    pass


# Modality-specific windowing parameters (center, width)
MODALITY_WINDOWS = {
    'CT': (40, 400),      # Soft tissue window
    'MR': (None, None),   # No windowing for MR
    'XC': (128, 256),     # General radiography
    'CR': (128, 256),     # Computed radiography
    'DX': (128, 256),     # Digital radiography
    'MG': (2000, 4000),   # Mammography
    'US': (None, None),   # No windowing for ultrasound
    'PT': (0, 1),         # PET (0-1 range typically)
}


def validate_dicom_file(file_path: str) -> bool:
    """
    Validate that a file is a valid DICOM file and contains pixel data.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        True if valid DICOM with pixel data, raises DicomValidationError otherwise
    """
    if not PYDICOM_AVAILABLE:
        raise DicomValidationError("pydicom is not installed")
    
    file_path = Path(file_path)
    
    # Check file exists
    if not file_path.exists():
        raise DicomValidationError(f"File not found: {file_path}")
    
    # Check file size (reasonable bounds)
    file_size = file_path.stat().st_size
    if file_size < 100:  # Too small to be valid DICOM
        raise DicomValidationError(f"File too small: {file_size} bytes")
    if file_size > 1024 * 1024 * 500:  # 500 MB upper limit
        raise DicomValidationError(f"File too large: {file_size / (1024*1024):.1f} MB")
    
    try:
        # Try to read DICOM
        ds = pydicom.dcmread(str(file_path))
        
        # Check for pixel data
        if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
            raise DicomValidationError("DICOM file has no pixel data")
        
        # Verify pixel array has valid shape
        pixels = ds.pixel_array
        if pixels.size == 0:
            raise DicomValidationError("Pixel array is empty")
        
        return True
        
    except pydicom.errors.InvalidDicomError as e:
        raise DicomValidationError(f"Invalid DICOM file: {str(e)}")
    except AttributeError as e:
        raise DicomValidationError(f"DICOM missing pixel data: {str(e)}")
    except Exception as e:
        raise DicomValidationError(f"Error reading DICOM: {str(e)}")


def get_dicom_metadata(file_path: str) -> Dict[str, any]:
    """
    Extract key metadata from DICOM file.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        Dictionary with modality, dimensions, bit-depth, patient ID (hashed), etc.
    """
    if not PYDICOM_AVAILABLE:
        raise DicomValidationError("pydicom is not installed")
    
    try:
        ds = pydicom.dcmread(str(file_path))
        
        # Get pixel array info
        pixels = ds.pixel_array
        height, width = pixels.shape[-2:] if pixels.ndim >= 2 else (0, 0)
        bit_depth = pixels.dtype.itemsize * 8
        
        # Get modality
        modality = ds.Modality if hasattr(ds, 'Modality') else 'UNKNOWN'
        
        # Get study/patient info
        patient_id = ds.PatientID if hasattr(ds, 'PatientID') else 'UNKNOWN'
        study_date = ds.StudyDate if hasattr(ds, 'StudyDate') else 'UNKNOWN'
        study_time = ds.StudyTime if hasattr(ds, 'StudyTime') else 'UNKNOWN'
        
        # Get series/instance info
        series_number = getattr(ds, 'SeriesNumber', None)
        instance_number = getattr(ds, 'InstanceNumber', None)
        
        metadata = {
            'modality': str(modality),
            'width': int(width),
            'height': int(height),
            'bit_depth': int(bit_depth),
            'patient_id': str(patient_id),
            'study_date': str(study_date),
            'study_time': str(study_time),
            'series_number': series_number,
            'instance_number': instance_number,
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting DICOM metadata: {str(e)}")
        raise DicomValidationError(f"Failed to extract metadata: {str(e)}")


def get_modality(file_path: str) -> str:
    """Get the modality of a DICOM file."""
    try:
        ds = pydicom.dcmread(str(file_path))
        return str(ds.Modality) if hasattr(ds, 'Modality') else 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


def get_windowing_params(file_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get windowing parameters (center, width) for a DICOM file.
    
    Tries to read from DICOM tags first, falls back to modality-based defaults.
    
    Returns:
        Tuple of (window_center, window_width) or (None, None) if not applicable
    """
    try:
        ds = pydicom.dcmread(str(file_path))
        
        # Try to read from DICOM
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = float(ds.WindowCenter) if ds.WindowCenter else None
            width = float(ds.WindowWidth) if ds.WindowWidth else None
            if center is not None and width is not None:
                return (center, width)
        
        # Fall back to modality-based defaults
        modality = str(ds.Modality) if hasattr(ds, 'Modality') else 'UNKNOWN'
        return MODALITY_WINDOWS.get(modality, (None, None))
        
    except Exception as e:
        logger.warning(f"Could not get windowing params: {str(e)}")
        return (None, None)


def load_dicom_pixels(file_path: str) -> np.ndarray:
    """
    Load pixel array from DICOM file with validation.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        Pixel array as numpy float32 array
    """
    if not PYDICOM_AVAILABLE:
        raise DicomValidationError("pydicom is not installed")
    
    try:
        validate_dicom_file(file_path)
        ds = pydicom.dcmread(str(file_path))
        pixels = ds.pixel_array.astype(np.float32)
        return pixels
    except Exception as e:
        logger.error(f"Error loading DICOM pixels: {str(e)}")
        raise DicomValidationError(f"Could not load pixels: {str(e)}")


def get_pixel_value_range(file_path: str) -> Tuple[float, float]:
    """
    Get the min/max pixel values in a DICOM file.
    
    Returns:
        Tuple of (min_value, max_value)
    """
    try:
        pixels = load_dicom_pixels(file_path)
        return (float(pixels.min()), float(pixels.max()))
    except Exception as e:
        logger.error(f"Error getting pixel range: {str(e)}")
        raise DicomValidationError(f"Could not determine pixel range: {str(e)}")
