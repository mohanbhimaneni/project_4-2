"""
Watermark Robustness Testing

Apply various attacks and test watermark extraction:
- JPEG compression (quality factors: 50, 75, 90, 95)
- Gaussian noise (σ: 1, 3, 5, 10)
- Scaling (factors: 0.8, 0.9, 1.0, 1.1, 1.2)
- Rotation (angles: 1°, 5°, 10°)
- Histogram equalization
"""

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from pathlib import Path
import cv2
from scipy import ndimage

API_URL = "http://localhost:5000"
SAMPLE_DICOM = Path("../dataset/siim-medical-images/versions/6/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm")

def embed_watermark(dcm_path):
    """Embed watermark in DICOM and return watermarked image + payload."""
    with open(dcm_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/embed",
            files={'file': f},
            data={
                'patient_id': 'ROBUSTNESS_TEST',
                'strength': 1.0,
                'return_format': 'json'
            },
            timeout=30
        )
    
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.json()}")
    
    data = response.json()
    payload = data['watermark_payload']
    
    # Decode image
    img_data = data['watermarked_image_preview'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    image = Image.open(io.BytesIO(img_bytes))
    
    return np.array(image), payload

def apply_jpeg_compression(image, quality):
    """Apply JPEG compression at given quality factor."""
    img = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def apply_gaussian_noise(image, sigma):
    """Apply Gaussian noise."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(float) + noise, 0, 255)
    return noisy.astype(np.uint8)

def apply_scaling(image, factor):
    """Scale image."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(image, (new_w, new_h))
    
    # Pad back to original size if scaled down
    if scaled.shape[0] < h or scaled.shape[1] < w:
        padded = np.zeros_like(image)
        padded[:scaled.shape[0], :scaled.shape[1]] = scaled[:h, :w]
        return padded
    return scaled[:h, :w]

def apply_rotation(image, angle):
    """Rotate image."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def apply_histogram_equalization(image):
    """Apply histogram equalization."""
    equalized = cv2.equalizeHist(image.astype(np.uint8))
    return equalized

def save_attacked_image_to_dicom(attacked_image, output_path="temp_attacked.dcm"):
    """Save attacked image as DICOM for verification."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    
    # Create basic DICOM dataset
    ds = FileDataset(
        output_path,
        {},
        file_meta=Dataset(),
        preamble=b"\0" * 128
    )
    
    ds.PatientName = "RobustnessTest"
    ds.PatientID = "12345"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = "1.2.3.4"
    ds.StudyInstanceUID = "1.2.3"
    ds.FrameOfReferenceUID = "1.2.3.5"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.Rows = attacked_image.shape[0]
    ds.Columns = attacked_image.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    
    # Add pixel array
    ds.PixelData = attacked_image.astype(np.uint8).tobytes()
    
    ds.save_as(output_path, write_like_original=False)
    return output_path

def extract_watermark(dicom_path, payload):
    """Extract watermark and get confidence score."""
    with open(dicom_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/verify",
            files={'file': f},
            data={'expected_payload': payload},
            timeout=30
        )
    
    if response.status_code != 200:
        return {'extracted_payload': None, 'confidence': 0.0, 'match': False}
    
    data = response.json()
    return {
        'extracted_payload': data.get('extracted_payload'),
        'confidence': data['extraction']['confidence'],
        'match': data['verification']['match']
    }

def test_robustness():
    """Run robustness tests on various attacks."""
    print("Embedding watermark for robustness testing...")
    original_image, payload = embed_watermark(SAMPLE_DICOM)
    print(f"Watermark payload: {payload[:32]}...")
    
    results = {
        'jpeg': {},
        'noise': {},
        'scaling': {},
        'rotation': {},
        'histogram_eq': {}
    }
    
    # Test 1: JPEG Compression
    print("\n" + "=" * 60)
    print("TEST 1: JPEG COMPRESSION")
    print("=" * 60)
    jpeg_qualities = [50, 75, 90, 95]
    
    for quality in jpeg_qualities:
        print(f"Quality {quality}...", end=" ")
        attacked = apply_jpeg_compression(original_image, quality)
        
        # Save as temporary DICOM
        dcm_path = save_attacked_image_to_dicom(attacked)
        
        # Extract
        result = extract_watermark(dcm_path, payload)
        results['jpeg'][quality] = result['confidence']
        
        print(f"Confidence: {result['confidence']:.3f} {'✓' if result['match'] else '✗'}")
        
        # Cleanup
        Path(dcm_path).unlink()
    
    # Test 2: Gaussian Noise
    print("\n" + "=" * 60)
    print("TEST 2: GAUSSIAN NOISE")
    print("=" * 60)
    noise_sigmas = [1, 3, 5, 10]
    
    for sigma in noise_sigmas:
        print(f"σ = {sigma}...", end=" ")
        attacked = apply_gaussian_noise(original_image, sigma)
        
        dcm_path = save_attacked_image_to_dicom(attacked)
        result = extract_watermark(dcm_path, payload)
        results['noise'][sigma] = result['confidence']
        
        print(f"Confidence: {result['confidence']:.3f} {'✓' if result['match'] else '✗'}")
        
        Path(dcm_path).unlink()
    
    # Test 3: Scaling
    print("\n" + "=" * 60)
    print("TEST 3: SCALING")
    print("=" * 60)
    scale_factors = [0.9, 1.0, 1.1]
    
    for factor in scale_factors:
        print(f"Scale {factor:.1f}x...", end=" ")
        attacked = apply_scaling(original_image, factor)
        
        dcm_path = save_attacked_image_to_dicom(attacked)
        result = extract_watermark(dcm_path, payload)
        results['scaling'][factor] = result['confidence']
        
        print(f"Confidence: {result['confidence']:.3f} {'✓' if result['match'] else '✗'}")
        
        Path(dcm_path).unlink()
    
    # Test 4: Rotation
    print("\n" + "=" * 60)
    print("TEST 4: ROTATION")
    print("=" * 60)
    rotation_angles = [1, 5, 10]
    
    for angle in rotation_angles:
        print(f"Angle {angle}°...", end=" ")
        attacked = apply_rotation(original_image, angle)
        
        dcm_path = save_attacked_image_to_dicom(attacked)
        result = extract_watermark(dcm_path, payload)
        results['rotation'][angle] = result['confidence']
        
        print(f"Confidence: {result['confidence']:.3f} {'✓' if result['match'] else '✗'}")
        
        Path(dcm_path).unlink()
    
    # Test 5: Histogram Equalization
    print("\n" + "=" * 60)
    print("TEST 5: HISTOGRAM EQUALIZATION")
    print("=" * 60)
    
    print("Applying histogram equalization...", end=" ")
    attacked = apply_histogram_equalization(original_image)
    
    dcm_path = save_attacked_image_to_dicom(attacked)
    result = extract_watermark(dcm_path, payload)
    results['histogram_eq'] = result['confidence']
    
    print(f"Confidence: {result['confidence']:.3f} {'✓' if result['match'] else '✗'}")
    
    Path(dcm_path).unlink()
    
    return results

def plot_robustness_results(results):
    """Create visualization of robustness results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # JPEG Compression
    ax = axes[0, 0]
    qualities = sorted(results['jpeg'].keys())
    confidences = [results['jpeg'][q] for q in qualities]
    ax.plot(qualities, confidences, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='r', linestyle='--', label='Expected Threshold')
    ax.set_xlabel('JPEG Quality Factor')
    ax.set_ylabel('Extraction Confidence')
    ax.set_title('Robustness to JPEG Compression')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # Gaussian Noise
    ax = axes[0, 1]
    sigmas = sorted(results['noise'].keys())
    confidences = [results['noise'][s] for s in sigmas]
    ax.plot(sigmas, confidences, 's-', linewidth=2, markersize=8, color='green')
    ax.axhline(y=0.9, color='r', linestyle='--', label='Expected Threshold')
    ax.set_xlabel('Noise Standard Deviation (σ)')
    ax.set_ylabel('Extraction Confidence')
    ax.set_title('Robustness to Gaussian Noise')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # Scaling
    ax = axes[1, 0]
    factors = sorted(results['scaling'].keys())
    confidences = [results['scaling'][f] for f in factors]
    ax.plot(factors, confidences, '^-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=0.9, color='r', linestyle='--', label='Expected Threshold')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Original Size')
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Extraction Confidence')
    ax.set_title('Robustness to Scaling')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # Rotation
    ax = axes[1, 1]
    angles = sorted(results['rotation'].keys())
    confidences = [results['rotation'][a] for a in angles]
    ax.plot(angles, confidences, 'D-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=0.9, color='r', linestyle='--', label='Expected Threshold')
    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Extraction Confidence')
    ax.set_title('Robustness to Rotation')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    plt.suptitle('Watermark Robustness under Various Attacks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('robustness_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved robustness plot to: robustness_results.png")
    plt.show()

def print_summary(results):
    """Print robustness summary."""
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    
    all_confidences = []
    for attack_type, values in results.items():
        if isinstance(values, dict):
            all_confidences.extend(values.values())
        else:
            all_confidences.append(values)
    
    print(f"Average Confidence: {np.mean(all_confidences):.3f}")
    print(f"Min Confidence: {np.min(all_confidences):.3f}")
    print(f"Max Confidence: {np.max(all_confidences):.3f}")
    print(f"Total Attacks Tested: {len(all_confidences)}")
    print(f"Successful Extractions (>0.9): {sum(1 for c in all_confidences if c > 0.9)}/{len(all_confidences)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if not SAMPLE_DICOM.exists():
        print(f"Error: DICOM file not found at {SAMPLE_DICOM}")
        exit(1)
    
    print("=" * 60)
    print("WATERMARK ROBUSTNESS TESTING")
    print("=" * 60)
    
    results = test_robustness()
    print_summary(results)
    plot_robustness_results(results)
    
    print("\n✓ Robustness testing complete!")
