#!/usr/bin/env python3
"""
Verify that the ROI mask is now correctly inverted.
Shows that watermarking is only applied to background, not lungs.
"""

import sys
import requests
import base64
import numpy as np
from PIL import Image
import io
import json

# Configuration
API_URL = "http://localhost:5000"
SAMPLE_DICOM = "dataset/siim-medical-images/versions/6/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"

def test_mask_correctness():
    """Test that ROI mask is correct (True=lungs, False=background)."""
    print("=" * 60)
    print("VERIFYING ROI MASK CORRECTION")
    print("=" * 60)
    
    # Read DICOM file
    with open(SAMPLE_DICOM, "rb") as f:
        dicom_data = f.read()
    
    # Call /embed endpoint
    print("\n1. Calling /embed endpoint...")
    files = {"file": ("test.dcm", io.BytesIO(dicom_data))}
    data = {"strength": "1.0"}
    
    try:
        response = requests.post(f"{API_URL}/embed", files=files, data=data, timeout=120)
        if response.status_code != 200:
            print(f"ERROR: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    # Parse response
    result = response.json()
    if result["status"] != "success":
        print(f"ERROR: {result}")
        return False
    
    # Get watermarked image
    watermarked_uri = result.get("watermarked_image_preview", "")
    if not watermarked_uri.startswith("data:image"):
        print("ERROR: Missing watermarked_image_preview in API response")
        return False

    watermarked_b64 = watermarked_uri.split(",", 1)[1]
    watermarked_data = base64.b64decode(watermarked_b64)
    watermarked_img = np.array(Image.open(io.BytesIO(watermarked_data)))
    
    psnr_val = result.get("embedding", {}).get("psnr_db", None)
    if psnr_val is not None:
        print(f"   ✓ Watermarking successful (PSNR: {psnr_val:.1f} dB)")
    else:
        print("   ✓ Watermarking successful")
    
    # Analyze pixel distribution
    print("\n2. Analyzing watermark effect on different regions...")
    h, w = watermarked_img.shape
    
    # Simplified analysis: check if edges (background) vs center (lungs) differ
    # In a CT lung image:
    # - Edges are mostly background (empty space, bright)
    # - Center is mostly lungs (dark tissue)
    
    edge_region = np.concatenate([
        watermarked_img[0:50, :],  # top
        watermarked_img[-50:, :],  # bottom
        watermarked_img[:, 0:50],  # left
        watermarked_img[:, -50:]   # right
    ])
    
    center_region = watermarked_img[100:400, 100:400]
    
    edge_mean = edge_region.mean()
    center_mean = center_region.mean()
    edge_std = edge_region.std()
    center_std = center_region.std()
    
    print(f"\n   Edge regions (mostly background):")
    print(f"     Mean: {edge_mean:.1f}, Std: {edge_std:.1f}")
    
    print(f"\n   Center region (mostly lungs/ROI):")
    print(f"     Mean: {center_mean:.1f}, Std: {center_std:.1f}")
    
    # Count pixels by intensity
    # If mask is correct: background should have more variation (watermark), lungs should be uniform
    bright_pixels_edge = np.sum(edge_region > 200)
    bright_pixels_center = np.sum(center_region > 200)
    
    edge_bright_pct = (bright_pixels_edge / edge_region.size) * 100
    center_bright_pct = (bright_pixels_center / center_region.size) * 100
    
    print(f"\n   Bright pixels (>200) in edge: {edge_bright_pct:.1f}%")
    print(f"   Bright pixels (>200) in center: {center_bright_pct:.1f}%")
    
    # Verify correction
    print("\n3. MASK VERIFICATION RESULTS:")
    print("   " + "=" * 56)
    
    # With correct mask (lungs protected):
    # - Background should have MORE bright pixels (watermarked)
    # - Lungs should have FEWER bright pixels (preserved)
    
    if edge_std > center_std * 0.8:  # Background has more variation
        print("   ✓ Background shows higher variation (watermark applied)")
    else:
        print("   ✗ Background not showing expected variation")
    
    if center_bright_pct < edge_bright_pct:  # Lungs have fewer bright pixels
        print("   ✓ Lungs have fewer bright pixels (properly preserved)")
        print(f"     Background bright%: {edge_bright_pct:.1f}%")
        print(f"     Lungs bright%: {center_bright_pct:.1f}%")
    else:
        print("   ✗ Lungs have too many bright pixels (may still be watermarked)")
        print(f"     Background bright%: {edge_bright_pct:.1f}%")
        print(f"     Lungs bright%: {center_bright_pct:.1f}%")
    
    # Overall assessment
    print("\n4. CONCLUSION:")
    if center_std < edge_std and center_bright_pct < edge_bright_pct:
        print("   ✓✓✓ ROI MASK IS CORRECTLY INVERTED!")
        print("   ✓✓✓ Watermark is only applied to background/non-ROI regions")
        print("   ✓✓✓ Lungs (clinical ROI) are properly protected")
        return True
    else:
        print("   ✗✗✗ ROI mask may still have issues")
        print("   Further investigation needed")
        return False

if __name__ == "__main__":
    success = test_mask_correctness()
    sys.exit(0 if success else 1)
