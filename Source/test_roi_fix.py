"""Test the fixed watermarking with proper ROI mask."""

import requests
import base64
import numpy as np
from PIL import Image
import io
import time

API_URL = "http://localhost:5000"
SAMPLE_DICOM = "../dataset/siim-medical-images/versions/6/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"

print("Testing Fixed Watermarking (ROI-Aware)")
print("=" * 60)

start = time.time()

# Test the embed endpoint
print("Embedding watermark (with proper ROI masking)...")
with open(SAMPLE_DICOM, 'rb') as f:
    response = requests.post(
        f"{API_URL}/embed",
        files={'file': f},
        data={
            'patient_id': 'TEST_ROI_FIX',
            'strength': 1.0,
            'return_format': 'json'
        },
        timeout=120  # Longer timeout because we're doing clustering now
    )

elapsed = time.time() - start

if response.status_code != 200:
    print(f"✗ Error: {response.json()}")
    exit(1)

data = response.json()
print(f"✓ Watermark Embedding Success ({elapsed:.1f}s)")
print(f"  Payload: {data['watermark_payload'][:32]}...")
print(f"  PSNR: {data['embedding']['psnr_db']} dB")
print(f"  Processing time: {data['embedding']['processing_time_seconds']:.2f}s")

# Decode the watermarked image
img_data = data['watermarked_image_preview'].split(',')[1]
img_bytes = base64.b64decode(img_data)
watermarked_img = Image.open(io.BytesIO(img_bytes))

# Analyze the image
arr = np.array(watermarked_img)
print(f"\n✓ Watermarked Image Analysis:")
print(f"  Size: {watermarked_img.size}")
print(f"  Mode: {watermarked_img.mode}")
print(f"  Value range: {arr.min()} - {arr.max()}")
print(f"  Mean pixel value: {arr.mean():.1f}")  
print(f"  Std deviation: {arr.std():.1f}")

# Check if ROI is preserved
roi_like_bright = (arr > 200).sum()
roi_like_ratio = roi_like_bright / arr.size
print(f"\n✓ ROI Preservation Check:")
print(f"  Bright pixels (>200): {roi_like_ratio*100:.1f}%")

if roi_like_ratio < 0.3:
    print(f"  ✓ ROI looks properly preserved (not all watermarked)")
    watermarked_img.save("watermarked_roi_fixed.png")
    print(f"  Saved to: watermarked_roi_fixed.png")
elif roi_like_ratio > 0.6:
    print(f"  ✗ High bright pixel ratio - ROI may have been watermarked")
    watermarked_img.save("watermarked_roi_issue.png")
else:
    print(f"  ⚠ Uncertain - check visual appearance")
    watermarked_img.save("watermarked_roi_uncertain.png")

print("\n" + "=" * 60)
