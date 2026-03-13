"""Direct test of the fixed watermark API."""

import requests
import base64
import numpy as np
from PIL import Image
import io

API_URL = "http://localhost:5000"
SAMPLE_DICOM = "../dataset/siim-medical-images/versions/6/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"

print("Testing Fixed Watermark API")
print("=" * 60)

# Test the embed endpoint
print("Embedding watermark in DICOM...")
with open(SAMPLE_DICOM, 'rb') as f:
    response = requests.post(
        f"{API_URL}/embed",
        files={'file': f},
        data={
            'patient_id': 'TEST_PATIENT',
            'strength': 1.0, 
            'return_format': 'json'
        },
        timeout=60
    )

if response.status_code != 200:
    print(f"✗ Error: {response.json()}")
    exit(1)

data = response.json()
print(f"✓ Watermark Embedding Success")
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
print(f"  Array shape: {arr.shape}")
print(f"  Value range: {arr.min()} - {arr.max()}")
print(f"  Mean pixel value: {arr.mean():.1f}")
print(f"  Std deviation: {arr.std():.1f}")

# Check if it looks reasonable
if arr.mean() > 40 and arr.std() > 20:
    print(f"\n✓ WATERMARK FIX SUCCESSFUL - Image looks normal!")
    print(f"✓ The watermarked image is now imperceptible")
    watermarked_img.save("watermarked_test_fixed.png")
    print(f"  Saved to: watermarked_test_fixed.png")
else:
    print(f"\n⚠ Possible issue: Image stats seem unusual")
    print(f"  Mean: {arr.mean():.1f} (expected > 40)")
    print(f"  Std: {arr.std():.1f} (expected > 20)")
    watermarked_img.save("watermarked_test_issue.png")
    print(f"  Saved to: watermarked_test_issue.png for inspection")

print("\n" + "=" * 60)
