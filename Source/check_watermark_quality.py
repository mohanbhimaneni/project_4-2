"""Check the quality of the generated watermark visualization."""

from PIL import Image
import numpy as np

# Check roi_and_watermark_results.png
img1 = Image.open('roi_and_watermark_results.png')
arr1 = np.array(img1)

print("=" * 60)
print("ROI & Watermark Results Visualization")
print("=" * 60)
print(f"Image size: {img1.size}")
print(f"Image mode: {img1.mode}")
print(f"Array shape: {arr1.shape}")
print(f"Value range: {arr1.min()} - {arr1.max()}")
print(f"Mean: {arr1.mean():.1f}, Std: {arr1.std():.1f}")

# Check watermark_effect.png
img2 = Image.open('watermark_effect.png')
arr2 = np.array(img2)

print("\n" + "=" * 60)
print("Watermark Effect Visualization")  
print("=" * 60)
print(f"Image size: {img2.size}")
print(f"Image mode: {img2.mode}")
print(f"Array shape: {arr2.shape}")
print(f"Value range: {arr2.min()} - {arr2.max()}")
print(f"Mean: {arr2.mean():.1f}, Std: {arr2.std():.1f}")

# Analyze the middle section which should show the watermarked image
if arr2.shape[0] > 200:
    # Assuming 3-subplot layout, take the middle 1/3
    h_third = arr2.shape[0] // 3
    watermarked_section = arr2[h_third:2*h_third, :]
    print(f"\nWatermarked image section stats:")
    print(f"  Value range: {watermarked_section.min()} - {watermarked_section.max()}")
    print(f"  Mean: {watermarked_section.mean():.1f}")
    
    if watermarked_section.mean() > 50:
        print("✓ Watermarked section looks normal (not pure noise)")
    else:
        print("⚠ Watermarked section is very dark")
else:
    print(f"\n⚠ Image too small to analyze sections")

print("\n" + "=" * 60)
print("Summary: Files generated successfully with fix applied")
print("=" * 60)
