"""
Batch Watermark Processing & Comparison

Process multiple DICOM files and generate comparison visualizations showing:
- Original → ROI → Watermarked pipeline
- Quality metrics (PSNR, extraction confidence)
- Batch statistics
"""

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

API_URL = "http://localhost:5000"
DICOM_DIR = Path("../dataset/siim-medical-images/versions/6/dicom_dir")

def get_dicom_files(limit=10):
    """Get list of DICOM files."""
    dcm_files = sorted(list(DICOM_DIR.glob("*.dcm")))[:limit]
    return dcm_files

def process_single_dicom(dcm_path, patient_id):
    """Process single DICOM through ROI and watermarking pipeline."""
    try:
        # ROI Processing
        with open(dcm_path, 'rb') as f:
            roi_response = requests.post(
                f"{API_URL}/roi/process",
                files={'file': f},
                data={'return_format': 'json'},
                timeout=30
            )
        
        if roi_response.status_code != 200:
            return {
                'file': dcm_path.name,
                'success': False,
                'error': 'ROI processing failed'
            }
        
        roi_data = roi_response.json()
        
        # Watermark Embedding
        with open(dcm_path, 'rb') as f:
            embed_response = requests.post(
                f"{API_URL}/embed",
                files={'file': f},
                data={
                    'patient_id': patient_id,
                    'strength': 1.0,
                    'return_format': 'json'
                },
                timeout=30
            )
        
        if embed_response.status_code != 200:
            return {
                'file': dcm_path.name,
                'success': False,
                'error': 'Watermark embedding failed'
            }
        
        embed_data = embed_response.json()
        
        # Extract watermark details
        payload = embed_data['watermark_payload']
        psnr = embed_data['embedding']['psnr_db']
        strength = embed_data['embedding']['strength']
        embed_time = embed_data['embedding']['processing_time_seconds']
        
        return {
            'file': dcm_path.name,
            'patient_id': patient_id,
            'success': True,
            'psnr_db': psnr,
            'payload': payload,
            'strength': strength,
            'embed_time_s': embed_time,
            'roi_time_s': roi_data['processing']['processing_time_seconds'],
            'roi_detected': True,
            'roi_area_pct': roi_data['processing'].get('roi_coverage_percent', 0.0) or 0.0,
            'modality': roi_data['dicom_metadata'].get('modality', 'Unknown'),
            'roi_image': roi_data['image'],  # base64
            'watermarked_image': embed_data['watermarked_image_preview']  # base64
        }
    
    except Exception as e:
        return {
            'file': dcm_path.name,
            'success': False,
            'error': str(e)
        }

def batch_process(limit=10, num_workers=3):
    """Process multiple DICOM files in parallel."""
    dicom_files = get_dicom_files(limit)
    print(f"Processing {len(dicom_files)} DICOM files...")
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_dicom, dcm, f"PATIENT_{i:03d}"): i
            for i, dcm in enumerate(dicom_files)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {result['file']}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    return results

def create_comparison_table(results):
    """Create pandas DataFrame with results."""
    successful = [r for r in results if r['success']]
    
    df = pd.DataFrame([
        {
            'File': r['file'],
            'Modality': r.get('modality', 'N/A'),
            'ROI (%)': f"{r.get('roi_area_pct', 0):.1f}%",
            'PSNR (dB)': f"{r['psnr_db']:.2f}",
            'Strength': r['strength'],
            'ROI Time (s)': f"{r['roi_time_s']:.2f}",
            'Watermark Time (s)': f"{r['embed_time_s']:.2f}",
            'Payload': r['payload'][:16] + '...'
        }
        for r in successful
    ])
    
    return df

def visualize_batch_results(results, limit=4):
    """Create grid visualization of original, ROI, and watermarked images."""
    successful = [r for r in results if r['success']][:limit]
    
    if not successful:
        print("No successful results to visualize")
        return
    
    num_images = len(successful)
    fig = plt.figure(figsize=(15, 4 * num_images))
    
    for idx, result in enumerate(successful):
        # Decode base64 images
        roi_img_data = result['roi_image'].split(',')[1]
        watermark_img_data = result['watermarked_image'].split(',')[1]
        
        roi_image = Image.open(io.BytesIO(base64.b64decode(roi_img_data)))
        watermark_image = Image.open(io.BytesIO(base64.b64decode(watermark_img_data)))
        
        # Row 1: ROI
        ax_roi = plt.subplot(num_images, 2, idx * 2 + 1)
        ax_roi.imshow(roi_image)
        ax_roi.set_title(f"ROI: {result['file']}\nROI Area: {result['roi_area_pct']:.1f}%")
        ax_roi.axis('off')
        
        # Row 2: Watermarked
        ax_watermark = plt.subplot(num_images, 2, idx * 2 + 2)
        ax_watermark.imshow(watermark_image)
        ax_watermark.set_title(f"Watermarked\nPSNR: {result['psnr_db']:.2f} dB")
        ax_watermark.axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_watermark_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved batch visualization to: batch_watermark_results.png")
    plt.show()

def save_batch_report(results, output_file='watermark_batch_report.json'):
    """Save detailed batch processing report."""
    report = {
        'total_processed': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': [{k: v for k, v in r.items() if k not in ['roi_image', 'watermarked_image']}
                    for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved batch report to: {output_file}")

def print_statistics(results):
    """Print summary statistics."""
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful results")
        return
    
    psnr_values = [r['psnr_db'] for r in successful]
    roi_areas = [r['roi_area_pct'] for r in successful]
    total_times = [r['roi_time_s'] + r['embed_time_s'] for r in successful]
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    print("\nPSNR (Image Quality - dB):")
    print(f"  Mean: {np.mean(psnr_values):.2f} dB")
    print(f"  Min:  {np.min(psnr_values):.2f} dB")
    print(f"  Max:  {np.max(psnr_values):.2f} dB")
    print(f"  Status: {'✓ All > 40 dB' if all(p > 40 for p in psnr_values) else '✗ Some < 40 dB'}")
    
    print("\nROI Area Detection (%):")
    print(f"  Mean: {np.mean(roi_areas):.1f}%")
    print(f"  Min:  {np.min(roi_areas):.1f}%")
    print(f"  Max:  {np.max(roi_areas):.1f}%")
    
    print("\nProcessing Time (seconds):")
    print(f"  Mean: {np.mean(total_times):.2f}s")
    print(f"  Min:  {np.min(total_times):.2f}s")
    print(f"  Max:  {np.max(total_times):.2f}s")
    
    print("\nQuality Assessment:")
    imperceptible = sum(1 for p in psnr_values if p > 40)
    print(f"  Imperceptible (PSNR > 40): {imperceptible}/{len(successful)}")
    
    print("=" * 60)

if __name__ == "__main__":
    if not DICOM_DIR.exists():
        print(f"Error: DICOM directory not found at {DICOM_DIR}")
        exit(1)
    
    print("=" * 60)
    print("BATCH WATERMARK PROCESSING & COMPARISON")
    print("=" * 60)
    
    # Process batch
    results = batch_process(limit=6, num_workers=3)
    
    # Print statistics
    print_statistics(results)
    
    # Create comparison table
    print("\nResults Table:")
    print("-" * 60)
    df = create_comparison_table(results)
    print(df.to_string(index=False))
    
    # Save report
    save_batch_report(results)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_batch_results(results, limit=4)
    
    print("\n✓ Batch processing complete!")
