"""
Test script for DICOM ROI Extraction API.

Requires the API server to be running:
  python api.py

Usage:
  python test_api.py --dicom <path_to_dicom> --api-url http://localhost:5000
"""

import argparse
import sys
from pathlib import Path
import time

try:
    import requests
except ImportError:
    print("ERROR: requests module not found. Install with: pip install requests")
    sys.exit(1)


def test_health(api_url):
    """Test the health check endpoint."""
    print(f"[TEST] Health check at {api_url}/health")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✓ Health check successful")
            print(f"  Status: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_roi_process(api_url, dicom_path, output_path="test_output.png", alpha=0.5):
    """Test the ROI processing endpoint."""
    print(f"\n[TEST] Processing DICOM at {api_url}/roi/process")
    print(f"  Input DICOM: {dicom_path}")
    print(f"  Output PNG: {output_path}")
    print(f"  Alpha: {alpha}")
    
    dicom_file = Path(dicom_path)
    if not dicom_file.exists():
        print(f"✗ DICOM file not found: {dicom_path}")
        return False
    
    try:
        start_time = time.time()
        
        with open(dicom_file, "rb") as f:
            files = {"file": f}
            data = {"alpha": alpha}
            response = requests.post(
                f"{api_url}/roi/process",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout
            )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            # Save PNG
            output_file = Path(output_path)
            output_file.write_bytes(response.content)
            print(f"✓ ROI processing successful in {elapsed:.2f} seconds")
            print(f"  Output size: {len(response.content)} bytes")
            print(f"  Saved to: {output_file.absolute()}")
            return True
        else:
            print(f"✗ ROI processing failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data}")
            except:
                print(f"  Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ ROI processing error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test DICOM ROI Extraction API",
        epilog="Example: python test_api.py --dicom /path/to/file.dcm"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--dicom",
        required=True,
        help="Path to DICOM file for testing"
    )
    parser.add_argument(
        "--output",
        default="test_output.png",
        help="Output PNG path (default: test_output.png)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DICOM ROI Extraction API - Test Suite")
    print("=" * 60)
    
    # Test health
    health_ok = test_health(args.api_url)
    if not health_ok:
        print("\n[ABORT] API health check failed. Is the server running?")
        print(f"  Expected URL: {args.api_url}/health")
        return 1
    
    # Test ROI processing
    roi_ok = test_roi_process(
        args.api_url,
        args.dicom,
        args.output,
        args.alpha
    )
    
    print("\n" + "=" * 60)
    if health_ok and roi_ok:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
