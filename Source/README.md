# DICOM ROI Extraction API

A REST API service that processes DICOM medical imaging files and extracts regions of interest (ROI) using a fine-tuned Vision Transformer (ViT) model.

## Features

- **Single DICOM Processing**: Upload a DICOM file via HTTP POST and receive an ROI-overlaid PNG image
- **Fine-tuned ViT Model**: Uses a ViT model fine-tuned on MIDI-B medical imaging dataset (7 modalities)
- **Unsupervised ROI Extraction**: k-means clustering on ViT patch embeddings for automatic ROI detection
- **Model Caching**: Model is loaded once on first request and cached for efficient subsequent calls
- **Flexible Configuration**: Supports CPU/GPU inference, adjustable overlay transparency
- **Health Check Endpoint**: Monitor API status and configuration

## Installation

### 1. Install Dependencies

```bash
cd Source
pip install -r requirements.txt
```

### 2. Verify Checkpoint Exists

Ensure the fine-tuned model checkpoint is present at:
```
project_understanding/checkpoints/midi_b_modality_vit_tiny_final/best.pt
```

## Usage

### Start the API Server

```bash
python api.py
```

**Optional flags:**
- `--host 0.0.0.0` (default) - Bind to all interfaces
- `--port 5000` (default) - API port
- `--debug` - Enable Flask debug mode
- `--device cpu|cuda` (default: cpu) - Compute device
- `--checkpoint PATH` - Path to checkpoint file
- `--model-name MODEL` (default: vit_tiny_patch16_224) - ViT model name

### API Endpoints

#### 1. Health Check

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "DICOM ROI Extraction API",
  "model": "vit_tiny_patch16_224",
  "checkpoint": "/path/to/checkpoint"
}
```

#### 2. Process DICOM File

**Request:**
```bash
curl -F "file=@input.dcm" http://localhost:5000/roi/process -o output.png
```

**Parameters:**
- `file` (required): DICOM file (.dcm or .dicom)
- `alpha` (optional): Overlay transparency (0.0-1.0, default: 0.5)
  - 0.0 = fully transparent (just original DICOM)
  - 1.0 = fully opaque (pure red mask)
- `device` (optional): Compute device (cpu or cuda, default: cpu)

**Response:**
- Success (200): PNG image as binary data
- Error (400/500): JSON error message

**Example with parameters:**
```bash
curl -F "file=@input.dcm" -F "alpha=0.7" http://localhost:5000/roi/process -o output.png
```

#### 3. API Documentation

**Request:**
```bash
curl http://localhost:5000/
```

## Model Details

- **Architecture**: Vision Transformer (ViT) with tiny configuration (`vit_tiny_patch16_224`)
- **Fine-tuning Dataset**: MIDI-B medical imaging dataset (233 samples, 7 modalities)
- **Training Metrics**: 
  - Validation Accuracy: 53.19%
  - Validation F1 (macro): 48.22%
  - Validation AUC: 0.8628
- **ROI Extraction**: Unsupervised k-means clustering on patch-wise embeddings from ViT's intermediate representations

## Performance

- **First inference**: ~28 seconds (includes model loading)
- **Subsequent inferences**: ~1-2 seconds (with cached model)
- **Max file size**: 50 MB
- **Supported input**: DICOM files up to 50 MB

## Example Usage (Python)

```python
import requests
from pathlib import Path

# Process a DICOM file
dicom_file = Path("input.dcm")
response = requests.post(
    "http://localhost:5000/roi/process",
    files={"file": open(dicom_file, "rb")},
    data={"alpha": 0.5}
)

if response.status_code == 200:
    # Save ROI image
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("ROI extraction successful!")
else:
    print(f"Error: {response.json()}")
```

## Troubleshooting

### Model fails to load
- Verify checkpoint path exists: `project_understanding/checkpoints/midi_b_modality_vit_tiny_final/best.pt`
- Check Python path includes `project_understanding` directory
- Ensure all dependencies installed: `pip install -r requirements.txt`

### DICOM file processing fails
- Ensure DICOM file is valid and not corrupted
- Check file size is under 50 MB limit
- Verify DICOM contains pixel data

### Out of memory errors
- Use `--device cpu` for CPU-only inference (slower but uses less memory)
- Reduce worker threads in production deployment

## Deployment Notes

### Development
```bash
python api.py --debug --device cpu
```

### Production
```bash
python api.py --host 0.0.0.0 --port 5000
```

For production use, wrap with a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "api:app"
```

## Architecture

```
DICOM File
    ↓
[Upload via multipart/form-data]
    ↓
API receives and validates file
    ↓
[Load model if not cached]
    ↓
Extract DICOM pixel data
    ↓
Normalize to [0, 1] range
    ↓
Resize to 224×224 and convert to 3-channel
    ↓
Forward through ViT for patch embeddings
    ↓
k-means clustering on patch features
    ↓
Select ROI cluster (highest mean intensity)
    ↓
Morphological post-processing (closing, fill holes)
    ↓
Overlay red mask on original DICOM
    ↓
Return as PNG image
```

## License

This API uses the SecureDICOM project's fine-tuned ViT model. See project documentation for licensing details.
