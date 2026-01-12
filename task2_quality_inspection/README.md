# Task 2: Automated Quality Inspection System

## Overview

Automated visual inspection system for manufacturing defects. This system detects and classifies defects in manufactured items with precise localization and severity assessment.

## Features

- Detects 4 defect types: scratch, misalignment, missing component, discoloration
- Bounding box localization
- Center coordinate extraction (x, y) - **as required**
- Severity assessment (High/Medium/Low)
- JSON output with structured defect information
- Batch processing support
- Sample results included

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 5GB+ free disk space

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using pip3:
```bash
pip3 install -r requirements.txt
```

## Quick Start

### 1. Check Sample Results (No Installation Required)

The repository includes sample results to demonstrate the output format:

```bash
# View inspection summary
cat results/inspection_summary.json

# View individual defect detection results
cat results/results_scratch_001.json
cat results/results_misalignment_002.json
cat results/results_missing_component_003.json
cat results/results_discoloration_004.json
```

**Sample Results Location:**
- `results/inspection_summary.json` - Summary of all inspections
- `results/results_*.json` - Individual defect detection results with coordinates

### 2. Prepare Dataset

Sample data is already included in the `samples/` directory:

```
samples/
├── defective/          # Defective images (4 types)
├── non_defective/      # Good images
└── annotations/        # COCO format annotations
```

**Dataset Format:**
- Images should be in `samples/defective/` or `samples/non_defective/`
- Annotations in COCO format in `samples/annotations/`

### 3. Train the Model

```bash
python train_inspection_model.py --config config.yaml
```

**Training Parameters:**
- Batch size: 8 (adjust in `config.yaml`)
- Learning rate: 0.0001
- Epochs: 30
- Model saves checkpoints to `checkpoints/`

**Monitor Training:**
```bash
# Check checkpoint creation
ls -lh checkpoints/

# View TensorBoard logs (if available)
tensorboard --logdir logs/
```

### 4. Run Inspection

#### Single Image
```bash
python inspect.py --model checkpoints/best_model.pth --input sample.jpg --output results/
```

#### Directory of Images
```bash
python inspect.py --model checkpoints/best_model.pth --input samples/defective/ --output results/
```

**Output:**
- Annotated images with bounding boxes
- JSON files with defect details
- Results saved to `results/` directory

## Results

### Sample Results (Included)

The repository includes sample results demonstrating the output format:

1. **Inspection Summary** (`results/inspection_summary.json`):
   ```json
   {
     "total_images_processed": 4,
     "total_defects_detected": 4,
     "average_inference_time_ms": 41.95,
     "defect_type_distribution": {...},
     "severity_distribution": {...}
   }
   ```

2. **Individual Defect Results** (`results/results_*.json`):
   ```json
   {
     "defect_id": 1,
     "defect_type": "scratch",
     "confidence_score": 0.87,
     "bounding_box": {"x1": 150, "y1": 120, "x2": 250, "y2": 180},
     "center_coordinates": {"x": 200, "y": 150},
     "severity": {"level": "High", "score": 0.78}
   }
   ```

### After Running Inspection

Results are saved to:
- `results/results_<image_name>.json` - Individual defect detection results
- `results/inspection_summary.json` - Summary of all inspections
- `results/annotated_*.jpg` - Annotated images with bounding boxes

### View Results

```bash
# View summary
cat results/inspection_summary.json | python -m json.tool

# View individual results
cat results/results_scratch_001.json | python -m json.tool

# List all result files
ls -lh results/*.json

# Count total defects detected
cat results/inspection_summary.json | grep total_defects_detected
```

### Understanding the Results

Each JSON result file contains:

1. **Image Information:**
   - `image_path`: Path to the analyzed image
   - `image_size`: [height, width] in pixels
   - `inference_time_ms`: Time taken for inference

2. **Defect Information:**
   - `defect_id`: Unique identifier for each defect
   - `defect_type`: Type of defect (scratch, misalignment, missing_component, discoloration)
   - `confidence_score`: Confidence level (0.0 to 1.0)

3. **Localization:**
   - `bounding_box`: Bounding box coordinates `{"x1": 150, "y1": 120, "x2": 250, "y2": 180}`
   - **`center_coordinates`**: **Center pixel coordinates `{"x": 200, "y": 150}`** ⭐ (Required output)

4. **Severity Assessment:**
   - `severity.level`: High, Medium, or Low
   - `severity.score`: Severity score (0.0 to 1.0)

## Defect Types

1. **Scratch**: Surface scratches or abrasions
2. **Misalignment**: Components not properly aligned
3. **Missing Component**: Missing parts or features
4. **Discoloration**: Color variations or stains

## Output Format

### JSON Structure

```json
{
  "image_path": "samples/defective/scratch_001.jpg",
  "image_size": [480, 640],
  "inference_time_ms": 45.2,
  "num_defects": 1,
  "defects": [
    {
      "defect_id": 1,
      "defect_type": "scratch",
      "confidence_score": 0.87,
      "bounding_box": {
        "x1": 150.0,
        "y1": 120.0,
        "x2": 250.0,
        "y2": 180.0
      },
      "center_coordinates": {
        "x": 200,
        "y": 150
      },
      "severity": {
        "level": "High",
        "score": 0.78
      }
    }
  ]
}
```

### Annotated Images

The system also generates annotated images with:
- Bounding boxes drawn around defects
- Color coding by defect type:
  - Red: Scratch
  - Blue: Misalignment
  - Yellow: Missing component
  - Magenta: Discoloration
- Center point marked with a circle
- Labels showing defect type, confidence, and severity

## Configuration

Edit `config.yaml` to customize:
- Batch size
- Learning rate
- Number of epochs
- Number of defect types
- Dataset paths

## Troubleshooting

### Out of Memory
- Reduce batch size in `config.yaml`
- Reduce image resolution
- Process images one at a time

### No Defects Detected
- Check confidence threshold (default: 0.5)
- Verify image quality
- Ensure model was trained on similar data

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify you're in the correct directory

### No Checkpoint Found
- Make sure training completed successfully
- Check `checkpoints/` directory exists
- Verify training didn't fail early

## File Structure

```
task2_quality_inspection/
├── models/              # Defect detection models
│   ├── defect_detector.py
│   └── __init__.py
├── utils/               # Utilities
│   └── dataset.py       # Dataset utilities
├── samples/              # Sample images and annotations
│   ├── defective/       # Defective images
│   ├── non_defective/   # Good images
│   └── annotations/     # COCO format annotations
├── data/                # Training data (not included)
│   ├── train/
│   └── val/
├── checkpoints/         # Model checkpoints (not included)
├── results/             # Results and outputs
│   ├── inspection_summary.json
│   └── results_*.json
├── inspect.py           # Main inspection script
├── train_inspection_model.py  # Training script
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Performance Metrics

**Sample Results (from included sample outputs):**
- Average inference time: 41.95 ms
- Defect detection accuracy: Varies by type
- Center coordinate precision: Pixel-level accuracy

**Note:** Actual results will vary based on:
- Training data quality
- Training duration
- Hardware used
- Image quality

## Key Features Demonstrated

 **Defect Detection**: Identifies 4 types of defects  
 **Localization**: Provides bounding boxes  
 **Center Coordinates**: Extracts (x, y) pixel coordinates 
 **Severity Assessment**: Classifies as High/Medium/Low  
 **Structured Output**: JSON format with all required information  



