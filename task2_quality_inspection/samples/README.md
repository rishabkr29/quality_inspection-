# Sample Images

This directory should contain sample images for quality inspection:

## Directory Structure

```
samples/
├── defective/
│   ├── scratch_001.jpg
│   ├── misalignment_001.jpg
│   ├── missing_component_001.jpg
│   └── discoloration_001.jpg
├── non_defective/
│   ├── good_001.jpg
│   └── good_002.jpg
└── annotations/
    ├── scratch_001.json
    └── ...
```

## Annotation Format

Each image should have a corresponding JSON annotation file in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "scratch_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "area": 30000
    }
  ],
  "categories": [
    {"id": 1, "name": "scratch"},
    {"id": 2, "name": "misalignment"},
    {"id": 3, "name": "missing_component"},
    {"id": 4, "name": "discoloration"}
  ]
}
```

## Defect Types

1. **scratch**: Surface scratches or abrasions
2. **misalignment**: Components not properly aligned
3. **missing_component**: Missing parts or features
4. **discoloration**: Color variations or stains

## Usage

To use these samples for testing:

```bash
python inspect.py --model checkpoints/best_model.pth --input samples/defective/ --output results/
```



