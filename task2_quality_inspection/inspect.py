"""
Quality Inspection Script
Analyzes images for defects, localizes them, and provides severity assessment
"""

import torch
import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
import time

from models.defect_detector import DefectDetector
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Defect type definitions
DEFECT_TYPES = {
    1: "scratch",
    2: "misalignment",
    3: "missing_component",
    4: "discoloration"
}

DEFECT_COLORS = {
    1: (0, 0, 255),      # Red for scratches
    2: (255, 0, 0),      # Blue for misalignment
    3: (0, 255, 255),    # Yellow for missing components
    4: (255, 0, 255)     # Magenta for discoloration
}


def load_model(checkpoint_path, num_defect_types, device):
    """Load trained defect detection model"""
    model = DefectDetector(num_defect_types=num_defect_types).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image, device):
    """Preprocess image for inference"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    return image_tensor


def calculate_severity(box, score, image_size):
    """Calculate defect severity based on size and confidence"""
    # Extract box dimensions
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Normalize area
    image_area = image_size[0] * image_size[1]
    relative_area = area / image_area
    
    # Severity calculation
    # Based on: area (40%), confidence (40%), position (20% if near center)
    area_score = min(relative_area * 10, 1.0)  # Normalize to [0, 1]
    confidence_score = score
    
    # Position penalty (defects near center are more critical)
    center_x, center_y = image_size[1] / 2, image_size[0] / 2
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    distance = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    position_score = 1.0 - (distance / max_distance) * 0.2  # 20% weight
    
    severity = (area_score * 0.4 + confidence_score * 0.4 + position_score * 0.2)
    
    # Categorize severity
    if severity >= 0.7:
        severity_level = "High"
    elif severity >= 0.4:
        severity_level = "Medium"
    else:
        severity_level = "Low"
    
    return severity_level, severity


def analyze_image(model, image_path, device, output_dir=None):
    """Analyze an image for defects"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    h, w = image.shape[:2]
    
    # Preprocess
    image_tensor = preprocess_image(image_rgb, device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        boxes_list, scores_list, labels_list = model(image_tensor)
    inference_time = time.time() - start_time
    
    # Process results
    if len(boxes_list) > 0:
        boxes = boxes_list[0].cpu().numpy()
        scores = scores_list[0].cpu().numpy()
        labels = labels_list[0].cpu().numpy()
    else:
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])
    
    # Prepare results
    results = {
        "image_path": str(image_path),
        "image_size": [h, w],
        "inference_time_ms": inference_time * 1000,
        "num_defects": len(boxes),
        "defects": []
    }
    
    # Process each detection
    defect_centers = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        
        # Calculate center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Calculate severity
        severity_level, severity_score = calculate_severity(box, score, (h, w))
        
        # Store defect information
        defect_info = {
            "defect_id": i + 1,
            "defect_type": DEFECT_TYPES.get(label, f"unknown_{label}"),
            "confidence_score": float(score),
            "bounding_box": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            },
            "center_coordinates": {
                "x": center_x,
                "y": center_y
            },
            "severity": {
                "level": severity_level,
                "score": float(severity_score)
            }
        }
        
        results["defects"].append(defect_info)
        defect_centers.append((center_x, center_y))
        
        # Draw on image
        color = DEFECT_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw center point
        cv2.circle(original_image, (center_x, center_y), 5, color, -1)
        
        # Draw label
        label_text = f"{DEFECT_TYPES.get(label, 'Unknown')}: {score:.2f} ({severity_level})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(original_image, (int(x1), int(y1) - text_height - 10),
                     (int(x1) + text_width, int(y1)), color, -1)
        cv2.putText(original_image, label_text, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add summary text
    summary_text = f"Defects: {len(boxes)} | Time: {inference_time*1000:.1f}ms"
    cv2.putText(original_image, summary_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save annotated image
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(output_path), original_image)
        results["annotated_image_path"] = str(output_path)
    
    return results, original_image


def main():
    parser = argparse.ArgumentParser(description='Quality Inspection System')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Path to output directory')
    parser.add_argument('--num_defect_types', type=int, default=4,
                       help='Number of defect types')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model}...')
    model = load_model(args.model, args.num_defect_types, device)
    print('Model loaded successfully!')
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f'Analyzing image: {input_path}')
        results, annotated_image = analyze_image(model, str(input_path), device, args.output)
        
        # Save results
        results_file = Path(args.output) / f"results_{input_path.stem}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nResults saved to {results_file}')
        print(f'Found {results["num_defects"]} defects')
        
        # Print defect details
        for defect in results["defects"]:
            print(f"\nDefect {defect['defect_id']}:")
            print(f"  Type: {defect['defect_type']}")
            print(f"  Confidence: {defect['confidence_score']:.3f}")
            print(f"  Center: ({defect['center_coordinates']['x']}, {defect['center_coordinates']['y']})")
            print(f"  Severity: {defect['severity']['level']} ({defect['severity']['score']:.3f})")
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f'Processing {len(image_files)} images...')
        
        all_results = []
        for img_file in image_files:
            print(f'\nProcessing: {img_file.name}')
            try:
                results, _ = analyze_image(model, str(img_file), device, args.output)
                all_results.append(results)
            except Exception as e:
                print(f'Error processing {img_file}: {e}')
        
        # Save combined results
        results_file = Path(args.output) / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f'\nProcessed {len(all_results)} images')
        print(f'Results saved to {results_file}')
        
        # Summary statistics
        total_defects = sum(r["num_defects"] for r in all_results)
        avg_inference_time = np.mean([r["inference_time_ms"] for r in all_results])
        print(f'\nSummary:')
        print(f'  Total defects found: {total_defects}')
        print(f'  Average inference time: {avg_inference_time:.2f} ms')
    
    else:
        print(f'Error: Input path {args.input} does not exist')


if __name__ == '__main__':
    main()



