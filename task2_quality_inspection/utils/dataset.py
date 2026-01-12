"""
Dataset utilities for defect detection
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """Dataset for defect detection"""
    
    def __init__(self, image_dir, annotations_file, transform=None, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Default transforms
        if transform is None:
            if is_training:
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.RandomGamma(p=0.2),
                    A.Blur(blur_limit=3, p=0.1),
                    A.GaussNoise(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                                      rotate_limit=10, p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', 
                                           label_fields=['class_labels']))
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', 
                                           label_fields=['class_labels']))
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Get annotations for this image
        image_id = img_info['id']
        boxes = []
        labels = []
        
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2]
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                labels.append(ann['category_id'])
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = []
            labels = []
        
        # Convert to tensors
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            'image_id': torch.tensor([image_id])
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = torch.stack(images, 0)
    return images, targets



