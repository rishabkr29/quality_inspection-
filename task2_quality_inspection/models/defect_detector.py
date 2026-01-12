"""
Defect Detection Model for Quality Inspection
Detects and classifies manufacturing defects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class DefectDetectionBackbone(nn.Module):
    """Backbone for defect detection"""
    
    def __init__(self, in_channels=3):
        super(DefectDetectionBackbone, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DefectClassifier(nn.Module):
    """Defect classification head"""
    
    def __init__(self, in_channels=512, num_defect_types=3):
        super(DefectClassifier, self).__init__()
        self.num_defect_types = num_defect_types
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_defect_types + 1)  # +1 for "no defect"
        )
        
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DefectLocalizer(nn.Module):
    """Defect localization head (bounding box regression)"""
    
    def __init__(self, in_channels=512):
        super(DefectLocalizer, self).__init__()
        
        # Convolutional layers for localization
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bounding box regression
        self.bbox_head = nn.Conv2d(64, 4, kernel_size=1)  # [x, y, w, h] or [x1, y1, x2, y2]
        
        # Objectness score
        self.objectness_head = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        bboxes = self.bbox_head(x)
        objectness = torch.sigmoid(self.objectness_head(x))
        return bboxes, objectness


class DefectDetector(nn.Module):
    """Complete defect detection model"""
    
    def __init__(self, num_defect_types=3, in_channels=3):
        super(DefectDetector, self).__init__()
        self.num_defect_types = num_defect_types
        
        # Backbone
        self.backbone = DefectDetectionBackbone(in_channels=in_channels)
        
        # Classification head (global)
        self.classifier = DefectClassifier(in_channels=512, num_defect_types=num_defect_types)
        
        # Localization head (per-pixel)
        self.localizer = DefectLocalizer(in_channels=512)
        
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone(x)
        
        # Global classification
        cls_logits = self.classifier(features)
        
        # Localization
        bbox_deltas, objectness = self.localizer(features)
        
        if self.training:
            return {
                'cls_logits': cls_logits,
                'bbox_deltas': bbox_deltas,
                'objectness': objectness,
                'features': features if return_features else None
            }
        else:
            # Post-process for inference
            boxes, scores, labels = self._post_process(
                bbox_deltas, objectness, cls_logits, x.shape[-2:]
            )
            return boxes, scores, labels
    
    def _post_process(self, bbox_deltas, objectness, cls_logits, image_size):
        """Post-process model outputs to get final detections"""
        batch_size = bbox_deltas.size(0)
        h, w = bbox_deltas.size(2), bbox_deltas.size(3)
        
        # Get class probabilities
        cls_probs = F.softmax(cls_logits, dim=-1)
        
        # Generate grid coordinates
        y_coords = torch.arange(0, h, device=bbox_deltas.device).float()
        x_coords = torch.arange(0, w, device=bbox_deltas.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Reshape for processing
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4]
        objectness = objectness.permute(0, 2, 3, 1).contiguous().squeeze(-1)  # [B, H, W]
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []
            
            # Filter by objectness threshold
            mask = objectness[b] > 0.5
            if mask.sum() == 0:
                all_boxes.append(torch.empty((0, 4), device=bbox_deltas.device))
                all_scores.append(torch.empty((0,), device=bbox_deltas.device))
                all_labels.append(torch.empty((0,), device=bbox_deltas.device, dtype=torch.long))
                continue
            
            # Get coordinates of positive detections
            y_pos = y_grid[mask]
            x_pos = x_grid[mask]
            obj_scores = objectness[b][mask]
            deltas = bbox_deltas[b][mask]  # [N, 4]
            
            # Convert deltas to absolute coordinates
            # Assuming deltas are [dx, dy, dw, dh] relative to grid cell
            stride = max(image_size) / max(h, w)
            
            cx = (x_pos + 0.5) * stride + deltas[:, 0] * stride
            cy = (y_pos + 0.5) * stride + deltas[:, 1] * stride
            width = torch.exp(deltas[:, 2]) * stride
            height = torch.exp(deltas[:, 3]) * stride
            
            # Convert to [x1, y1, x2, y2]
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
            
            # Clip to image boundaries
            x1 = x1.clamp(0, image_size[1])
            y1 = y1.clamp(0, image_size[0])
            x2 = x2.clamp(0, image_size[1])
            y2 = y2.clamp(0, image_size[0])
            
            boxes_batch = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Get class predictions
            cls_probs_batch = cls_probs[b]  # [num_classes+1]
            # Get max class (excluding background if index 0 is background)
            max_cls_prob, max_cls_idx = cls_probs_batch[1:].max(0)  # Skip background
            max_cls_idx = max_cls_idx + 1  # Adjust for skipped background
            
            # Combine objectness and class probability
            final_scores = obj_scores * max_cls_prob
            
            # Filter by final score
            score_mask = final_scores > 0.3
            if score_mask.sum() > 0:
                boxes_batch = boxes_batch[score_mask]
                final_scores = final_scores[score_mask]
                labels_batch = torch.full((score_mask.sum(),), max_cls_idx.item(), 
                                         device=bbox_deltas.device, dtype=torch.long)
                
                # NMS
                if len(boxes_batch) > 0:
                    keep = nms(boxes_batch, final_scores, iou_threshold=0.5)
                    boxes_batch = boxes_batch[keep]
                    final_scores = final_scores[keep]
                    labels_batch = labels_batch[keep]
            
            all_boxes.append(boxes_batch if len(boxes_batch) > 0 else 
                           torch.empty((0, 4), device=bbox_deltas.device))
            all_scores.append(final_scores if len(final_scores) > 0 else 
                            torch.empty((0,), device=bbox_deltas.device))
            all_labels.append(labels_batch if len(labels_batch) > 0 else 
                            torch.empty((0,), device=bbox_deltas.device, dtype=torch.long))
        
        return all_boxes, all_scores, all_labels



