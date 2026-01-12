"""
Training script for defect detection model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from tqdm import tqdm

from models.defect_detector import DefectDetector
from utils.dataset import DefectDataset, collate_fn


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    cls_loss_total = 0.0
    loc_loss_total = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        # Classification loss
        cls_targets = []
        for target in targets:
            if len(target['labels']) > 0:
                # Use most common defect type in image, or 0 for no defect
                cls_targets.append(target['labels'][0].item() + 1)  # +1 for background
            else:
                cls_targets.append(0)  # No defect
        
        cls_targets = torch.tensor(cls_targets, dtype=torch.long, device=device)
        cls_loss = nn.CrossEntropyLoss()(outputs['cls_logits'], cls_targets)
        
        # Localization loss (simplified)
        loc_loss = torch.tensor(0.0, device=device)
        if len(targets) > 0 and len(targets[0]['boxes']) > 0:
            # Simplified localization loss
            # In practice, would match predictions to ground truth
            pass
        
        # Total loss
        loss = cls_loss + 0.1 * loc_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        cls_loss_total += cls_loss.item()
        loc_loss_total += loc_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'loc': f'{loc_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    return {
        'loss': avg_loss,
        'cls_loss': cls_loss_total / len(dataloader),
        'loc_loss': loc_loss_total / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Defect Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Create dataset
    train_dataset = DefectDataset(
        image_dir=config['train_image_dir'],
        annotations_file=config['train_annotations'],
        is_training=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    num_defect_types = config.get('num_defect_types', 4)
    model = DefectDetector(num_defect_types=num_defect_types).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.1)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # TensorBoard writer
    writer = SummaryWriter(config['log_dir'])
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        
        # Train
        metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log metrics
        writer.add_scalar('Train/Loss', metrics['loss'], epoch)
        writer.add_scalar('Train/Cls_Loss', metrics['cls_loss'], epoch)
        writer.add_scalar('Train/Loc_Loss', metrics['loc_loss'], epoch)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
        }
        
        # Save latest
        torch.save(checkpoint, 
                  os.path.join(config['checkpoint_dir'], 'latest.pth'))
        
        # Save best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(checkpoint,
                      os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f'New best model saved with loss: {best_loss:.4f}')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()

