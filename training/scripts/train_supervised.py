#!/usr/bin/env python3
"""
Train policy-value network on PGN dataset (supervised learning).
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from datasets import PGNDataset
from utils import (
    load_config,
    setup_logger,
    log_metrics,
    save_checkpoint,
    cleanup_old_checkpoints,
    AverageMeter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised training on PGN dataset')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Path to training config')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to processed dataset NPZ file')
    parser.add_argument('--output-dir', type=str, default='weights/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    
    policy_loss_meter = AverageMeter('policy_loss')
    value_loss_meter = AverageMeter('value_loss')
    total_loss_meter = AverageMeter('total_loss')
    
    for batch_idx, (planes, move_indices, results) in enumerate(dataloader):
        planes = planes.to(device)
        move_indices = move_indices.to(device)
        results = results.to(device)
        
        # Forward pass
        policy_logits, value_pred = model(planes)
        
        # Policy loss (cross-entropy)
        policy_loss = nn.functional.cross_entropy(policy_logits, move_indices)
        
        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), results)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update meters
        policy_loss_meter.update(policy_loss.item())
        value_loss_meter.update(value_loss.item())
        total_loss_meter.update(total_loss.item())
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                       f"policy_loss={policy_loss.item():.4f}, "
                       f"value_loss={value_loss.item():.4f}, "
                       f"total_loss={total_loss.item():.4f}")
    
    return {
        'policy_loss': policy_loss_meter.avg,
        'value_loss': value_loss_meter.avg,
        'total_loss': total_loss_meter.avg
    }


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    
    policy_loss_meter = AverageMeter()
    value_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    
    with torch.no_grad():
        for planes, move_indices, results in dataloader:
            planes = planes.to(device)
            move_indices = move_indices.to(device)
            results = results.to(device)
            
            policy_logits, value_pred = model(planes)
            
            policy_loss = nn.functional.cross_entropy(policy_logits, move_indices)
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), results)
            total_loss = policy_loss + value_loss
            
            policy_loss_meter.update(policy_loss.item())
            value_loss_meter.update(value_loss.item())
            total_loss_meter.update(total_loss.item())
    
    return {
        'policy_loss': policy_loss_meter.avg,
        'value_loss': value_loss_meter.avg,
        'total_loss': total_loss_meter.avg
    }


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    train_config = config['training']['supervised']
    model_config = config['model']
    
    # Setup logger
    logger = setup_logger('train_supervised')
    logger.info(f"Training configuration: {train_config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = PGNDataset(args.dataset)
    
    # Split train/val
    train_size = int(train_config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    logger.info("Creating model")
    model = create_model(model_config)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, train_config['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{train_config['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, logger)
        log_metrics(logger, train_metrics, epoch=epoch + 1)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        log_metrics(logger, {f"val_{k}": v for k, v in val_metrics.items()}, epoch=epoch + 1)
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
        
        checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        save_checkpoint(
            model, optimizer, epoch + 1,
            {'train': train_metrics, 'val': val_metrics},
            checkpoint_path,
            is_best=is_best
        )
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(args.output_dir, keep_last_n=5)
    
    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()
