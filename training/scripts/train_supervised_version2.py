#!/usr/bin/env python3
"""
Train policy-value network on Stockfish-labeled NPZ dataset (grayscale planes).
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import random
from training.models.resnet_policy_value import create_model

  # your existing model creator
from training.utils import (
    setup_logger,
    log_metrics,
    save_checkpoint,
    cleanup_old_checkpoints,
    AverageMeter
)


class NPZDataset(Dataset):
    """
    Load NPZ dataset containing:
      - X: input planes (grayscale/18 planes)
      - y_policy_best: best move index (0..20479)
      - delta_cp / game_result etc. (can be used as value target)
    """
    def __init__(self, npz_path: str, value_target: str = 'delta_cp', value_scale: float = 1.0):
        data = np.load(npz_path)
        self.X = data['X'].astype(np.float32)
        self.y_policy = data['y_policy_best'].astype(np.int64)
        self.value_target = value_target
        self.y_value = data[value_target].astype(np.float32) * value_scale
        self.X = self.X * 2.0 - 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_policy[idx], self.y_value[idx]


def parse_args():
    parser = argparse.ArgumentParser(description='Train policy-value network on NPZ dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to NPZ dataset')
    parser.add_argument('--output-dir', type=str, default='weights/checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--value-target', type=str, default='delta_cp', help='Which NPZ field to use as value')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, logger):
    model.train()
    policy_meter = AverageMeter('policy_loss')
    value_meter = AverageMeter('value_loss')
    total_meter = AverageMeter('total_loss')

    for batch_idx, (planes, move_indices, value_targets) in enumerate(dataloader):
        planes = torch.tensor(planes, device=device)
        move_indices = torch.tensor(move_indices, device=device)
        value_targets = torch.tensor(value_targets, device=device)

        policy_logits, value_pred = model(planes)

        policy_loss = nn.functional.cross_entropy(policy_logits, move_indices)
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        policy_meter.update(policy_loss.item())
        value_meter.update(value_loss.item())
        total_meter.update(total_loss.item())

        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                        f"policy_loss={policy_loss.item():.4f}, "
                        f"value_loss={value_loss.item():.4f}, "
                        f"total_loss={total_loss.item():.4f}")

    return {'policy_loss': policy_meter.avg,
            'value_loss': value_meter.avg,
            'total_loss': total_meter.avg}


def validate(model, dataloader, device):
    model.eval()
    policy_meter = AverageMeter()
    value_meter = AverageMeter()
    total_meter = AverageMeter()

    with torch.no_grad():
        for planes, move_indices, value_targets in dataloader:
            planes = torch.tensor(planes, device=device)
            move_indices = torch.tensor(move_indices, device=device)
            value_targets = torch.tensor(value_targets, device=device)

            policy_logits, value_pred = model(planes)

            policy_loss = nn.functional.cross_entropy(policy_logits, move_indices)
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)
            total_loss = policy_loss + value_loss

            policy_meter.update(policy_loss.item())
            value_meter.update(value_loss.item())
            total_meter.update(total_loss.item())

    return {'policy_loss': policy_meter.avg,
            'value_loss': value_meter.avg,
            'total_loss': total_meter.avg}


def main():
    class Args:
        dataset = "training\data\processed\sf_supervised_dataset2024.npz"
        output_dir = "checkpoints"
        epochs = 10
        batch_size = 32
        learning_rate = 0.001
        value_target = "delta_cp"  # <-- choose a valid key from your NPZ
        resume = None
        
    args = Args()
    logger = setup_logger('train_npz')
    logger.info(f"Loading NPZ dataset from {args.dataset}")
    dataset = NPZDataset(args.dataset, value_target=args.value_target)

    # Train/validation split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = create_model({'num_planes': dataset.X.shape[1], 'policy_dim': 64*64*5})  # 18 planes, 20480 policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    start_epoch = 0

    # Optional: resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, logger)
        val_metrics = validate(model, val_loader, device)

        log_metrics(logger, train_metrics, epoch=epoch + 1)
        log_metrics(logger, {f'val_{k}': v for k, v in val_metrics.items()}, epoch=epoch + 1)

        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']

        checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch + 1:03d}.pt'
        save_checkpoint(
            model, optimizer, epoch + 1,
            {'train': train_metrics, 'val': val_metrics, 'best_val_loss': best_val_loss},
            checkpoint_path,
            is_best=is_best
        )

        cleanup_old_checkpoints(args.output_dir, keep_last_n=5)

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
