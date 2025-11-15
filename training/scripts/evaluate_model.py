#!/usr/bin/env python3
"""
Evaluate trained ResNetPolicyValue model on NPZ test dataset.
Reports:
- Top-1 move accuracy
- Top-k move accuracy
- Mean/median value error (in centipawns)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# -----------------------------
# Add project root to sys.path
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve() #make sure it has 3 parents here 
sys.path.append(str(PROJECT_ROOT))

from training.models.resnet_policy_value import create_model
from training.scripts.train_supervised_version2 import NPZDataset, setup_logger

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
TEST_DATASET_PATH = PROJECT_ROOT / "training" / "data" / "processed" / "sf_supervised_dataset2024.npz"
TOP_K = 3
BATCH_SIZE = 256
NUM_PLANES =  18     # must match training
POLICY_DIM = 64*64*5 # 20480
VALUE_SCALE = 100.0   # scale centipawns to [-1,1] for network

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model, dataset, device, top_k=3, value_scale=VALUE_SCALE):
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    value_errors = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for X, y_policy, y_value in dataloader:
            X = X.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            # Forward pass
            policy_logits, value_pred = model(X)

            # Top-1 accuracy
            pred_moves = policy_logits.argmax(dim=-1)
            correct_top1 += (pred_moves == y_policy).sum().item()

            # Top-k accuracy
            _, topk_pred = policy_logits.topk(top_k, dim=-1)
            correct_topk += (topk_pred == y_policy.unsqueeze(-1)).any(dim=-1).sum().item()

            # Value error (convert back to centipawns)
            value_error = torch.abs(value_pred.squeeze() - y_value) * value_scale
            value_errors.extend(value_error.cpu().numpy().tolist())

    total = len(dataset)
    metrics = {
        'top1_accuracy': correct_top1 / total,
        'topk_accuracy': correct_topk / total,
        'mean_value_error': np.mean(value_errors),
        'median_value_error': np.median(value_errors),
    }

    return metrics

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Starting evaluation script...")

    # Logger
    logger = setup_logger('evaluate')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Verify paths
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    assert TEST_DATASET_PATH.exists(), f"Test dataset not found: {TEST_DATASET_PATH}"

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    model = create_model({'num_planes': NUM_PLANES, 'policy_dim': POLICY_DIM})
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Depending on checkpoint structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Load test data
    logger.info(f"Loading test data from {TEST_DATASET_PATH}")
    test_dataset = NPZDataset(str(TEST_DATASET_PATH), value_target='delta_cp', value_scale=VALUE_SCALE)
    logger.info(f"Test set size: {len(test_dataset)}")

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_dataset, device, top_k=TOP_K, value_scale=VALUE_SCALE)

    # Report results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Top-1 Move Accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-{TOP_K} Move Accuracy: {metrics['topk_accuracy']:.2%}")
    print(f"Mean Value Error: {metrics['mean_value_error']:.2f} centipawns")
    print(f"Median Value Error: {metrics['median_value_error']:.2f} centipawns")
    print("="*50)

if __name__ == '__main__':
    main()
