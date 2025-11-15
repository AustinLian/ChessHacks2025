#!/usr/bin/env python3
"""
Upgraded Trainer (ResNet backbone)
=================================

FEATURES:
  ✔ Folder-based dataset loading
  ✔ Train/Val split
  ✔ ResNet-style backbone (much more accurate eval)
  ✔ Weighted value loss
  ✔ CP clipping + scaling
  ✔ TQDM progress bars
  ✔ Pretty validation sample comparison
  ✔ Saves best.pt, last.pt, and per-epoch checkpoints

Required NPZ contents in folder:
  - X.npy
  - y_policy_best.npy
  - cp_before.npy
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random


# =====================================================================
# CONFIG
# =====================================================================

class Config:
    DATASET_FOLDER = r"training/data/processed/"   # folder of NPZs
    OUTPUT_DIR = "checkpoints_resnet_value"
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 3e-4
    NUM_WORKERS = 0

    POLICY_DIM = 64 * 64 * 5      # 20,480
    CP_CLIP = 2000                # clip mate eval spikes
    CP_SCALE = 200.0              # smaller scale → better gradients
    VALUE_WEIGHT = 3.0            # increase influence of value loss

    VAL_SPLIT = 0.10


cfg = Config()


# =====================================================================
# RESNET BACKBONE
# =====================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h += self.shortcut(x)
        return self.relu(h)


class SmallResNetPolicyValue(nn.Module):
    def __init__(self, num_planes, policy_dim, cp_scale):
        super().__init__()

        self.cp_scale = cp_scale

        self.l1 = BasicBlock(num_planes, 64)
        self.l2 = BasicBlock(64, 128)
        self.l3 = BasicBlock(128, 128)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 256)

        self.policy_head = nn.Linear(256, policy_dim)
        self.value_head = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc(x))

        policy_logits = self.policy_head(x)
        cp_scaled = self.value_head(x).squeeze(-1)

        cp_real = cp_scaled * self.cp_scale

        return policy_logits, cp_real, cp_scaled


# =====================================================================
# DATASET (folder of NPZs merged)
# =====================================================================

class FolderNPZDataset(Dataset):
    def __init__(self, folder, cp_clip, cp_scale):
        folder = Path(folder)
        files = sorted(folder.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ files found in {folder}")

        X_list = []
        pol_list = []
        cp_list = []

        for file in files:
            d = np.load(file)
            X_list.append(d["X"])
            pol_list.append(d["y_policy_best"])

            cp = d["cp_before"].astype(np.float32)
            cp = np.clip(cp, -cp_clip, cp_clip)
            cp_list.append(cp)

        self.X = np.concatenate(X_list).astype(np.float32)
        self.policy = np.concatenate(pol_list).astype(np.int64)
        cp_before = np.concatenate(cp_list).astype(np.float32)

        self.cp_scaled = cp_before / cp_scale
        print(f"Loaded {len(self.X)} samples from {len(files)} NPZ files.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        p = torch.from_numpy(self.X[idx])
        move = torch.tensor(self.policy[idx])
        cp = torch.tensor(self.cp_scaled[idx])
        return p, move, cp


# =====================================================================
# TRAIN OR VAL EPOCH
# =====================================================================

def train_or_val_epoch(model, loader, opt, device, epoch, train=True):
    tag = "Train" if train else "Val"
    model.train(train)

    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss(beta=3.0)

    total_loss = total_pol = total_val = 0
    total = 0

    with tqdm(loader, desc=f"{tag} Epoch {epoch}", ncols=90) as t:
        for planes, policy, cp_scaled in t:
            planes = planes.to(device)
            policy = policy.to(device)
            cp_scaled = cp_scaled.to(device)

            with torch.set_grad_enabled(train):
                logits, cp_real, cp_pred_scaled = model(planes)

                loss_pol = ce(logits, policy)
                loss_val = l1(cp_pred_scaled, cp_scaled)
                loss = loss_pol + cfg.VALUE_WEIGHT * loss_val

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            bs = planes.size(0)
            total += bs
            total_pol += loss_pol.item() * bs
            total_val += loss_val.item() * bs
            total_loss += loss.item() * bs

            t.set_postfix({
                "policy": f"{total_pol/total:.3f}",
                "value": f"{total_val/total:.3f}",
                "total": f"{total_loss/total:.3f}"
            })

    return total_pol/total, total_val/total, total_loss/total


# =====================================================================
# PRETTY VALIDATION SAMPLES
# =====================================================================

def show_val_samples(model, val_dataset, device, n=5):
    print("\n=== VALIDATION SAMPLE OUTPUTS ===\n")
    model.eval()

    for _ in range(n):
        idx = random.randint(0, len(val_dataset)-1)
        planes, move_gt, cp_gt_scaled = val_dataset[idx]

        planes = planes.unsqueeze(0).to(device)
        move_gt = int(move_gt)
        cp_gt = float(cp_gt_scaled) * cfg.CP_SCALE

        with torch.no_grad():
            logits, cp_real, cp_scaled = model(planes)
            move_pred = int(torch.argmax(logits[0]).item())

        print("-------------------------------")
        print(f"Ground truth eval:     {cp_gt:+.1f} cp")
        print(f"Predicted eval:        {float(cp_real.item()):+.1f} cp")
        print(f"Eval difference:       {float(cp_real.item()-cp_gt):+.1f} cp")
        print(f"Ground truth move idx: {move_gt}")
        print(f"Predicted move idx:    {move_pred}")
        print(f"Policy match:          {'YES' if move_pred==move_gt else 'NO'}")

    print("=== END VALIDATION SAMPLES ===\n")


# =====================================================================
# MAIN
# =====================================================================

def main():
    ds = FolderNPZDataset(cfg.DATASET_FOLDER, cfg.CP_CLIP, cfg.CP_SCALE)
    N = len(ds)

    val_size = int(N * cfg.VAL_SPLIT)
    train_size = N - val_size
    train_set, val_set = random_split(ds, [train_size, val_size])

    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SmallResNetPolicyValue(
        num_planes=ds.X.shape[1],
        policy_dim=cfg.POLICY_DIM,
        cp_scale=cfg.CP_SCALE
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        train_or_val_epoch(model, train_loader, opt, device, epoch, train=True)
        _, _, val_total = train_or_val_epoch(model, val_loader, opt, device, epoch, train=False)

        # Show sample predictions
        show_val_samples(model, val_set, device)

        torch.save(model.state_dict(), out / "last.pt")

        if val_total < best_val:
            best_val = val_total
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_loss": best_val
            }, out / "best.pt")
            print(f"** Saved BEST checkpoint at epoch {epoch} (val={best_val:.4f}) **")

        torch.save(model.state_dict(), out / f"epoch_{epoch:03d}.pt")

    print("Training complete.")
    print(f"Best Validation Loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
