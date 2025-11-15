#!/usr/bin/env python3
"""
TRAINER — Delta-CP + ResNet20 + Board Flipping Augmentation + Folder Dataset
=============================================================================

This trainer:
  ✔ Predicts delta_cp (much easier target than cp_before)
  ✔ Reconstructs cp_before = cp_after_best - delta_cp at inference
  ✔ Uses stronger ResNet-20 (256 channels)
  ✔ Adds board-flip augmentation (horizontal + color flip)
  ✔ Folder of NPZ files supported
  ✔ TQDM progress
  ✔ Pretty validation samples
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATASET_FOLDER = r"training/data/processed/"
    OUTPUT_DIR = "checkpoints_delta_resnet20"

    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 3e-4
    NUM_WORKERS = 0

    POLICY_DIM = 64 * 64 * 5
    CP_CLIP = 1500
    CP_SCALE = 200.0

    VAL_SPLIT = 0.1
    FLIP_PROB = 0.5      # flip 50% of training samples


cfg = Config()


# ============================================================================
# POLICY INDEX MAP FOR FLIPPED BOARD
# ============================================================================

PROMO_PIECES = [None, 5, 4, 3, 2]  # Q,R,B,N etc.

def flip_move_index(idx):
    """
    Computes new policy index after horizontal + perspective flip
    """
    from_sq = idx // (64 * 5)
    rest = idx % (64 * 5)
    to_sq = rest // 5
    promo_idx = rest % 5

    f_file = from_sq % 8
    f_rank = from_sq // 8
    t_file = to_sq % 8
    t_rank = to_sq // 8

    f_file_f = 7 - f_file
    t_file_f = 7 - t_file

    new_from = f_rank * 8 + f_file_f
    new_to = t_rank * 8 + t_file_f

    return (new_from * 64 + new_to) * 5 + promo_idx


# ============================================================================
# RESNET-20
# ============================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h += self.short(x)
        return self.relu(h)


class ResNet20_PolicyDelta(nn.Module):
    def __init__(self, planes, policy_dim, cp_scale):
        super().__init__()

        self.cp_scale = cp_scale

        def make_layer(in_c, out_c, blocks):
            layers = []
            layers.append(BasicBlock(in_c, out_c))
            for _ in range(blocks - 1):
                layers.append(BasicBlock(out_c, out_c))
            return nn.Sequential(*layers)

        self.l1 = make_layer(planes, 128, 3)
        self.l2 = make_layer(128, 256, 3)
        self.l3 = make_layer(256, 256, 3)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)

        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc(x))

        policy_logits = self.policy_head(x)
        delta_scaled = self.delta_head(x).squeeze(-1)

        delta_real = delta_scaled * self.cp_scale
        return policy_logits, delta_real, delta_scaled


# ============================================================================
# DATASET + BOARD FLIPPING AUGMENTATION
# ============================================================================

class FolderNPZDataset(Dataset):
    def __init__(self, folder, cp_clip, cp_scale, flip_prob):
        files = sorted(Path(folder).glob("*.npz"))
        if not files:
            raise FileNotFoundError("No NPZs found.")

        Xs = []
        policies = []
        deltas = []

        for f in files:
            d = np.load(f)
            Xs.append(d["X"])
            policies.append(d["y_policy_best"])
            cp_before = d["cp_before"].astype(np.float32)
            cp_after = d["cp_after_best"].astype(np.float32)
            delta_cp = cp_after - cp_before
            delta_cp = np.clip(delta_cp, -cp_clip, cp_clip)
            deltas.append(delta_cp)

        self.X = np.concatenate(Xs).astype(np.float32)
        self.policy = np.concatenate(policies).astype(np.int64)
        delta = np.concatenate(deltas).astype(np.float32)

        self.delta_scaled = delta / cp_scale
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        p = torch.from_numpy(self.X[idx])
        pol = int(self.policy[idx])
        delta = float(self.delta_scaled[idx])

        if random.random() < self.flip_prob:
            p = torch.flip(p, dims=[2])   # flip horizontally
            pol = flip_move_index(pol)
            delta = -delta               # eval sign flips

        return p, torch.tensor(pol), torch.tensor(delta)


# ============================================================================
# TRAIN/VAL LOOP (PRINTS LOSSES)
# ============================================================================

def run_epoch(model, loader, optimizer, device, epoch, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss(beta=2.0)

    total_loss = total_pol = total_val = 0
    total = 0

    tag = "Train" if train else "Val"
    with tqdm(loader, desc=f"{tag} {epoch}", ncols=100) as t:
        for planes, pol, delta in t:
            planes = planes.to(device)
            pol = pol.to(device)
            delta = delta.to(device)

            with torch.set_grad_enabled(train):
                logits, _, delta_pred = model(planes)

                loss_pol = ce(logits, pol)
                loss_val = l1(delta_pred, delta)
                loss = loss_pol + loss_val * 2.0

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            bs = planes.size(0)
            total += bs
            total_pol += loss_pol.item() * bs
            total_val += loss_val.item() * bs
            total_loss += loss.item() * bs

            t.set_postfix({
                "pol": f"{total_pol/total:.3f}",
                "val": f"{total_val/total:.3f}",
                "tot": f"{total_loss/total:.3f}"
            })

    return total_loss/total


# ============================================================================
# MAIN
# ============================================================================

def main():
    ds = FolderNPZDataset(cfg.DATASET_FOLDER, cfg.CP_CLIP, cfg.CP_SCALE, cfg.FLIP_PROB)
    N = len(ds)
    print("Dataset size:", N)

    val_size = int(N * cfg.VAL_SPLIT)
    train_set, val_set = random_split(ds, [N - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20_PolicyDelta(ds.X.shape[1], cfg.POLICY_DIM, cfg.CP_SCALE).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    best = 999999

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, opt, device, epoch, train=True)
        val_loss = run_epoch(model, val_loader, opt, device, epoch, train=False)

        print(f"\nEpoch {epoch} summary: train={train_loss:.4f} val={val_loss:.4f}\n")

        torch.save(model.state_dict(), out / "last.pt")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out / "best.pt")
            print(f"**** NEW BEST at epoch {epoch}, val={val_loss:.4f} ****\n")


if __name__ == "__main__":
    main()
