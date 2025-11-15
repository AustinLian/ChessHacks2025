# training/scripts/train_from_sf_npz.py

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from training.datasets.sf_supervised_dataset import SfSupervisedDataset


# ---------- Model definition (policy + value) ----------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + x
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self, in_channels: int, policy_size: int,
                 channels: int = 64, num_blocks: int = 4):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.p_conv = nn.Conv2d(channels, 32, 1)
        self.p_fc   = nn.Linear(32 * 8 * 8, policy_size)

        # value head
        self.v_conv = nn.Conv2d(channels, 32, 1)
        self.v_fc1  = nn.Linear(32 * 8 * 8, 128)
        self.v_fc2  = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.blocks(x)

        # policy
        p = F.relu(self.p_conv(x))
        p = p.view(p.size(0), -1)
        p_logits = self.p_fc(p)

        # value
        v = F.relu(self.v_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)

        return p_logits, v


# ---------- Training script ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="training/data/processed/sf_supervised_dataset1519.npz",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        type=str,
        default="weights/sf_supervised.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Peek at npz to get dimensions
    np_data = np.load(args.dataset)
    boards = np_data["X"]
    policy = np_data["y_policy_best"]

    in_channels = boards.shape[1]
    policy_size = int(policy.max()) + 1  # since y_policy_best is an index

    full_ds = SfSupervisedDataset(args.dataset)

    n = len(full_ds)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ChessNet(in_channels=in_channels, policy_size=policy_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(p_logits, v_pred, p_tgt, v_tgt):
        # policy: cross-entropy on best-move index
        l_policy = F.cross_entropy(p_logits, p_tgt.long())
        # value: MSE on game result in [-1,0,1]
        l_value = F.mse_loss(v_pred, v_tgt)
        return l_policy + l_value, l_policy, l_value

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, total_pl, total_vl, steps = 0.0, 0.0, 0.0, 0

        for boards, p_tgt, v_tgt in train_loader:
            boards = boards.to(device)
            p_tgt  = p_tgt.to(device)
            v_tgt  = v_tgt.to(device)

            opt.zero_grad()
            p_logits, v_pred = model(boards)
            loss, l_pl, l_vl = loss_fn(p_logits, v_pred, p_tgt, v_tgt)
            loss.backward()
            opt.step()

            total += loss.item()
            total_pl += l_pl.item()
            total_vl += l_vl.item()
            steps += 1

        print(f"Epoch {epoch}: "
              f"train loss={total/steps:.4f}, "
              f"policy={total_pl/steps:.4f}, "
              f"value={total_vl/steps:.4f}")

        # quick validation
        model.eval()
        with torch.no_grad():
            v_loss, v_steps = 0.0, 0
            for boards, p_tgt, v_tgt in val_loader:
                boards = boards.to(device)
                p_tgt  = p_tgt.to(device)
                v_tgt  = v_tgt.to(device)
                p_logits, v_pred = model(boards)
                l, _, _ = loss_fn(p_logits, v_pred, p_tgt, v_tgt)
                v_loss += l.item()
                v_steps += 1
            print(f"           val loss={v_loss/max(1, v_steps):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.out)
    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
