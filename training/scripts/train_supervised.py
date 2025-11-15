import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torch.nn as nn
import torch.nn.functional as F

POLICY_DIM = 64 * 64 * 5  # 20,480


class SFNPZDataset(Dataset):
    """
    Dataset for Stockfish-labeled NPZ.

    Trains on:
      - policy_idx : best-move index (int)
      - value_before      : scaled cp_before
      - value_after_best  : scaled cp_after_best
      - delta_target      : scaled delta_cp
    """

    def __init__(
        self,
        npz_path: str,
        cp_scale: float = 400.0,
        delta_scale: float = 100.0,
    ):
        data = np.load(npz_path)

        # Required fields from your generator
        self.X = data["X"]                          # (N, 18, 8, 8)
        self.y_policy_best = data["y_policy_best"]  # (N,)
        self.cp_before = data["cp_before"]          # (N,)
        self.cp_after_best = data["cp_after_best"]  # (N,)
        self.delta_cp = data["delta_cp"]            # (N,)

        # Precompute regression targets, scaled into [-1, 1]
        self.value_before = np.tanh(self.cp_before / cp_scale).astype(np.float32)
        self.value_after_best = np.tanh(self.cp_after_best / cp_scale).astype(np.float32)
        self.delta_target = np.tanh(self.delta_cp / delta_scale).astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        planes = torch.from_numpy(self.X[idx]).float()  # (18, 8, 8)
        policy_idx = torch.tensor(int(self.y_policy_best[idx]), dtype=torch.long)

        value_before = torch.tensor(self.value_before[idx], dtype=torch.float32)
        value_after_best = torch.tensor(self.value_after_best[idx], dtype=torch.float32)
        delta_target = torch.tensor(self.delta_target[idx], dtype=torch.float32)

        return {
            "planes": planes,
            "policy_idx": policy_idx,
            "value_before": value_before,
            "value_after_best": value_after_best,
            "delta_target": delta_target,
        }


def collate_sf(batch):
    """
    Collate dict -> dict of batched tensors.
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out




class DummyChessModel(nn.Module):
    """
    Dummy model with:
      - policy head (logits over 20,480 moves)
      - value_before head
      - value_after_best head
      - delta head
    """

    def __init__(self, policy_dim: int = POLICY_DIM):
        super().__init__()
        # Super simple conv stack
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc_shared = nn.Linear(64 * 8 * 8, 512)

        self.policy_head = nn.Linear(512, policy_dim)
        self.value_before_head = nn.Linear(512, 1)
        self.value_after_best_head = nn.Linear(512, 1)
        self.delta_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, 18, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)      # (B, 64*8*8)
        x = F.relu(self.fc_shared(x))  # (B, 512)

        policy_logits = self.policy_head(x)                  # (B, 20480)
        value_before = torch.tanh(self.value_before_head(x))        # (B, 1)
        value_after_best = torch.tanh(self.value_after_best_head(x))  # (B, 1)
        delta_pred = torch.tanh(self.delta_head(x))                 # (B, 1)

        return policy_logits, value_before, value_after_best, delta_pred


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    w_value_before: float = 1.0,
    w_value_after: float = 1.0,
    w_delta: float = 1.0,
):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_vb_loss = 0.0
    total_va_loss = 0.0
    total_delta_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        planes = batch["planes"].to(device)          # (B, 18, 8, 8)
        policy_idx = batch["policy_idx"].to(device)  # (B,)
        value_before_t = batch["value_before"].to(device)          # (B,)
        value_after_best_t = batch["value_after_best"].to(device)  # (B,)
        delta_target_t = batch["delta_target"].to(device)          # (B,)

        optimizer.zero_grad()

        (policy_logits,
         value_before_pred,
         value_after_best_pred,
         delta_pred) = model(planes)

        # policy loss
        policy_loss = ce_loss_fn(policy_logits, policy_idx)

        # regression losses
        value_before_pred = value_before_pred.squeeze(-1)  # (B,)
        value_after_best_pred = value_after_best_pred.squeeze(-1)
        delta_pred = delta_pred.squeeze(-1)

        vb_loss = mse_loss_fn(value_before_pred, value_before_t)
        va_loss = mse_loss_fn(value_after_best_pred, value_after_best_t)
        d_loss = mse_loss_fn(delta_pred, delta_target_t)

        loss = (
            policy_loss
            + w_value_before * vb_loss
            + w_value_after * va_loss
            + w_delta * d_loss
        )

        loss.backward()
        optimizer.step()

        bsz = planes.size(0)
        total_samples += bsz
        total_loss += loss.item() * bsz
        total_policy_loss += policy_loss.item() * bsz
        total_vb_loss += vb_loss.item() * bsz
        total_va_loss += va_loss.item() * bsz
        total_delta_loss += d_loss.item() * bsz

    avg_loss = total_loss / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    avg_vb_loss = total_vb_loss / total_samples
    avg_va_loss = total_va_loss / total_samples
    avg_delta_loss = total_delta_loss / total_samples

    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_before_loss": avg_vb_loss,
        "value_after_best_loss": avg_va_loss,
        "delta_loss": avg_delta_loss,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    w_value_before: float = 1.0,
    w_value_after: float = 1.0,
    w_delta: float = 1.0,
    print_examples: bool = True,
    num_examples: int = 5,
):
    """
    Validation loop that computes losses and optionally prints some
    predicted move indices vs best-move indices.
    """
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_vb_loss = 0.0
    total_va_loss = 0.0
    total_delta_loss = 0.0
    total_samples = 0

    examples_printed = 0

    for batch in dataloader:
        planes = batch["planes"].to(device)          # (B, 18, 8, 8)
        policy_idx = batch["policy_idx"].to(device)  # (B,)
        value_before_t = batch["value_before"].to(device)          # (B,)
        value_after_best_t = batch["value_after_best"].to(device)  # (B,)
        delta_target_t = batch["delta_target"].to(device)          # (B,)

        (policy_logits,
         value_before_pred,
         value_after_best_pred,
         delta_pred) = model(planes)

        policy_loss = ce_loss_fn(policy_logits, policy_idx)

        value_before_pred = value_before_pred.squeeze(-1)
        value_after_best_pred = value_after_best_pred.squeeze(-1)
        delta_pred = delta_pred.squeeze(-1)

        vb_loss = mse_loss_fn(value_before_pred, value_before_t)
        va_loss = mse_loss_fn(value_after_best_pred, value_after_best_t)
        d_loss = mse_loss_fn(delta_pred, delta_target_t)

        loss = (
            policy_loss
            + w_value_before * vb_loss
            + w_value_after * va_loss
            + w_delta * d_loss
        )

        bsz = planes.size(0)
        total_samples += bsz
        total_loss += loss.item() * bsz
        total_policy_loss += policy_loss.item() * bsz
        total_vb_loss += vb_loss.item() * bsz
        total_va_loss += va_loss.item() * bsz
        total_delta_loss += d_loss.item() * bsz

        # Print a few prediction vs target examples
        if print_examples and examples_printed < num_examples:
            # predicted move indices: argmax over policy logits
            preds = policy_logits.argmax(dim=1)  # (B,)
            for i in range(bsz):
                if examples_printed >= num_examples:
                    break
                pred_idx = preds[i].item()
                true_idx = policy_idx[i].item()
                print(f"  Example {examples_printed + 1}: pred={pred_idx}, best={true_idx}")
                examples_printed += 1

    avg_loss = total_loss / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    avg_vb_loss = total_vb_loss / total_samples
    avg_va_loss = total_va_loss / total_samples
    avg_delta_loss = total_delta_loss / total_samples

    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_before_loss": avg_vb_loss,
        "value_after_best_loss": avg_va_loss,
        "delta_loss": avg_delta_loss,
    }


def main_train():
    npz_path = r"training\data\processed\sf_supervised_dataset2425.npz"

    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Full dataset
    full_dataset = SFNPZDataset(npz_path)

    # --- train/validation split ---
    val_frac = 0.25
    val_size = int(len(full_dataset) * val_frac)
    train_size = len(full_dataset) - val_size

    # Optional: fix seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sf,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sf,
    )

    model = DummyChessModel(policy_dim=POLICY_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    w_vb = 0.8
    w_va = 0.8
    w_delta = 0.2

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            w_value_before=w_vb,
            w_value_after=w_va,
            w_delta=w_delta,
        )

        print(
            f"Epoch {epoch} TRAIN: "
            f"loss={train_stats['loss']:.4f}, "
            f"policy={train_stats['policy_loss']:.4f}, "
            f"vb={train_stats['value_before_loss']:.4f}, "
            f"va={train_stats['value_after_best_loss']:.4f}, "
            f"delta={train_stats['delta_loss']:.4f}"
        )

        # --- Validation + example predictions ---
        print(f"Epoch {epoch} VAL:")
        val_stats = evaluate(
            model,
            val_loader,
            device,
            w_value_before=w_vb,
            w_value_after=w_va,
            w_delta=w_delta,
            print_examples=True,
            num_examples=5,  # adjust if you want more/less
        )

        print(
            f"Epoch {epoch} VAL STATS: "
            f"loss={val_stats['loss']:.4f}, "
            f"policy={val_stats['policy_loss']:.4f}, "
            f"vb={val_stats['value_before_loss']:.4f}, "
            f"va={val_stats['value_after_best_loss']:.4f}, "
            f"delta={val_stats['delta_loss']:.4f}"
        )

    torch.save(model.state_dict(), "model_policy_value_multihead.pth")
    print("Saved model to model_policy_value_multihead.pth")


if __name__ == "__main__":
    main_train()
