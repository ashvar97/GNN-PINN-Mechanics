"""
3-stage training pipeline for the solid-mechanics GNN-PINN.

Mirrors the *structure* of the original GNN-PINN repo's staged training
(pretrain encoder -> auxiliary property model -> physics-informed finetune)
but re-derived for a different domain, dataset, framework and physics law:

  Stage 1 (Pretrain):  encoder + psi_head trained with plain MSE against
                        FEA strain energy Psi(delta). Builds graph embeddings
                        that are predictive of the mechanical response.

  Stage 2 (Force head):  encoder + psi_head frozen; force_head trained with
                        plain MSE against FEA reaction force F(delta), using
                        the frozen embeddings from Stage 1. Analogous to the
                        original's separate saturation-pressure model reusing
                        frozen graph embeddings.

  Stage 3 (PINN finetune): everything unfrozen; joint loss =
                        MSE(psi) + MSE(force) + lambda_pinn * energy_force_consistency
                        where the consistency term enforces F == dPsi/ddelta
                        via autograd (see physics.py).

Usage:
    python src/train.py --synthetic --epochs1 5 --epochs2 5 --epochs3 5
    python src/train.py --data-root data/raw --epochs1 200 --epochs2 200 --epochs3 100
"""
from __future__ import annotations

import argparse
import os
import json

import torch
from torch_geometric.loader import DataLoader

from dataset import MechanicalMNISTDataset
from model import MechGNNPINN
from physics import energy_force_consistency, monotonic_energy_penalty

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default(*parts):
    """Resolves a default path relative to the project root, regardless of
    the caller's current working directory (so `python src/train.py` and
    `python train.py` from inside src/ both land in the same place)."""
    return os.path.join(_ROOT, *parts)


def mse(a, b):
    return torch.mean((a - b) ** 2)


def run_epoch(model, loader, optimizer, device, stage: int, lambda_pinn: float,
              lambda_mono: float, train: bool):
    model.train(train)
    total, total_mse_psi, total_mse_force, total_pde = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        delta = batch.delta.view(-1, 1)
        y_psi = batch.y_psi.view(-1, 1)
        y_force = batch.y_force.view(-1, 1)

        if train:
            optimizer.zero_grad()

        if stage == 3:
            pde_loss, dpsi_ddelta, psi_pred, force_pred = energy_force_consistency(
                model, batch.x, batch.edge_index, batch.edge_attr, batch.batch, delta)
            l_psi = mse(psi_pred, y_psi)
            l_force = mse(force_pred, y_force)
            l_mono = monotonic_energy_penalty(dpsi_ddelta)
            loss = l_psi + l_force + lambda_pinn * pde_loss + lambda_mono * l_mono
        else:
            psi_pred, force_pred = model(batch.x, batch.edge_index, batch.edge_attr,
                                          batch.batch, delta)
            if stage == 1:
                l_psi = mse(psi_pred, y_psi)
                l_force = torch.zeros((), device=device)
                loss = l_psi
            else:  # stage == 2
                l_psi = torch.zeros((), device=device)
                l_force = mse(force_pred, y_force)
                loss = l_force
            pde_loss = torch.zeros((), device=device)

        if train:
            loss.backward()
            optimizer.step()

        total += loss.item()
        total_mse_psi += l_psi.item()
        total_mse_force += l_force.item()
        total_pde += pde_loss.item()
        n_batches += 1

    return {
        "loss": total / n_batches,
        "mse_psi": total_mse_psi / n_batches,
        "mse_force": total_mse_force / n_batches,
        "pde": total_pde / n_batches,
    }


def make_loaders(args, batch_size):
    common = dict(root=args.data_root, synthetic=args.synthetic,
                  flip_rows=args.flip_rows)
    if args.synthetic:
        train_ds = MechanicalMNISTDataset(split="train", n_synthetic=args.n_synthetic,
                                           seed=0, **common)
        val_ds = MechanicalMNISTDataset(split="train", n_synthetic=max(32, args.n_synthetic // 4),
                                         seed=1, **common)
    else:
        train_ds = MechanicalMNISTDataset(split="train", max_samples=args.max_samples, **common)
        val_ds = MechanicalMNISTDataset(split="test", max_samples=args.max_samples, **common)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=_default("data", "raw"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--flip-rows", action="store_true")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs1", type=int, default=20)
    ap.add_argument("--epochs2", type=int, default=20)
    ap.add_argument("--epochs3", type=int, default=20)
    ap.add_argument("--lambda-pinn", type=float, default=0.5)
    ap.add_argument("--lambda-mono", type=float, default=0.05)
    ap.add_argument("--ckpt-dir", default=_default("checkpoints"))
    ap.add_argument("--log-file", default=_default("outputs", "train_log.json"))
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(args, args.batch_size)

    model = MechGNNPINN(hidden=args.hidden).to(device)
    history = {"stage1": [], "stage2": [], "stage3": []}

    # ---------------- Stage 1: pretrain encoder + psi_head ----------------
    print("\n=== Stage 1: pretrain (MSE on strain energy) ===")
    model.set_trainable(encoder=True, psi_head=True, force_head=False)
    opt1 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs1):
        tr = run_epoch(model, train_loader, opt1, device, stage=1,
                        lambda_pinn=0.0, lambda_mono=0.0, train=True)
        va = run_epoch(model, val_loader, opt1, device, stage=1,
                        lambda_pinn=0.0, lambda_mono=0.0, train=False)
        history["stage1"].append({"epoch": ep, "train": tr, "val": va})
        print(f"[stage1 {ep:03d}] train_mse_psi={tr['mse_psi']:.5f}  val_mse_psi={va['mse_psi']:.5f}")
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "stage1_pretrain.pt"))

    # ---------------- Stage 2: force head off frozen embedding -------------
    print("\n=== Stage 2: force head (MSE on reaction force, frozen encoder) ===")
    model.set_trainable(encoder=False, psi_head=False, force_head=True)
    opt2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs2):
        tr = run_epoch(model, train_loader, opt2, device, stage=2,
                        lambda_pinn=0.0, lambda_mono=0.0, train=True)
        va = run_epoch(model, val_loader, opt2, device, stage=2,
                        lambda_pinn=0.0, lambda_mono=0.0, train=False)
        history["stage2"].append({"epoch": ep, "train": tr, "val": va})
        print(f"[stage2 {ep:03d}] train_mse_force={tr['mse_force']:.5f}  val_mse_force={va['mse_force']:.5f}")
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "stage2_force.pt"))

    # ---------------- Stage 3: joint PINN finetune -------------------------
    print("\n=== Stage 3: PINN finetune (MSE(psi) + MSE(force) + lambda*|F - dPsi/ddelta|^2) ===")
    model.set_trainable(encoder=True, psi_head=True, force_head=True)
    opt3 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.5)
    for ep in range(args.epochs3):
        tr = run_epoch(model, train_loader, opt3, device, stage=3,
                        lambda_pinn=args.lambda_pinn, lambda_mono=args.lambda_mono, train=True)
        va = run_epoch(model, val_loader, opt3, device, stage=3,
                        lambda_pinn=args.lambda_pinn, lambda_mono=args.lambda_mono, train=False)
        history["stage3"].append({"epoch": ep, "train": tr, "val": va})
        print(f"[stage3 {ep:03d}] loss={tr['loss']:.5f} pde={tr['pde']:.6f}  "
              f"val_loss={va['loss']:.5f} val_pde={va['pde']:.6f}")
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "stage3_finetune.pt"))

    with open(args.log_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {args.log_file}")


if __name__ == "__main__":
    main()
