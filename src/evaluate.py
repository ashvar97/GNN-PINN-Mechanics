"""
Evaluate a trained MechGNNPINN checkpoint: reports MAE/RMSE/R^2 for both
strain-energy and reaction-force predictions, checks how well F == dPsi/ddelta
holds on held-out data, and saves parity plots + a predictions CSV.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import MechanicalMNISTDataset
from model import MechGNNPINN
from physics import energy_force_consistency

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default(*parts):
    return os.path.join(_ROOT, *parts)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=_default("data", "raw"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--flip-rows", action="store_true")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ckpt", default=_default("checkpoints", "stage3_finetune.pt"))
    ap.add_argument("--out", default=_default("outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(root=args.data_root, synthetic=args.synthetic, flip_rows=args.flip_rows)
    if args.synthetic:
        ds = MechanicalMNISTDataset(split="test", n_synthetic=args.n_synthetic, seed=2, **common)
    else:
        ds = MechanicalMNISTDataset(split="test", max_samples=args.max_samples, **common)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = MechGNNPINN(hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    psi_true, psi_pred_all, force_true, force_pred_all, residuals = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        delta = batch.delta.view(-1, 1)
        pde_loss, dpsi_ddelta, psi_pred, force_pred = energy_force_consistency(
            model, batch.x, batch.edge_index, batch.edge_attr, batch.batch, delta)

        psi_true.append(batch.y_psi.view(-1).detach().cpu().numpy())
        psi_pred_all.append(psi_pred.view(-1).detach().cpu().numpy())
        force_true.append(batch.y_force.view(-1).detach().cpu().numpy())
        force_pred_all.append(force_pred.view(-1).detach().cpu().numpy())
        residuals.append((force_pred - dpsi_ddelta).view(-1).detach().cpu().numpy())

    psi_true = np.concatenate(psi_true)
    psi_pred_all = np.concatenate(psi_pred_all)
    force_true = np.concatenate(force_true)
    force_pred_all = np.concatenate(force_pred_all)
    residuals = np.concatenate(residuals)

    psi_metrics = metrics(psi_true, psi_pred_all)
    force_metrics = metrics(force_true, force_pred_all)
    print("Strain energy (Psi):", psi_metrics)
    print("Reaction force (F): ", force_metrics)
    print(f"Physics residual |F - dPsi/ddelta|: mean={np.mean(np.abs(residuals)):.6f}  "
          f"max={np.max(np.abs(residuals)):.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, yt, yp, name in [
            (axes[0], psi_true, psi_pred_all, "Psi (strain energy)"),
            (axes[1], force_true, force_pred_all, "F (reaction force)"),
        ]:
            ax.scatter(yt, yp, s=6, alpha=0.4)
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlabel("true"); ax.set_ylabel("pred"); ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "parity_plots.png"), dpi=150)
        print(f"Saved parity plots to {args.out}/parity_plots.png")
    except ImportError:
        pass

    import csv
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["psi_true", "psi_pred", "force_true", "force_pred", "physics_residual"])
        for row in zip(psi_true, psi_pred_all, force_true, force_pred_all, residuals):
            w.writerow(row)
    print(f"Saved predictions to {args.out}/predictions.csv")


if __name__ == "__main__":
    main()
