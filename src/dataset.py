"""
PyTorch Geometric dataset for the Mechanical MNIST (Uniaxial Extension)
benchmark: https://github.com/elejeune11/Mechanical-MNIST

Each raw sample (one MNIST digit -> one heterogeneous material block) is
expanded into up to N_STEPS graph samples, one per FEA loading increment,
with:
    x, edge_index, edge_attr  -> fixed grid-graph (see graph_utils.py)
    delta                     -> scalar load-step control variable (proxy for
                                  prescribed top-edge displacement)
    y_psi                     -> strain energy at that step (regression target)
    y_force                   -> reaction force at that step (regression target)

Two data sources are supported:

1. Real Mechanical MNIST files (after running `python src/download_data.py`).
   The loader currently parses:
     - MNIST_input_files/{train,test}/mnist_img_{split}.txt  (flattened bitmaps)
     - FEA_psi_results/summary_psi_{split}_all.txt           (N x 13 strain energy)
   Reaction-force parsing depends on the exact internal layout of
   `FEA_rxnforce_results.zip`, which was NOT verified against the actual
   downloaded files in this environment (network/size constraints). Run
   `python src/dataset.py --inspect data/raw` after downloading to print the
   discovered file tree, then adjust `_load_reaction_force` accordingly if it
   doesn't match. Until then, `MechanicalMNISTDataset` degrades gracefully:
   if the force file can't be parsed it derives a *placeholder* force target
   via finite differences of the strain energy curve (dPsi/dstep) so the rest
   of the pipeline still runs -- clearly flagged as an approximation, not a
   silent substitution.

2. Synthetic fallback (`synthetic=True`): generates a small in-memory dataset
   with the exact same schema/shapes from a closed-form toy hyperelastic
   energy model. This lets the full pipeline (data -> graph -> GNN -> PINN
   loss -> training loop) be exercised and unit-tested without needing the
   multi-GB real dataset.
"""
from __future__ import annotations

import os
import glob
import zipfile
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from graph_utils import build_grid_topology, bitmap_to_node_features, GRID, N_NODES

N_STEPS = 12  # Mechanical MNIST provides 12 applied-displacement increments per sample


class MechanicalMNISTDataset(InMemoryDataset):
    def __init__(self, root: str, split: str = "train", synthetic: bool = False,
                 n_synthetic: int = 256, max_samples: int | None = None,
                 flip_rows: bool = False, seed: int = 0):
        self.split = split
        self.synthetic = synthetic
        self.n_synthetic = n_synthetic
        self.max_samples = max_samples
        self.flip_rows = flip_rows
        self.seed = seed
        self.edge_index, self.edge_attr, self.coords, self.boundary = build_grid_topology()
        super().__init__(root)
        self.data, self.slices = self._build()

    # InMemoryDataset plumbing (unused for the synthetic/on-the-fly path, but
    # required by the base class).
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    # ------------------------------------------------------------------ #

    def _build(self):
        if self.synthetic:
            bitmaps, psi_all, force_all = _make_synthetic_arrays(
                self.n_synthetic, seed=self.seed)
        else:
            bitmaps, psi_all = _load_bitmaps_and_psi(self.root, self.split)
            force_all = _load_reaction_force(self.root, self.split, psi_all)

        if self.max_samples is not None:
            bitmaps = bitmaps[: self.max_samples]
            psi_all = psi_all[: self.max_samples]
            force_all = force_all[: self.max_samples]

        data_list = []
        for i in range(bitmaps.shape[0]):
            x = bitmap_to_node_features(bitmaps[i], self.coords, self.boundary,
                                         flip_rows=self.flip_rows)
            for step in range(1, N_STEPS + 1):
                delta = torch.tensor([step / N_STEPS], dtype=torch.float32)
                y_psi = torch.tensor([psi_all[i, step]], dtype=torch.float32)
                y_force = torch.tensor([force_all[i, step]], dtype=torch.float32)
                data_list.append(Data(
                    x=x, edge_index=self.edge_index, edge_attr=self.edge_attr,
                    delta=delta, y_psi=y_psi, y_force=y_force,
                ))
        return self.collate(data_list)


# ---------------------------------------------------------------------- #
# Real-data loaders
# ---------------------------------------------------------------------- #

def _load_bitmaps_and_psi(root: str, split: str):
    img_path = _find_one(root, f"mnist_img_{split}*.txt")
    psi_path = _find_one(root, f"summary_psi_{split}_all.txt")
    if img_path is None or psi_path is None:
        raise FileNotFoundError(
            f"Could not find Mechanical MNIST '{split}' bitmap/psi files under {root}. "
            "Run `python src/download_data.py` first, or pass synthetic=True."
        )
    flat = np.loadtxt(img_path)
    n = flat.shape[0]
    bitmaps = flat.reshape(n, GRID, GRID)
    psi_all = np.loadtxt(psi_path)  # [n, 13] -> column 0 is step 0 (undeformed, ~0)
    return bitmaps, psi_all


def _load_reaction_force(root: str, split: str, psi_all: np.ndarray):
    force_path = _find_one(root, f"*rxnforce*{split}*")
    if force_path is not None:
        try:
            force_all = np.loadtxt(force_path)
            if force_all.shape == psi_all.shape:
                return force_all
        except Exception:
            pass
    print(
        "[WARN] Reaction-force file not found or not in the expected "
        "(N, 13) plain-text layout. Falling back to a finite-difference "
        "approximation dPsi/dstep as a placeholder force target -- fix "
        "`_load_reaction_force` in src/dataset.py once you've inspected the "
        "real FEA_rxnforce_results.zip contents (`python src/dataset.py "
        "--inspect data/raw`)."
    )
    return np.gradient(psi_all, axis=1)


def _find_one(root: str, pattern: str):
    matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return matches[0] if matches else None


def inspect_raw_tree(root: str):
    print(f"Contents under {root}:")
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            print(" ", os.path.relpath(os.path.join(dirpath, f), root))
    for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                print(f"\n{os.path.relpath(z, root)} contains {len(names)} entries, e.g.:")
                for n in names[:10]:
                    print("   ", n)
        except zipfile.BadZipFile:
            print(f"\n{z}: not a valid zip (probably still an LFS pointer / partial download)")


# ---------------------------------------------------------------------- #
# Synthetic fallback (closed-form toy Neo-Hookean-ish energy, for smoke tests)
# ---------------------------------------------------------------------- #

def _make_synthetic_arrays(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    bitmaps = rng.integers(0, 256, size=(n, GRID, GRID)).astype(np.float32)

    # Toy closed-form energy: Psi(step) = k_eff * step^2, force = dPsi/dstep = 2*k_eff*step,
    # where k_eff is a simple stand-in "effective stiffness" derived from the mean
    # pixel intensity of each block (stiffer material -> higher effective stiffness).
    mean_E = 1.0 + (bitmaps.mean(axis=(1, 2)) / 255.0) * 99.0  # [n]
    k_eff = mean_E / 100.0  # normalize to O(1)

    steps = np.arange(0, N_STEPS + 1, dtype=np.float32)  # 0..12
    psi_all = k_eff[:, None] * (steps[None, :] ** 2) / N_STEPS**2
    force_all = 2.0 * k_eff[:, None] * (steps[None, :] / N_STEPS**2)
    return bitmaps, psi_all, force_all


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", type=str, default=None,
                    help="Path to a raw data directory to print its file tree / zip contents.")
    args = p.parse_args()
    if args.inspect:
        inspect_raw_tree(args.inspect)
