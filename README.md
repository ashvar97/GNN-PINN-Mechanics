# Mech-GNN-PINN

## A GNN + PINN pipeline for solid-mechanics structure-property prediction

This project is inspired by the general idea of the [GNN-PINN thermodynamic
property paper](https://doi.org/10.1021/acs.iecr.5c02302) — a graph neural
network encoder feeding a physics-informed, multi-stage training scheme — but
is a from-scratch project in a **different domain**: solid mechanics
(structural stress/strain response) instead of molecular thermodynamics, built
on a **different dataset**, a **different graph construction**, a
**different framework** (PyTorch + PyTorch Geometric instead of hand-rolled
TensorFlow layers), and a **different physical law** for the PINN term.

It is *not* a port of that repo's code.

## Problem

Given a 2D heterogeneous elastic material block (Neo-Hookean, spatially
varying Young's modulus) and a prescribed edge displacement, predict:

- **Ψ(δ)** — total stored strain energy
- **F(δ)** — reaction force at the loaded edge

This is a classic **structure-property metamodeling** task in computational
solid mechanics: replace an expensive FEM solve with a fast learned
surrogate, conditioned on both the material's spatial heterogeneity (via a
GNN over a grid graph) and the loading state.

## Dataset: Mechanical MNIST (Uniaxial Extension)

We use the public **[Mechanical MNIST](https://github.com/elejeune11/Mechanical-MNIST)**
benchmark (Lejeune Lab, Boston University): MNIST digit bitmaps are
reinterpreted as 28×28 heterogeneous Neo-Hookean material blocks (pixel
intensity → local Young's modulus, E=1 to E=100), FEM-simulated in FEniCS
under uniaxial extension across 12 displacement increments, with strain
energy and reaction force recorded at each step (70,000 simulations total).

> E. Lejeune, "Mechanical MNIST: A benchmark dataset for mechanical
> metamodels," *Extreme Mechanics Letters*, 2020.
> https://doi.org/10.1016/j.eml.2020.100659

**Getting the data:**

```bash
python src/download_data.py
```

OpenBU (the DSpace host) blocked a plain scripted request with HTTP 403
during development of this script — it may or may not work for you. If it
fails, download the three files manually from
https://open.bu.edu/handle/2144/38693 into `data/raw/`:
`MNIST_input_files.zip`, `FEA_psi_results.zip`, `FEA_rxnforce_results.zip`,
then run `python src/download_data.py --extract-only`.

Once extracted, sanity-check the file layout before your first real training
run:

```bash
python src/dataset.py --inspect data/raw
```

**⚠️ Assumptions to verify against the real files** (documented in code, not
silently assumed):
- `src/dataset.py::_load_reaction_force` expects a plain-text `(N, 13)`
  reaction-force summary file analogous to `summary_psi_*_all.txt`. If
  `FEA_rxnforce_results.zip` instead contains one file per simulation, that
  function needs a small rewrite to aggregate them — it currently falls back
  to a finite-difference approximation of `dΨ/dstep` and prints a loud
  warning so this is never silently wrong.
- The exact physical displacement value per load step isn't hard-coded;
  `delta` is currently the normalized step index (`step/12`). If you find the
  true displacement magnitude per step in the dataset documentation, wire it
  into `MechanicalMNISTDataset._build` for physically-scaled derivatives.

**No real data yet / just want to see it run?** Every script accepts
`--synthetic`, which generates an in-memory toy dataset with identical
shapes/schema from a closed-form energy model — enough to exercise the full
graph → GNN → PINN-loss → training loop without downloading anything.

## Method

### Graph construction (`src/graph_utils.py`)
Each 28×28 material block → an 8-connected grid graph (784 nodes). Node
features: local Young's modulus, (x, y) position, and boundary-condition
indicators (bottom row = fixed edge, top row = displaced edge). Edge
features: relative offset + distance. Topology is fixed and built once;
only node features change per sample.

### Model (`src/model.py`)
- `EdgeConditionedConv`: a `MessagePassing` layer whose messages are an MLP
  over `(x_i, x_j, edge_attr)`, so it uses the physical relative position
  between grid nodes (unlike plain GCN).
- `MultiStatReadout`: global pooling that concatenates mean/max/sum/std node
  statistics into a graph embedding.
- `MechGNNPINN`: encoder + 2 message-passing layers + readout → two heads
  (`psi_head`, `force_head`), each conditioned on the scalar load step `δ`.

### Physics-informed loss (`src/physics.py`)
The PINN term enforces **energy–force work conjugacy**:

```
F(δ) = dΨ/dδ
```

computed via `torch.autograd.grad` on the model's own strain-energy output
with respect to `δ`. This is the solid-mechanics analogue of the original
paper's Gibbs-free-energy equilibrium penalty — a genuine physical
consistency law for this domain, expressed as a differential residual rather
than an algebraic one. A small secondary penalty discourages non-physical
energy decrease under monotonically increasing load (`dΨ/dδ < 0`).

### 3-stage training (`src/train.py`)
1. **Pretrain** — encoder + `psi_head`, plain MSE on strain energy.
2. **Force head** — freeze encoder + `psi_head`; train `force_head` with MSE
   on reaction force, reusing the frozen embedding (mirrors the original
   repo's separate saturation-pressure model reusing frozen graph
   embeddings).
3. **PINN finetune** — unfreeze everything; joint loss
   `MSE(Ψ) + MSE(F) + λ_pinn · consistency + λ_mono · monotonicity`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test with synthetic data (no download needed):
python src/train.py --synthetic --epochs1 20 --epochs2 20 --epochs3 20
python src/evaluate.py --synthetic --ckpt checkpoints/stage3_finetune.pt

# Real training once data/raw/ is populated:
python src/train.py --data-root data/raw --epochs1 200 --epochs2 200 --epochs3 100
python src/evaluate.py --data-root data/raw --ckpt checkpoints/stage3_finetune.pt
```

`evaluate.py` reports MAE/RMSE/R² for both targets, checks how well
`F ≈ dΨ/dδ` holds on held-out data, and saves parity plots + a predictions
CSV to `outputs/`.

## Project structure

```
├── requirements.txt
├── src/
│   ├── graph_utils.py     # fixed grid-graph topology + node feature encoding
│   ├── dataset.py         # Mechanical MNIST loading (real + synthetic fallback)
│   ├── download_data.py   # best-effort OpenBU downloader
│   ├── model.py           # EdgeConditionedConv, MultiStatReadout, MechGNNPINN
│   ├── physics.py         # energy-force consistency PINN loss
│   ├── train.py           # 3-stage training loop
│   └── evaluate.py        # metrics, parity plots, predictions CSV
├── data/raw/               # (gitignored) downloaded Mechanical MNIST files
├── checkpoints/            # (gitignored) saved model weights per stage
└── outputs/                 # (gitignored) logs, plots, predictions
```

## Status / what's verified vs. not

- ✅ Full pipeline (graph construction → GNN → 3-stage training →
  physics-consistency loss → evaluation) runs end-to-end and was smoke-tested
  on synthetic data in this environment (PyTorch 2.13, PyTorch Geometric 2.8,
  CPU). The PINN residual `|F − dΨ/dδ|` visibly drops during Stage 3.
- ⚠️ Not yet run against the real Mechanical MNIST files — the reaction-force
  file layout in particular should be checked with `dataset.py --inspect`
  once downloaded (see "Assumptions to verify" above).
