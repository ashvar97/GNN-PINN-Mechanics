"""
Grid-graph construction for Mechanical MNIST samples.

Each Mechanical MNIST sample is a 28x28 bitmap that defines a heterogeneous
Neo-Hookean material block (pixel intensity -> local Young's modulus). We
treat every pixel as one graph node laid out on a regular grid, connect it to
its 8 neighbours (von Neumann + diagonals, i.e. an "8-connected" mesh-like
stencil), and attach the FEA boundary conditions (bottom row fixed, top row
prescribed displacement) as node-level indicator features.

The topology (edge_index / edge_attr / coordinates / boundary masks) is
IDENTICAL for every sample -- only the per-node Young's modulus changes -- so
it is built once and reused.

Orientation note: we define row 0 of the bitmap as the BOTTOM of the physical
domain (Dirichlet-fixed edge) and row (GRID-1) as the TOP (prescribed
displacement edge), matching the Mechanical MNIST paper's y-up convention.
If you load bitmaps that are stored top-first (typical image convention),
set `flip_rows=True` in `load_bitmap_as_node_features` to correct it.
"""
from __future__ import annotations

import numpy as np
import torch

GRID = 28
N_NODES = GRID * GRID

# Mechanical MNIST material model: white pixel (255) -> E=100, black (0) -> E=1,
# linear interpolation in between. See Lejeune, "Mechanical MNIST: A benchmark
# dataset for mechanical metamodels", 2020.
E_MIN, E_MAX = 1.0, 100.0


def _node_id(r: int, c: int) -> int:
    return r * GRID + c


def build_grid_topology(connect_diagonals: bool = True):
    """Builds the fixed graph topology shared by every Mechanical MNIST sample.

    Returns
    -------
    edge_index : LongTensor [2, E]
    edge_attr  : FloatTensor [E, 3]   -> (dx, dy, distance), grid-spacing normalized
    coords     : FloatTensor [N, 2]   -> (x, y) in [0, 1], y=0 bottom / y=1 top
    boundary   : FloatTensor [N, 2]   -> (is_bottom, is_top) one-hot indicators
    """
    if connect_diagonals:
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1),           (0, 1),
                             (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    src, dst, attrs = [], [], []
    spacing = 1.0 / (GRID - 1)

    for r in range(GRID):
        for c in range(GRID):
            i = _node_id(r, c)
            for dr, dc in neighbor_offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID and 0 <= cc < GRID:
                    j = _node_id(rr, cc)
                    dx, dy = dc * spacing, dr * spacing
                    dist = float(np.hypot(dx, dy))
                    src.append(i)
                    dst.append(j)
                    attrs.append((dx, dy, dist))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    coords = torch.zeros((N_NODES, 2), dtype=torch.float32)
    boundary = torch.zeros((N_NODES, 2), dtype=torch.float32)  # (is_bottom, is_top)
    for r in range(GRID):
        y = r / (GRID - 1)
        for c in range(GRID):
            x = c / (GRID - 1)
            i = _node_id(r, c)
            coords[i, 0] = x
            coords[i, 1] = y
            if r == 0:
                boundary[i, 0] = 1.0
            if r == GRID - 1:
                boundary[i, 1] = 1.0

    return edge_index, edge_attr, coords, boundary


def bitmap_to_node_features(bitmap: np.ndarray, coords: torch.Tensor,
                             boundary: torch.Tensor, flip_rows: bool = False
                             ) -> torch.Tensor:
    """Converts one 28x28 uint8/float bitmap into a [N, 5] node feature matrix:
    (E_normalized, x, y, is_bottom, is_top).

    E_normalized is the local Young's modulus rescaled to roughly [0, 1] via
    (E - E_MIN) / (E_MAX - E_MIN), which keeps it on a similar scale to the
    coordinate / boundary features for the GNN's input layer.
    """
    bmp = np.asarray(bitmap, dtype=np.float32).reshape(GRID, GRID)
    if flip_rows:
        bmp = bmp[::-1, :]
    intensity = bmp / 255.0
    E = E_MIN + intensity * (E_MAX - E_MIN)
    E_norm = (E - E_MIN) / (E_MAX - E_MIN)  # -> [0, 1]

    E_flat = torch.tensor(E_norm.reshape(-1), dtype=torch.float32).unsqueeze(-1)
    return torch.cat([E_flat, coords, boundary], dim=-1)  # [N, 5]
