"""Interpretability helpers — per-feature importance maps.

One PNG (+ SVG) is written for each method, alongside its scatter plot.
Three styles:

* :func:`pearson_loading_map` – correlate every CV column with every
  pairwise Cα distance. Used as a universal "pseudo-loading" for black-box
  linear/manifold methods where there is no explicit basis (Kernel PCA,
  Isomap, LLE, UMAP, diffusion maps, autoencoders post-hoc).
* :func:`jacobian_contact_map` – mean ``|∂CV_k / ∂x_pair|`` over a sampled
  batch, for any torch encoder whose input is the pairwise-distance vector.
* :func:`gnn_residue_importance` – mean ``|∂CV_k / ∂pos|`` per atom for GNN
  encoders that take a PyG graph as input.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ───────────────────────────── helpers ──────────────────────────────────────

def _save(fig, save_path: Path) -> None:
    save_path = Path(save_path).with_suffix(".svg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved interpret → %s", save_path)


def _pairs_to_matrix(vec: np.ndarray, n_atoms: int) -> np.ndarray:
    """Upper-triangle (k=1) pair vector → symmetric (n_atoms, n_atoms)."""
    M = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    iu, ju = np.triu_indices(n_atoms, k=1)
    if len(vec) != len(iu):
        # Fallback: pad/truncate.
        v = np.zeros(len(iu))
        v[: min(len(iu), len(vec))] = vec[: min(len(iu), len(vec))]
        vec = v
    M[iu, ju] = vec
    M[ju, iu] = vec
    return M


def _contact_panels(
    loadings: np.ndarray, n_atoms: int, save_path: Path, title: str,
    mode: str = "pearson",
) -> None:
    """Render per-CV contact-map panels with a colour scale shared across CVs.

    mode="pearson"   → fixed [-1, 1], diverging RdBu_r
    mode="magnitude" → [0, max|loading| across all CVs], viridis
    """
    loadings = np.asarray(loadings)
    if loadings.ndim == 1:
        loadings = loadings[:, None]
    n_cvs = loadings.shape[1]
    if mode == "pearson":
        vmin, vmax, cmap = -1.0, 1.0, "RdBu_r"
    else:
        vmax = max(1e-12, float(np.abs(loadings).max()))
        vmin, cmap = 0.0, "viridis"
    fig, axes = plt.subplots(1, n_cvs, figsize=(4.2 * n_cvs, 4.0), squeeze=False)
    axes = axes[0]
    for k, ax in enumerate(axes):
        M = _pairs_to_matrix(loadings[:, k], n_atoms)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"CV {k + 1}")
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title, fontsize=12, y=1.02)
    _save(fig, save_path)


# ───────────────────────────── public API ───────────────────────────────────

def pearson_loading_map(
    cvs: np.ndarray, feat_distances: np.ndarray, n_atoms: int,
    save_path: Path, title: str,
) -> None:
    """Pearson correlation of each CV with each pairwise distance."""
    cvs = np.asarray(cvs)
    X = np.asarray(feat_distances)
    if cvs.shape[0] != X.shape[0]:
        logger.warning("pearson_loading_map: length mismatch %d vs %d — skipping.",
                       cvs.shape[0], X.shape[0])
        return
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    Xn = (X - X_mean) / np.where(X_std > 1e-12, X_std, 1.0)
    d_out = cvs.shape[1]
    loadings = np.zeros((X.shape[1], d_out))
    for k in range(d_out):
        c = cvs[:, k].astype(np.float64)
        cs = c.std()
        if cs < 1e-12:
            continue
        cn = (c - c.mean()) / cs
        loadings[:, k] = (Xn * cn[:, None]).mean(axis=0)
    _contact_panels(loadings, n_atoms, save_path, title, mode="pearson")


def jacobian_contact_map(
    encoder_fn, X_np: np.ndarray, n_atoms: int,
    save_path: Path, title: str,
    n_samples: int = 512, device=None,
) -> None:
    """Mean |∂z_k/∂x_pair| over a sample, per output k, as a contact map.

    ``encoder_fn`` must map a ``(B, n_pairs)`` float tensor to ``(B, d_latent)``.
    """
    import torch
    X = torch.as_tensor(X_np, dtype=torch.float32)
    if device is not None:
        X = X.to(device)
    n = len(X)
    if n == 0:
        return
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    Xs = X[idx].detach().clone().requires_grad_(True)
    try:
        Z = encoder_fn(Xs)
    except Exception:
        logger.exception("Jacobian forward failed for %s", title)
        return
    if isinstance(Z, (tuple, list)):
        Z = Z[0]
    if Z.ndim != 2:
        logger.warning("Jacobian skipped for %s (output ndim=%d).", title, Z.ndim)
        return
    d_out = Z.shape[1]
    loadings = np.zeros((X.shape[1], d_out))
    for k in range(d_out):
        grads = torch.autograd.grad(
            Z[:, k].sum(), Xs, retain_graph=(k < d_out - 1),
        )[0]
        loadings[:, k] = grads.abs().mean(0).detach().cpu().numpy()
    _contact_panels(loadings, n_atoms, save_path, title, mode="magnitude")


def gnn_residue_importance(
    forward_fn, graphs, n_atoms: int,
    save_path: Path, title: str,
    n_samples: int = 128, device=None,
) -> None:
    """Mean |∂z_k/∂pos| per atom across sampled graphs → bar chart."""
    import torch
    from torch_geometric.data import Batch
    n = len(graphs)
    if n == 0:
        return
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    chunks = [graphs[int(i)] for i in idx]
    batch = Batch.from_data_list(chunks)
    if device is not None:
        batch = batch.to(device)
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    try:
        Z = forward_fn(batch)
    except Exception:
        logger.exception("GNN residue importance forward failed for %s", title)
        return
    if isinstance(Z, (tuple, list)):
        # conventional "(recon, z)" encoder output
        Z = Z[-1]
    if Z.ndim != 2:
        logger.warning("GNN residue importance skipped (output ndim=%d).", Z.ndim)
        return
    d_out = Z.shape[1]
    B = len(chunks)
    importance = np.zeros((n_atoms, d_out))
    for k in range(d_out):
        grads = torch.autograd.grad(
            Z[:, k].sum(), batch.pos, retain_graph=(k < d_out - 1),
        )[0]
        norms = grads.norm(dim=-1).detach().cpu().numpy()
        importance[:, k] = norms.reshape(B, n_atoms).mean(0)

    fig, axes = plt.subplots(1, d_out, figsize=(4.5 * d_out, 3.2),
                             squeeze=False, sharey=True)
    axes = axes[0]
    ymax = max(1e-12, float(importance.max())) * 1.05
    for k, ax in enumerate(axes):
        ax.bar(np.arange(1, n_atoms + 1), importance[:, k], color="C0")
        ax.set_xlabel("residue")
        if k == 0:
            ax.set_ylabel("mean |∂CV/∂pos|")
        ax.set_title(f"CV {k + 1}")
        ax.set_ylim(0.0, ymax)
    fig.suptitle(title, fontsize=12, y=1.02)
    _save(fig, save_path)
