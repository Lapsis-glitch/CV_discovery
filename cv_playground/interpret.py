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


def _pair_to_residue(pair_imp: np.ndarray, n_atoms: int) -> np.ndarray:
    """Reduce a (n_pairs, n_cvs) pair-level importance to (n_atoms, n_cvs)
    by averaging |importance| over all pairs touching each residue."""
    pair_imp = np.abs(np.asarray(pair_imp, dtype=np.float64))
    if pair_imp.ndim == 1:
        pair_imp = pair_imp[:, None]
    n_pairs, n_cvs = pair_imp.shape
    iu, ku = np.triu_indices(n_atoms, k=1)
    if len(iu) != n_pairs:
        logger.warning("_pair_to_residue: %d pairs vs n_atoms=%d (expected %d).",
                       n_pairs, n_atoms, len(iu))
        m = min(len(iu), n_pairs)
        iu, ku = iu[:m], ku[:m]
        pair_imp = pair_imp[:m]
    res = np.zeros((n_atoms, n_cvs))
    counts = np.zeros(n_atoms)
    np.add.at(res, iu, pair_imp)
    np.add.at(res, ku, pair_imp)
    np.add.at(counts, iu, 1.0)
    np.add.at(counts, ku, 1.0)
    res /= np.where(counts > 0, counts, 1.0)[:, None]
    return res


def eta2_pair(cvs: np.ndarray, X: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """Pair-level η² (correlation ratio) of each CV w.r.t. each feature.

    For feature column j, bin frames by quantile of x_j, then
    η²_jk = Var_bin( E[CV_k | bin] ) / Var(CV_k)  ∈ [0, 1].

    Equivalent to the variance of a model-free PDP curve. Captures any
    monotonic or non-monotonic dependence — strictly stronger than Pearson²
    for nonlinear / non-monotonic CV maps.
    """
    cvs = np.asarray(cvs, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n_frames, n_pairs = X.shape
    n_cvs = cvs.shape[1]
    cv_var = cvs.var(axis=0)
    cv_var = np.where(cv_var > 1e-12, cv_var, 1.0)

    q_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros((n_pairs, n_cvs))
    for j in range(n_pairs):
        x = X[:, j]
        edges = np.unique(np.quantile(x, q_edges))
        if len(edges) < 3:
            continue
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1,
                       0, len(edges) - 2)
        n_b = len(edges) - 1
        counts = np.bincount(bins, minlength=n_b).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            continue
        p = counts / total
        for k in range(n_cvs):
            sums = np.bincount(bins, weights=cvs[:, k], minlength=n_b)
            means = np.divide(sums, counts, out=np.zeros_like(sums),
                              where=counts > 0)
            mu = (p * means).sum()
            out[j, k] = (p * (means - mu) ** 2).sum() / cv_var[k]
    return out


def pearson_pair(cvs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Signed Pearson correlation of every CV column with every feature column.
    Returns ``(n_pairs, n_cvs)``."""
    cvs = np.asarray(cvs, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Xm = X.mean(axis=0)
    Xs = X.std(axis=0)
    Xn = (X - Xm) / np.where(Xs > 1e-12, Xs, 1.0)
    out = np.zeros((X.shape[1], cvs.shape[1]))
    for k in range(cvs.shape[1]):
        c = cvs[:, k]
        cs = c.std()
        if cs < 1e-12:
            continue
        cn = (c - c.mean()) / cs
        out[:, k] = (Xn * cn[:, None]).mean(axis=0)
    return out


def condmean_range_pair(cvs: np.ndarray, X: np.ndarray,
                        n_bins: int = 15) -> np.ndarray:
    """``max_bin E[CV_k|bin] − min_bin E[CV_k|bin]`` per feature, per CV.
    Returns ``(n_pairs, n_cvs)`` in raw CV units (un-normalised swing of
    the conditional mean — the "PDP-from-data" curve range)."""
    cvs = np.asarray(cvs, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n_pairs = X.shape[1]
    n_cvs = cvs.shape[1]
    q_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros((n_pairs, n_cvs))
    for j in range(n_pairs):
        x = X[:, j]
        edges = np.unique(np.quantile(x, q_edges))
        if len(edges) < 3:
            continue
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1,
                       0, len(edges) - 2)
        n_b = len(edges) - 1
        counts = np.bincount(bins, minlength=n_b).astype(np.float64)
        for k in range(n_cvs):
            sums = np.bincount(bins, weights=cvs[:, k], minlength=n_b)
            mask = counts > 0
            if mask.sum() < 2:
                continue
            means = sums[mask] / counts[mask]
            out[j, k] = means.max() - means.min()
    return out


def per_residue_eta2(cvs: np.ndarray, X: np.ndarray, n_atoms: int,
                     n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Universal model-free per-residue importance via η². Returns
    ``(residue (n_atoms, n_cvs), pair (n_pairs, n_cvs))``."""
    pair = eta2_pair(cvs, X, n_bins=n_bins)
    return _pair_to_residue(pair, n_atoms), pair


def per_residue_pdp(transform_fn, X: np.ndarray, n_atoms: int,
                    top_pair_ids: np.ndarray | None = None,
                    n_grid: int = 10, n_sub: int = 500,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray] | None:
    """Model-based PDP: perturb one pair-distance at a time, query the
    model, measure normalised variance of the conditional CV mean.

    Only ``top_pair_ids`` are queried (typically the top-30 η² pairs).
    Unscored pairs contribute zero to the per-residue sum.
    """
    X = np.asarray(X, dtype=np.float64)
    n_frames, n_pairs = X.shape
    rng = np.random.default_rng(seed)
    sub = rng.choice(n_frames, size=min(n_sub, n_frames), replace=False)
    Xs = X[sub]
    try:
        z0 = np.asarray(transform_fn(Xs))
    except Exception:
        logger.exception("per_residue_pdp: probe forward pass failed.")
        return None
    if z0.ndim != 2:
        logger.warning("per_residue_pdp: transform output ndim=%d ≠ 2.", z0.ndim)
        return None
    n_cvs = z0.shape[1]
    cv_var = z0.var(axis=0)
    cv_var = np.where(cv_var > 1e-12, cv_var, 1.0)

    if top_pair_ids is None:
        top_pair_ids = np.arange(n_pairs)
    q = np.linspace(0.05, 0.95, n_grid)

    pair_pdp = np.zeros((n_pairs, n_cvs))
    for j in top_pair_ids:
        grid = np.quantile(X[:, int(j)], q)
        # batched: (n_grid * n_sub) rows, repeating Xs n_grid times
        batch = np.tile(Xs, (n_grid, 1))
        batch[:, int(j)] = np.repeat(grid, len(sub))
        try:
            z = np.asarray(transform_fn(batch))
        except Exception:
            logger.exception("per_residue_pdp: batch forward failed (pair %d).", j)
            continue
        z = z.reshape(n_grid, len(sub), n_cvs).mean(axis=1)
        for k in range(n_cvs):
            pair_pdp[int(j), k] = z[:, k].var() / cv_var[k]
    return _pair_to_residue(pair_pdp, n_atoms), pair_pdp


def per_residue_pdp_cartesian(transform_fn_3d, coords_3d: np.ndarray,
                              n_grid: int = 10, n_sub: int = 500,
                              seed: int = 42) -> np.ndarray | None:
    """Axis-sweep PDP for models whose native input is Cartesian coordinates.

    For each residue i and axis a ∈ {x, y, z}, sweep coords[:, i, a] over its
    empirical 5–95 % quantile grid on a subsample of frames; measure the
    normalised variance of the conditional CV mean; average over the three
    axes. Returns ``(n_atoms, n_cvs)``.
    """
    coords_3d = np.asarray(coords_3d, dtype=np.float32)
    n_frames, n_atoms, _ = coords_3d.shape
    rng = np.random.default_rng(seed)
    sub = rng.choice(n_frames, size=min(n_sub, n_frames), replace=False)
    Xs = coords_3d[sub].copy()
    try:
        z0 = np.asarray(transform_fn_3d(Xs))
    except Exception:
        logger.exception("per_residue_pdp_cartesian: probe forward failed.")
        return None
    if z0.ndim != 2:
        logger.warning("per_residue_pdp_cartesian: probe output ndim=%d.", z0.ndim)
        return None
    n_cvs = z0.shape[1]
    cv_var = z0.var(axis=0)
    cv_var = np.where(cv_var > 1e-12, cv_var, 1.0)

    q = np.linspace(0.05, 0.95, n_grid).astype(np.float32)
    out = np.zeros((n_atoms, n_cvs), dtype=np.float64)
    for i in range(n_atoms):
        for a in range(3):
            grid = np.quantile(coords_3d[:, i, a], q).astype(np.float32)
            batch = np.tile(Xs, (n_grid, 1, 1))
            batch[:, i, a] = np.repeat(grid, len(sub))
            try:
                z = np.asarray(transform_fn_3d(batch))
            except Exception:
                logger.exception(
                    "per_residue_pdp_cartesian: forward failed (atom %d axis %d).",
                    i, a,
                )
                continue
            z = z.reshape(n_grid, len(sub), n_cvs).mean(axis=1)
            for k in range(n_cvs):
                out[i, k] += z[:, k].var() / cv_var[k]
        out[i] /= 3.0
    return out


def plot_residue_importance(eta2_res: np.ndarray,
                            pdp_res: np.ndarray | None,
                            save_path: Path, title: str) -> None:
    """Bar chart of per-residue importance, one panel per CV. If both
    estimators are provided, η² and PDP are drawn side-by-side per residue."""
    eta2_res = np.asarray(eta2_res)
    n_atoms, n_cvs = eta2_res.shape
    has_pdp = pdp_res is not None
    fig, axes = plt.subplots(1, n_cvs, figsize=(4.5 * n_cvs, 3.2),
                             squeeze=False, sharey=False)
    axes = axes[0]
    x = np.arange(1, n_atoms + 1)
    for k, ax in enumerate(axes):
        if has_pdp:
            w = 0.4
            ax.bar(x - w / 2, eta2_res[:, k], width=w, color="C0", label="η² (data)")
            ax.bar(x + w / 2, pdp_res[:, k], width=w, color="C3", label="PDP (model)")
            if k == 0:
                ax.legend(loc="upper right", fontsize=8)
        else:
            ax.bar(x, eta2_res[:, k], color="C0")
        ax.set_xlabel("residue (Cα index)")
        if k == 0:
            ax.set_ylabel("normalised importance")
        ax.set_title(f"CV {k + 1}")
        ax.set_xlim(0.5, n_atoms + 0.5)
    fig.suptitle(title, fontsize=12, y=1.02)
    _save(fig, save_path)


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
