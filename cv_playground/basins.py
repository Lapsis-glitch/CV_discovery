"""Basin detection on 2-D CV maps and representative-structure extraction.

Two methods:
* :func:`find_basins_watershed` — steepest-descent watershed on the
  free-energy surface F = -kT ln P estimated by Gaussian KDE.
* :func:`find_basins_clustering` — HDBSCAN in CV space.

Both return a dict with per-frame ``assignments``, basin ``minima`` (or
centroids), and the index of the frame closest to each minimum.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KDE grid (shared with utils.fes_contour conceptually, kept local for clarity)
# ---------------------------------------------------------------------------

def _kde_grid(cvs: np.ndarray, grid: int = 120,
              bw: str | float = "scott", subsample: int = 50_000):
    from scipy.stats import gaussian_kde
    x, y = cvs[:, 0], cvs[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if subsample and x.size > subsample:
        rng = np.random.default_rng(0)
        sel = rng.choice(x.size, subsample, replace=False)
        xs, ys = x[sel], y[sel]
    else:
        xs, ys = x, y
    bw_v: str | float = bw
    if isinstance(bw_v, str):
        try:
            bw_v = float(bw_v)
        except ValueError:
            pass
    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw_v)
    xg = np.linspace(x.min(), x.max(), grid)
    yg = np.linspace(y.min(), y.max(), grid)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    P = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    P = P / max(P.sum(), 1e-300)
    return xg, yg, X, Y, P


# ---------------------------------------------------------------------------
# Steepest-descent watershed (pure numpy/scipy)
# ---------------------------------------------------------------------------

_NEIGHBORS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
              if not (di == 0 and dj == 0)]


def _steepest_descent_basins(F: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Assign each valid grid cell to a local minimum via steepest descent.

    Returns ``(basin_grid, minima_ij)``. Cells outside ``valid`` get -1.
    """
    H, W = F.shape
    parent = np.full((H, W), -1, dtype=np.int64)  # flat index of next cell
    minima_flat: list[int] = []

    # Pre-compute neighbour with the lowest F for each valid cell.
    # If no neighbour is strictly lower, the cell is a local minimum.
    for i in range(H):
        for j in range(W):
            if not valid[i, j]:
                continue
            best = F[i, j]
            best_ij = (i, j)
            for di, dj in _NEIGHBORS:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and valid[ni, nj]:
                    if F[ni, nj] < best:
                        best = F[ni, nj]
                        best_ij = (ni, nj)
            if best_ij == (i, j):
                minima_flat.append(i * W + j)
                parent[i, j] = i * W + j
            else:
                parent[i, j] = best_ij[0] * W + best_ij[1]

    # Path compression: follow parent pointers to a fixed point (a minimum).
    basin_of = np.full(H * W, -1, dtype=np.int64)
    min_to_id: dict[int, int] = {flat: k for k, flat in enumerate(sorted(set(minima_flat)))}

    flat_F = F.ravel()
    flat_parent = parent.ravel()
    for start in range(H * W):
        if flat_parent[start] < 0:
            continue
        node = start
        # Follow until we reach a minimum (parent == self).
        while flat_parent[node] != node:
            node = flat_parent[node]
        bid = min_to_id[node]
        # Compress.
        node2 = start
        while flat_parent[node2] != node2:
            nxt = flat_parent[node2]
            basin_of[node2] = bid
            node2 = nxt
        basin_of[node2] = bid

    basin_grid = basin_of.reshape(H, W)
    minima_ij = [(flat // W, flat % W) for flat in sorted(min_to_id.keys())]
    return basin_grid, minima_ij


def find_basins_watershed(cvs: np.ndarray, kT: float = 0.593, grid: int = 120,
                          kde_bw: str | float = "scott",
                          min_basin_frac: float = 0.01,
                          max_F: float = 10.0) -> dict | None:
    """Watershed segmentation of the FES via steepest descent. Returns
    ``None`` when no basins survive the size filter."""
    cvs = np.asarray(cvs, dtype=np.float64)
    if cvs.shape[0] < 10 or cvs.shape[1] < 2:
        return None

    xg, yg, X, Y, P = _kde_grid(cvs, grid=grid, bw=kde_bw)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(P)
    finite = np.isfinite(F)
    if not finite.any():
        return None
    F = F - F[finite].min()
    F = np.where(finite, F, np.inf)
    valid = F < max_F
    if not valid.any():
        return None

    basin_grid, minima_ij = _steepest_descent_basins(F, valid)
    if not minima_ij:
        return None

    # Map every frame to its grid cell and basin.
    x_idx = np.clip(np.searchsorted(xg, cvs[:, 0]) - 1, 0, len(xg) - 1)
    y_idx = np.clip(np.searchsorted(yg, cvs[:, 1]) - 1, 0, len(yg) - 1)
    frame_basins = basin_grid[x_idx, y_idx]

    # Drop tiny basins.
    n_total = len(cvs)
    populations = {b: int((frame_basins == b).sum()) for b in range(len(minima_ij))}
    keep = [b for b, n in populations.items() if n / n_total >= min_basin_frac]
    if not keep:
        # Keep the largest one anyway so we always return at least one rep.
        keep = [max(populations, key=populations.get)]

    relabel = {old: new for new, old in enumerate(keep)}
    new_assign = np.full(len(cvs), -1, dtype=np.int64)
    for old, new in relabel.items():
        new_assign[frame_basins == old] = new

    minima_xy: list[np.ndarray] = []
    rep_frame_idx: list[int] = []
    for old, new in relabel.items():
        i, j = minima_ij[old]
        cv_min = np.array([xg[i], yg[j]])
        minima_xy.append(cv_min)
        in_basin = np.where(new_assign == new)[0]
        if in_basin.size == 0:
            in_basin = np.arange(len(cvs))
        d = np.linalg.norm(cvs[in_basin] - cv_min, axis=1)
        rep_frame_idx.append(int(in_basin[int(d.argmin())]))

    return {
        "method": "watershed",
        "assignments": new_assign,
        "minima": np.asarray(minima_xy, dtype=np.float64),
        "rep_frame_idx": np.asarray(rep_frame_idx, dtype=np.int64),
        "n_basins": len(keep),
    }


# ---------------------------------------------------------------------------
# HDBSCAN in CV space
# ---------------------------------------------------------------------------

def find_basins_clustering(cvs: np.ndarray,
                           min_cluster_size_frac: float = 0.02,
                           min_samples: int = 10) -> dict | None:
    """HDBSCAN in CV space. Returns rep frame as the in-cluster frame closest
    to the cluster centroid."""
    cvs = np.asarray(cvs, dtype=np.float64)
    if cvs.shape[0] < 50:
        return None
    n = len(cvs)
    min_size = max(20, int(min_cluster_size_frac * n))

    labels: np.ndarray | None = None
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_size,
                         min_samples=min_samples).fit_predict(cvs)
    except Exception as exc:
        logger.warning("sklearn HDBSCAN failed (%s); trying KMeans fallback.", exc)
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(cvs)
        except Exception as exc2:
            logger.warning("KMeans fallback also failed (%s); skipping clustering.", exc2)
            return None

    unique = sorted([int(u) for u in np.unique(labels) if u >= 0])
    if not unique:
        return None
    relabel = {old: new for new, old in enumerate(unique)}
    new_assign = np.full(len(cvs), -1, dtype=np.int64)
    for old, new in relabel.items():
        new_assign[labels == old] = new

    centroids: list[np.ndarray] = []
    rep_frame_idx: list[int] = []
    for new in range(len(unique)):
        in_basin = np.where(new_assign == new)[0]
        c = cvs[in_basin].mean(axis=0)
        centroids.append(c)
        d = np.linalg.norm(cvs[in_basin] - c, axis=1)
        rep_frame_idx.append(int(in_basin[int(d.argmin())]))

    return {
        "method": "clustering",
        "assignments": new_assign,
        "minima": np.asarray(centroids, dtype=np.float64),
        "rep_frame_idx": np.asarray(rep_frame_idx, dtype=np.int64),
        "n_basins": len(unique),
    }


# ---------------------------------------------------------------------------
# PDB writer for representative structures
# ---------------------------------------------------------------------------

def write_basin_pdbs(basins: dict, coords_3d: np.ndarray,
                     top_path: str, selection: str,
                     tag: str, out_dir: str | Path,
                     protein_coords_3d: np.ndarray | None = None,
                     protein_selection: str = "protein") -> list[Path]:
    """Write one PDB per basin as ``<tag>_<cv1>_<cv2>_<id>_rep.pdb``.

    Prefers the full-protein coords when available so the rep PDB contains
    the entire protein (heavy + H atoms in the ``protein`` selection); falls
    back to the CV-selection coords (typically Cα-only) otherwise.
    Loads only the topology with MDAnalysis to avoid re-reading the
    trajectory.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_full = (protein_coords_3d is not None
                and protein_coords_3d.ndim == 3
                and protein_coords_3d.shape[1] > 0
                and protein_coords_3d.shape[0] == coords_3d.shape[0])

    u = mda.Universe(top_path)
    if use_full:
        write_sel = protein_selection
        write_coords = protein_coords_3d
    else:
        write_sel = selection
        write_coords = coords_3d

    ag = u.select_atoms(write_sel)
    if len(ag) != write_coords.shape[1]:
        logger.warning(
            "Selection '%s' has %d atoms but cached coords have %d — "
            "skipping basin PDB writing for %s.",
            write_sel, len(ag), write_coords.shape[1], tag,
        )
        return []
    # Topology-only universes have no Timestep — attach a placeholder.
    if not hasattr(u, "trajectory") or getattr(u, "trajectory", None) is None \
            or not hasattr(u.trajectory, "ts"):
        init_pos = np.zeros((1, len(u.atoms), 3), dtype=np.float32)
        u.trajectory = MemoryReader(init_pos, order="fac")

    written: list[Path] = []
    for k in range(basins["n_basins"]):
        local_frame = int(basins["rep_frame_idx"][k])
        cv1, cv2 = (float(x) for x in basins["minima"][k])
        ag.positions = write_coords[local_frame].astype(np.float32)
        path = out_dir / f"{tag}_{cv1:.3f}_{cv2:.3f}_{k}_rep.pdb"
        ag.write(str(path))
        written.append(path)
    logger.info("Wrote %d %s reps for %s → %s",
                len(written), basins["method"], tag, out_dir)
    return written


def save_basin_summary(basins_ws: dict | None, basins_cl: dict | None,
                       save_path: str | Path) -> None:
    """Write a JSON snapshot of basin minima / centroids and rep frame indices."""
    import json
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for key, b in (("watershed", basins_ws), ("clustering", basins_cl)):
        if b is None:
            out[key] = None
            continue
        out[key] = {
            "n_basins": int(b["n_basins"]),
            "minima": b["minima"].tolist(),
            "rep_frame_idx": b["rep_frame_idx"].tolist(),
        }
    save_path.write_text(json.dumps(out, indent=2))
