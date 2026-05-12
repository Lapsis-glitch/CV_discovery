"""Shared utilities — CLI parsing, plotting, timing, device detection, HTML summary."""

from __future__ import annotations

import argparse
import base64
import logging
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – must come before pyplot import
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ───────────────────────────── logging ──────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ───────────────────────────── CLI ──────────────────────────────────────────

def base_argparser(description: str = "") -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with every common flag."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--top",       required=True,  help="Topology file (PDB/PSF/PRMTOP/…)")
    p.add_argument("--traj",      required=True,  help="Trajectory file (DCD/XTC/TRR/NC/…)")
    p.add_argument("--stride",    type=int, default=1,   help="Frame stride  (default 1)")
    p.add_argument("--selection", default="protein and name CA",
                   help="MDAnalysis atom selection  (default 'protein and name CA')")
    p.add_argument("--seed",      type=int, default=42,  help="Random seed  (default 42)")
    p.add_argument("--lag",       type=int, default=100,
                   help="Lag time in *strided* frames for tICA / time-lagged methods "
                        "(default 100 — equals 20 ns with --dt-ps 200 and --stride 1, "
                        "matching the Trp-cage lag used in Ghorbani–Hoffmann–Ferguson 2022 "
                        "(GraphVAMPNet) and Bonati–Piccini–Parrinello 2021 (Deep-TICA).")
    p.add_argument("--dt-ps",     type=float, default=200.0, dest="dt_ps",
                   help="Time between saved frames in ps  (default 200 — DESRES Trp-cage).")
    p.add_argument("--native",    default=None,
                   help="Native structure file (PDB/GRO/…). If omitted, frame 0 is used.")
    p.add_argument("--labels",    default=None,
                   help="Path to .npy with integer state labels  (optional)")
    p.add_argument("--epochs",    type=int, default=500,
                   help="Max training epochs — acts as a cap; actual length is "
                        "governed by early stopping (default 500)")
    p.add_argument("--patience",  type=int, default=20,
                   help="Early-stopping patience in epochs with no improvement "
                        "on the monitored loss (default 20)")
    p.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta",
                   help="Minimum improvement in monitored loss to reset "
                        "early-stopping patience (default 1e-4)")
    p.add_argument("--batch-size",type=int, default=32,  dest="batch_size",
                   help="Batch size  (default 32)")
    p.add_argument("--output-dir",default="outputs",     dest="output_dir",
                   help="Root output directory  (default outputs)")
    p.add_argument("--cache-dir", default="outputs/cache", dest="cache_dir",
                   help="Directory for the featurized-trajectory cache "
                        "(.npz). When --read is on and a cache file exists, "
                        "MDAnalysis is skipped entirely.  (default outputs/cache)")
    p.add_argument("--read",      action="store_true",
                   help="Skip CV training; reload each method's cached .npy "
                        "from --output-dir and regenerate the plots, metrics, "
                        "and summary.html only.  Also enables the featurized-"
                        "trajectory cache load from --cache-dir (skips MDA).")
    p.add_argument("--fes-method", choices=("kde", "hist"), default="kde",
                   dest="fes_method",
                   help="Density estimator for the free-energy-landscape "
                        "contour plot. 'kde' (default) uses Gaussian KDE on a "
                        "fine grid — smooth P(x, y) with no empty-bin "
                        "artifacts. 'hist' falls back to a 2-D histogram.")
    p.add_argument("--cluster-samples", type=int, default=50,
                   dest="cluster_samples",
                   help="Per-cluster frame samples written as DCD trajectories "
                        "alongside the basin representative PDBs. Each DCD "
                        "contains a random (seeded by --seed) subsample of the "
                        "frames assigned to that cluster, with full TRP-cage "
                        "atoms when protein coords are cached. 0 disables. "
                        "(default 50)")
    p.add_argument("--kde-bw", default="0.25", dest="kde_bw",
                   help="Gaussian-KDE bandwidth for the FES contour plot. "
                        "Pass a float (smaller = sharper basins, larger = "
                        "smoother) or the string 'scott' / 'silverman' for "
                        "scipy's automatic rules. Default 0.25 — narrower "
                        "than scipy's 'scott' default so basins are crisper.")
    return p


# ───────────────────────────── seeding ──────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ───────────────────────────── device ───────────────────────────────────────

def get_device():
    """Return a ``torch.device`` – CUDA when available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        else:
            dev = torch.device("cpu")
            logger.info("CUDA not available – using CPU.")
        return dev
    except ImportError:
        logger.warning("PyTorch not installed – device detection skipped.")
        return None


# ───────────────────────────── lag-time logging ─────────────────────────────

def effective_tau_ns(args: argparse.Namespace) -> float:
    """τ in ns, accounting for stride and frame spacing."""
    return float(args.lag) * float(args.stride) * float(args.dt_ps) / 1000.0


def log_effective_tau(args: argparse.Namespace, method_hint: str = "") -> None:
    tau = effective_tau_ns(args)
    suffix = f"  [{method_hint}]" if method_hint else ""
    logger.info(
        "τ = lag(%d frames) × stride(%d) × dt(%.1f ps) = %.2f ns%s",
        args.lag, args.stride, args.dt_ps, tau, suffix,
    )


# ───────────────────────────── labels ───────────────────────────────────────

def get_q_labels(
    q: np.ndarray, cuts: tuple[float, float] = (0.3, 0.7),
) -> np.ndarray:
    """Map fraction-of-native-contacts Q to {0=unfolded, 1=intermediate, 2=folded}.

    Thresholds follow common practice for small fast-folders.
    """
    lo, hi = cuts
    labels = np.full(len(q), 1, dtype=np.int64)
    labels[q < lo] = 0
    labels[q >= hi] = 2
    return labels


def get_or_make_labels(
    args: argparse.Namespace,
    feat_or_rmsd,
    n_clusters: int = 3,
) -> np.ndarray:
    """Return physically-motivated state labels for supervised CV methods.

    Priority:
      1. ``--labels`` file if provided.
      2. Q-based (fraction of native contacts) labels if Q is in
         ``feat.colorings`` and yields at least two non-empty classes.
      3. Fallback: k-means on Cα-RMSD (the legacy placeholder).

    ``feat_or_rmsd`` may be a :class:`TrajectoryFeatures` instance **or** a
    raw 1-D RMSD array (back-compat).
    """
    if args.labels is not None:
        labels = np.load(args.labels)
        logger.info("Loaded %d labels from %s", len(labels), args.labels)
        return labels

    # Accept either a full features container or a bare RMSD array.
    q = None
    rmsd = None
    if hasattr(feat_or_rmsd, "colorings"):
        q = feat_or_rmsd.colorings.get("Q (native contacts)")
        rmsd = feat_or_rmsd.rmsd
    else:
        rmsd = np.asarray(feat_or_rmsd)

    if q is not None:
        labels = get_q_labels(q)
        counts = np.bincount(labels, minlength=3)
        if (counts > 0).sum() >= 2:
            logger.info(
                "Q-based labels  unfolded=%d  intermediate=%d  folded=%d  "
                "(cuts Q<0.3 / 0.3≤Q<0.7 / Q≥0.7)",
                counts[0], counts[1], counts[2],
            )
            return labels
        logger.warning(
            "Q cuts produced only one populated class (%s) – "
            "falling back to k-means on RMSD.", counts.tolist(),
        )

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(np.asarray(rmsd).reshape(-1, 1))
    logger.info(
        "Generated %d-cluster k-means labels from RMSD (placeholder).", n_clusters,
    )
    return labels


# ───────────────────────────── early stopping ──────────────────────────────

class EarlyStopping:
    """Minimal early-stopping helper for pure-PyTorch training loops.

    Call :meth:`step` once per epoch with the monitored loss; it returns
    ``True`` when patience is exhausted and training should break.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4,
                 name: str = "model"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.name = name
        self.best: float = float("inf")
        self.bad_epochs = 0
        self.stopped_epoch: int | None = None

    def step(self, loss: float, epoch: int) -> bool:
        if loss < self.best - self.min_delta:
            self.best = float(loss)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            logger.info(
                "  %s early-stopping at epoch %d (best loss=%.6f, no improvement for %d epochs)",
                self.name, epoch + 1, self.best, self.patience,
            )
            return True
        return False


# ───────────────────────────── timing ───────────────────────────────────────

@contextmanager
def timer(method_name: str):
    """Context manager that logs wall-clock time for *method_name*."""
    start = time.perf_counter()
    logger.info("▶ %s – started", method_name)
    yield
    elapsed = time.perf_counter() - start
    logger.info("✔ %s – finished in %.2f s", method_name, elapsed)


# ───────────────────────────── committor ───────────────────────────────────

COMMITTOR_CSV = Path("/home/rat/Nancy_D/GNN/TRP_gnn/trp_k10.0.csv")


def load_committor(frame_indices: np.ndarray,
                   csv_path: Path | str = COMMITTOR_CSV) -> np.ndarray | None:
    """Read the ``committor`` column from ``csv_path`` and align it to the
    selected trajectory frames. Returns ``None`` if the file is missing or
    indexing would fall out of range."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Committor CSV not found: %s", csv_path)
        return None
    committor = np.loadtxt(
        csv_path, delimiter=",", skiprows=1, usecols=9, dtype=np.float64,
    )
    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.size == 0 or idx.max() >= committor.size:
        logger.warning(
            "Committor length %d < max frame index %d — skipping.",
            committor.size, int(idx.max()) if idx.size else -1,
        )
        return None
    return committor[idx]


def attach_committor(feat, csv_path: Path | str = COMMITTOR_CSV) -> None:
    """Inject committor into ``feat.colorings`` as key ``"Committor"`` when
    available — a no-op otherwise."""
    committor = load_committor(feat.frame_indices, csv_path)
    if committor is None:
        return
    feat.colorings["Committor"] = committor
    logger.info(
        "Committor loaded: n=%d  mean=%.3f  std=%.3f",
        committor.size, float(np.nanmean(committor)), float(np.nanstd(committor)),
    )


# ───────────────────────────── plotting ─────────────────────────────────────

def scatter_cv(
    cvs: np.ndarray,
    colorings: dict | np.ndarray | None,
    title: str,
    save_path: str | Path,
    gridsize: int = 40,
    min_count: int = 1,
) -> None:
    """Hexbin CV map — per coloring, one column with two rows:
    top = per-bin mean, bottom = per-bin standard deviation.

    ``colorings`` may be:
      * a dict ``{label: (n,) array}`` → one column per entry,
      * a 1-D array  → single column,
      * ``None``     → coloured by frame index.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = save_path.with_suffix(".svg")

    if colorings is None:
        colorings = {"frame index": np.arange(len(cvs))}
    elif isinstance(colorings, np.ndarray):
        colorings = {"value": colorings}

    ncols = len(colorings)
    fig, axes = plt.subplots(
        2, ncols, figsize=(4.1 * ncols, 8.0), squeeze=False,
    )
    x, y = cvs[:, 0], cvs[:, 1]

    for col, (label, vals) in enumerate(colorings.items()):
        vals = np.asarray(vals, dtype=np.float64)
        # Scale factor used to normalize the std panel: relative-std = σ / range.
        vrange = float(np.nanmax(vals) - np.nanmin(vals))
        scale = vrange if vrange > 0 else 1.0
        is_committor = label.lower().startswith("committor")
        for row, (reducer, stat_name) in enumerate(
            ((np.mean, "mean"), (np.std, "std")),
        ):
            ax = axes[row, col]
            hb_kwargs: dict = dict(
                gridsize=gridsize,
                mincnt=min_count,
                cmap="viridis",
                reduce_C_function=reducer,
            )
            if is_committor and stat_name == "mean":
                # Diverging blue→white→red around 0.5 for committor values in [0, 1].
                hb_kwargs["cmap"] = "bwr"
                hb_kwargs["vmin"] = 0.0
                hb_kwargs["vmax"] = 1.0
            if stat_name == "std":
                # Normalize σ by the coloring's range, then clip colorbar at 10 %.
                hb_kwargs["reduce_C_function"] = lambda a, s=scale: np.std(a) / s
                hb_kwargs["cmap"] = "magma"
                hb_kwargs["vmin"] = 0.0
                hb_kwargs["vmax"] = 0.1
            hb = ax.hexbin(x, y, C=vals, **hb_kwargs)
            cbar = fig.colorbar(hb, ax=ax, shrink=0.85, extend="max" if stat_name == "std" else "neither")
            cbar_label = (f"{label} (σ / range, clipped at 10 %)"
                          if stat_name == "std" else f"{label} (mean)")
            cbar.set_label(cbar_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            ax.set_xlabel("CV 1")
            ax.set_ylabel("CV 2")
            title = (f"{label} — relative σ (norm., ≤10 %)"
                     if stat_name == "std" else f"{label} — mean")
            ax.set_title(title, fontsize=10)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(svg_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", svg_path)


def fes_contour(
    cvs: np.ndarray,
    save_path: str | Path,
    bins: int = 60,
    kT: float = 0.593,
    title: str = "Free energy landscape",
    max_F: float | None = 10.0,
    method: str = "kde",
    kde_grid: int = 120,
    kde_bw: str | float = "scott",
    kde_subsample: int = 50_000,
    rep_points: dict | None = None,
) -> None:
    """Plot the free energy landscape F(CV1, CV2) = -kT ln P(CV1, CV2) as a
    filled contour. The minimum is shifted to 0 and the colorbar is clipped
    above ``max_F`` kcal/mol (default 10).

    ``method``:
      * ``"kde"`` (default) — Gaussian KDE on a ``kde_grid``×``kde_grid`` mesh;
        smooth P(x, y) with no empty-bin artifacts. ``kde_subsample`` caps the
        number of samples passed to ``scipy.stats.gaussian_kde`` for speed.
      * ``"hist"`` — classical 2-D histogram with ``bins`` per axis.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = save_path.with_suffix(".svg")

    x, y = cvs[:, 0], cvs[:, 1]
    finite_xy = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite_xy], y[finite_xy]

    if method == "kde":
        from scipy.stats import gaussian_kde
        if kde_subsample and x.size > kde_subsample:
            rng = np.random.default_rng(0)
            sel = rng.choice(x.size, size=kde_subsample, replace=False)
            xs, ys = x[sel], y[sel]
        else:
            xs, ys = x, y
        bw: str | float = kde_bw
        if isinstance(bw, str):
            try:
                bw = float(bw)
            except ValueError:
                pass          # keep as 'scott' / 'silverman' / etc.
        kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw)
        xg = np.linspace(x.min(), x.max(), kde_grid)
        yg = np.linspace(y.min(), y.max(), kde_grid)
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        P = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        P = P / P.sum()
        with np.errstate(divide="ignore"):
            F = -kT * np.log(P)
    elif method == "hist":
        H, xe, ye = np.histogram2d(x, y, bins=bins)
        P = H / H.sum() if H.sum() > 0 else H
        with np.errstate(divide="ignore"):
            F = -kT * np.log(P)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        X, Y = np.meshgrid(xc, yc, indexing="ij")
    else:
        raise ValueError(f"fes_contour: unknown method {method!r} (expected 'kde' or 'hist').")

    finite = np.isfinite(F)
    if finite.any():
        F -= F[finite].min()
    F = np.where(finite, F, np.nan)
    if max_F is not None:
        F = np.where(F > max_F, np.nan, F)
    F = np.ma.masked_invalid(F)

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    top = float(max_F) if max_F is not None else float(np.nanmax(F))
    step = 0.25
    levels = np.arange(0.0, top + step, step)
    cf = ax.contourf(X, Y, F, levels=levels, cmap="viridis", extend="max")
    ax.contour(X, Y, F, levels=levels, colors="k", linewidths=0.8, alpha=0.7)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label("Free energy (kcal/mol)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("CV 1")
    ax.set_ylabel("CV 2")
    ax.set_title(title, fontsize=11)
    # Overlay basin representative-structure markers as diamonds.
    if rep_points:
        styles = {
            "watershed": dict(facecolor="white", edgecolor="black",
                              label="watershed rep"),
            "clustering": dict(facecolor="gold",  edgecolor="black",
                               label="clustering rep"),
        }
        for key, pts in rep_points.items():
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts)
            style = styles.get(key, dict(facecolor="red", edgecolor="black",
                                         label=f"{key} rep"))
            ax.scatter(pts[:, 0], pts[:, 1], marker="D", s=90,
                       linewidths=1.2, zorder=10, **style)
        if any(v is not None and len(v) for v in rep_points.values()):
            ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # Keep the full data range visible so the FES isn't cropped.
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.margins(0)
    fig.savefig(str(svg_path), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Saved plot → %s", svg_path)


# ───────────────────────────── publication panel ───────────────────────────

# Paul Tol "iridescent" colormap (23 stops, ends near-white).
# Source: Paul Tol's "Colour Schemes" technical note (SRON/TN/2018-003).
_TOL_IRIDESCENT_HEX = [
    "#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF", "#D0E7CA",
    "#C2E3D2", "#B5DDD8", "#A8D8DC", "#9BD2E1", "#8DCBE4", "#81C4E7",
    "#7BBCE7", "#7EB2E4", "#88A5DD", "#9398D2", "#9B8AC4", "#9D7DB2",
    "#9A709E", "#906388", "#805770", "#684957", "#46353A",
]


def _tol_iridescent_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # Reverse so highest values map to the near-white end (#FEFBE9).
    return LinearSegmentedColormap.from_list(
        "tol_iridescent", list(reversed(_TOL_IRIDESCENT_HEX)), N=256,
    )


_PUB_RC = {
    "font.family":     "serif",
    "font.serif":      ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size":       28,
    "axes.labelsize":  32,
    "axes.titlesize":  32,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 24,
}


def _hexbin_coloring(ax, x, y, vals, cmap, *, vmin=None, vmax=None,
                     gridsize=45, mincnt=1):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals)
    if not finite.any():
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return None
    return ax.hexbin(
        x[finite], y[finite], C=vals[finite],
        gridsize=gridsize, mincnt=mincnt, cmap=cmap,
        reduce_C_function=np.mean, vmin=vmin, vmax=vmax,
    )


def _kde_F(x, y, kT=0.593, kde_grid=120, kde_bw="scott", kde_subsample=50_000):
    from scipy.stats import gaussian_kde
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 5:
        return None
    if kde_subsample and x.size > kde_subsample:
        rng = np.random.default_rng(0)
        sel = rng.choice(x.size, size=kde_subsample, replace=False)
        xs, ys = x[sel], y[sel]
    else:
        xs, ys = x, y
    bw = kde_bw
    if isinstance(bw, str):
        try:
            bw = float(bw)
        except ValueError:
            pass
    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw)
    xg = np.linspace(x.min(), x.max(), kde_grid)
    yg = np.linspace(y.min(), y.max(), kde_grid)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    P = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    P = P / max(P.sum(), 1e-300)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(P)
    finite_F = np.isfinite(F)
    if finite_F.any():
        F = F - F[finite_F].min()
    return X, Y, F


def publication_panel(
    cvs: np.ndarray,
    colorings: dict,
    save_path: str | Path,
    kde_bw: str | float = "scott",
    kT: float = 0.593,
    max_F: float = 10.0,
) -> None:
    """One compound figure per analysis: top row of three hexbin maps
    (RMSD, Q, helicity 2–8) sharing CV2 axis; bottom panel ~1.5× size,
    left-aligned, hexbin of committor with KDE-PMF contour lines overlaid."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = save_path.with_suffix(".svg")
    png_path = save_path.with_suffix(".png")

    rmsd_key = "Cα-RMSD to native (Å)"
    q_key    = "Q (native contacts)"
    hel_key  = "Helicity (res 2–8)"
    com_key  = "Committor"

    rmsd = np.asarray(colorings.get(rmsd_key, np.full(len(cvs), np.nan)), dtype=np.float64)
    qval = np.asarray(colorings.get(q_key,    np.full(len(cvs), np.nan)), dtype=np.float64)
    hel  = np.asarray(colorings.get(hel_key,  np.full(len(cvs), np.nan)), dtype=np.float64)
    com  = colorings.get(com_key)
    com  = (np.asarray(com, dtype=np.float64) if com is not None
            else np.full(len(cvs), np.nan))

    x, y = cvs[:, 0], cvs[:, 1]
    cmap_top = "YlOrBr"

    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    cbar_label_size = max(1, int(round(_PUB_RC["axes.labelsize"] * 8 / 9)))
    cbar_tick_size  = max(1, int(round(_PUB_RC["xtick.labelsize"] * 8 / 9)))

    # Fixed-size axes in figure fractions: every panel has identical physical
    # dimensions regardless of CV data range (no dynamic scaling). Display
    # squareness comes from setting axes width = height in figure-fractions
    # given figsize aspect, and aspect="auto" so data fills the box.
    figsize = (20.0, 15.0)
    fw, fh = figsize
    # Top tile: 4×4 in plot + 0.2 in cbar
    top_plot_w_in, top_plot_h_in = 4.0, 4.0
    top_cbar_w_in, top_cbar_pad_in = 0.20, 0.18
    bot_plot_w_in, bot_plot_h_in = 7.0, 7.0
    bot_cbar_w_in, bot_cbar_pad_in = 0.35, 0.25
    # Convert to fig fractions
    tpw, tph = top_plot_w_in / fw, top_plot_h_in / fh
    tcw, tcp = top_cbar_w_in / fw, top_cbar_pad_in / fw
    bpw, bph = bot_plot_w_in / fw, bot_plot_h_in / fh
    bcw, bcp = bot_cbar_w_in / fw, bot_cbar_pad_in / fw
    left0 = 0.10
    top_panel_gap_in = 1.6
    tgap = top_panel_gap_in / fw
    top_y0 = 0.68
    bot_y0 = 0.10

    with plt.rc_context(_PUB_RC):
        fig = plt.figure(figsize=figsize)

        def _add_panel(x0, y0, pw, ph, cw, cp):
            ax = fig.add_axes([x0, y0, pw, ph])
            cax = fig.add_axes([x0 + pw + cp, y0, cw, ph])
            return ax, cax

        x_top0 = left0
        x_top1 = x_top0 + tpw + tcw + tcp + tgap
        x_top2 = x_top1 + tpw + tcw + tcp + tgap
        ax1, cax1 = _add_panel(x_top0, top_y0, tpw, tph, tcw, tcp)
        ax2, cax2 = _add_panel(x_top1, top_y0, tpw, tph, tcw, tcp)
        ax3, cax3 = _add_panel(x_top2, top_y0, tpw, tph, tcw, tcp)
        ax_b, cax_b = _add_panel(left0, bot_y0, bpw, bph, bcw, bcp)

        top_specs = [
            (ax1, cax1, rmsd, rmsd_key, None, None, True),
            (ax2, cax2, qval, q_key,    0.0,  1.0,  False),
            (ax3, cax3, hel,  hel_key,  0.0,  1.0,  False),
        ]
        for ax, cax, vals, label, vmin, vmax, force_5_cbar in top_specs:
            hb = _hexbin_coloring(ax, x, y, vals, cmap_top, vmin=vmin, vmax=vmax)
            if hb is not None:
                cbar = fig.colorbar(hb, cax=cax)
                cbar.set_label(label, fontsize=cbar_label_size)
                cbar.ax.tick_params(labelsize=cbar_tick_size)
                if force_5_cbar:
                    lo, hi = hb.get_clim()
                    cbar.set_ticks(np.linspace(lo, hi, 5))
                    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                else:
                    cbar.locator = MaxNLocator(5)
                    cbar.update_ticks()
            else:
                cax.set_visible(False)
            ax.set_xlabel("CV 1")
            ax.tick_params(axis="x", labelrotation=30)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
        ax1.set_ylabel("CV 2")
        # Force matching y-limits so hidden ticks on ax2/ax3 match ax1 visually.
        for ax in (ax2, ax3):
            ax.sharey(ax1)
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.set_ylabel("")

        # Bottom: committor hexbin + KDE-PMF contour lines with kcal/mol labels.
        hb_c = _hexbin_coloring(ax_b, x, y, com, "bwr", vmin=0.0, vmax=1.0,
                                gridsize=55)
        if hb_c is not None:
            cbar = fig.colorbar(hb_c, cax=cax_b)
            cbar.set_label("Committor", fontsize=cbar_label_size)
            cbar.ax.tick_params(labelsize=cbar_tick_size)
            cbar.set_ticks(np.linspace(0.0, 1.0, 5))
        else:
            cax_b.set_visible(False)
        kde_out = _kde_F(x, y, kT=kT, kde_bw=kde_bw)
        if kde_out is not None:
            import matplotlib.patheffects as pe
            X, Y, F = kde_out
            F_masked = np.where((F <= max_F) & np.isfinite(F), F, np.nan)
            levels = np.arange(1.0, max_F + 0.001, 1.0)
            cs = ax_b.contour(X, Y, F_masked, levels=levels,
                              colors="k", linewidths=2.2, alpha=0.9)
            labels_c = ax_b.clabel(cs, inline=True, fontsize=22, fmt="%.0f")
            for txt in labels_c:
                txt.set_path_effects([
                    pe.Stroke(linewidth=3.0, foreground="white"),
                    pe.Normal(),
                ])
        # Cluster representatives — diamond markers numbered 1..N.
        try:
            from cv_playground.basins import find_basins_clustering
            basins_cl = find_basins_clustering(cvs)
        except Exception:
            basins_cl = None
        if basins_cl is not None and basins_cl.get("n_basins", 0) > 0:
            import matplotlib.patheffects as pe
            mins = np.asarray(basins_cl["minima"], dtype=np.float64)
            ax_b.scatter(mins[:, 0], mins[:, 1], marker="D", s=900,
                         facecolor="gold", edgecolor="black",
                         linewidths=2.0, zorder=10)
            for k, (cx, cy) in enumerate(mins, start=1):
                t = ax_b.text(cx, cy, str(k), ha="center", va="center",
                              fontsize=20, fontweight="bold",
                              color="black", zorder=11)
                t.set_path_effects([
                    pe.Stroke(linewidth=2.5, foreground="white"),
                    pe.Normal(),
                ])

        ax_b.set_xlabel("CV 1")
        ax_b.set_ylabel("CV 2")
        ax_b.tick_params(axis="x", labelrotation=30)
        ax_b.xaxis.set_major_locator(MaxNLocator(5))
        ax_b.yaxis.set_major_locator(MaxNLocator(5))

        fig.savefig(str(svg_path))
        fig.savefig(str(png_path), dpi=200)
        plt.close(fig)
    logger.info("Saved publication panel → %s", svg_path)


def scatter_rg_q_by_cv(
    cvs: np.ndarray,
    q: np.ndarray,
    rg: np.ndarray,
    title: str,
    save_path: str | Path,
    gridsize: int = 40,
    min_count: int = 1,
) -> None:
    """Hexbin of (Q, Rg) coloured by each CV component — one column per CV,
    top row = per-bin mean(CV), bottom row = per-bin σ(CV) normalized by the
    CV's value range and clipped at 10 %. Produces 2×2 = 4 panels."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = save_path.with_suffix(".svg")

    q = np.asarray(q, dtype=np.float64)
    rg = np.asarray(rg, dtype=np.float64)
    n_cv = min(2, cvs.shape[1])
    fig, axes = plt.subplots(2, n_cv, figsize=(4.1 * n_cv, 8.0), squeeze=False)

    for col in range(n_cv):
        vals = np.asarray(cvs[:, col], dtype=np.float64)
        vrange = float(np.nanmax(vals) - np.nanmin(vals))
        scale = vrange if vrange > 0 else 1.0
        for row, stat_name in enumerate(("mean", "std")):
            ax = axes[row, col]
            hb_kwargs: dict = dict(
                gridsize=gridsize, mincnt=min_count, cmap="viridis",
            )
            if stat_name == "mean":
                hb_kwargs["reduce_C_function"] = np.mean
            else:
                hb_kwargs["reduce_C_function"] = lambda a, s=scale: np.std(a) / s
                hb_kwargs["vmin"] = 0.0
                hb_kwargs["vmax"] = 0.1
            hb = ax.hexbin(q, rg, C=vals, **hb_kwargs)
            cbar = fig.colorbar(
                hb, ax=ax, shrink=0.85,
                extend="max" if stat_name == "std" else "neither",
            )
            cbar_label = (f"CV{col+1} (σ / range, clipped at 10 %)"
                          if stat_name == "std" else f"CV{col+1} (mean)")
            cbar.set_label(cbar_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            ax.set_xlabel("Q (native contacts)")
            ax.set_ylabel("Radius of gyration (Å)")
            sub = ("relative σ (norm., ≤10 %)" if stat_name == "std" else "mean")
            ax.set_title(f"CV{col+1} — {sub}", fontsize=10)

    fig.suptitle(f"{title} — CV on (Q, Rg)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(svg_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Rg-vs-Q plot → %s", svg_path)


def save_cv(cvs: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), cvs)
    logger.info("Saved CV array → %s", save_path)


# ───────────────────────── run context (metrics) ────────────────────────────

_RUN_CTX: dict = {"feat": None, "args": None, "interpret_mode": None}


def set_run_context(feat=None, args=None, interpret_mode: str | None = None) -> None:
    """Attach the current features + CLI args so :func:`run_method` can score
    and optionally emit an interpretability panel.

    ``interpret_mode``:
      * ``"pearson"`` – auto-generate a Pearson pseudo-loading contact map
        (CV vs every pairwise Cα distance) for every method.
      * ``None``      – no auto interpret step (the script handles it manually,
        e.g. gradient-based for torch/GNN models).
    """
    _RUN_CTX["feat"] = feat
    _RUN_CTX["args"] = args
    _RUN_CTX["interpret_mode"] = interpret_mode


# ───────────────────────── safe method runner ───────────────────────────────

def run_method(
    name: str,
    func: Callable[[], np.ndarray],
    colorings: dict | np.ndarray | None,
    out_dir: Path,
) -> Optional[np.ndarray]:
    """Execute *func*, save multi-panel plot + .npy, score metrics if a run
    context is set, catch & log any exception."""
    try:
        tag = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        npy_path = Path(out_dir) / f"{tag}.npy"
        args_ctx = _RUN_CTX.get("args")
        read_mode = bool(args_ctx is not None and getattr(args_ctx, "read", False))
        if read_mode:
            if not npy_path.exists():
                logger.warning("%s – cached %s not found; skipping (--read).",
                               name, npy_path)
                return None
            logger.info("▶ %s – loading cached CV from %s", name, npy_path)
            cvs = np.load(str(npy_path))
        else:
            with timer(name):
                result = func()
        transform_fn = None
        pdp_kind = "distances"
        if read_mode:
            pass
        elif isinstance(result, tuple) and len(result) == 2:
            cvs, transform_fn = result
        elif isinstance(result, tuple) and len(result) == 3:
            cvs, transform_fn, pdp_kind = result
        else:
            cvs = result
        if cvs is None or cvs.ndim < 2 or cvs.shape[1] < 2:
            logger.warning("%s returned invalid CV array – skipping.", name)
            return None
        # Drop any trailing cluster-id columns from a previous --read load
        # before passing to plotting / scoring (first two columns are the CVs).
        cvs_full = cvs
        cvs = cvs_full[:, :2]
        scatter_cv(cvs, colorings, name, out_dir / f"{tag}.svg")

        # ── basin detection + representative-structure extraction ──────────
        basins_ws = basins_cl = None
        rep_pts: dict[str, np.ndarray] = {}
        try:
            from cv_playground.basins import (
                find_basins_watershed, find_basins_clustering,
                write_basin_pdbs, write_basin_cluster_dcds,
                save_basin_summary,
            )
            kde_bw_val = getattr(args_ctx, "kde_bw", "scott") if args_ctx else "scott"
            basins_ws = find_basins_watershed(cvs, kde_bw=kde_bw_val)
            basins_cl = find_basins_clustering(cvs)
            if basins_ws is not None:
                rep_pts["watershed"] = basins_ws["minima"]
            if basins_cl is not None:
                rep_pts["clustering"] = basins_cl["minima"]
            top_path = getattr(args_ctx, "top", None) if args_ctx else None
            sel = getattr(args_ctx, "selection", "protein and name CA") \
                if args_ctx else "protein and name CA"
            feat_for_pdb = _RUN_CTX.get("feat")
            if top_path and feat_for_pdb is not None:
                prot_coords = getattr(feat_for_pdb, "protein_coords_3d", None)
                prot_sel = getattr(feat_for_pdb, "protein_selection", "protein")
                if basins_ws is not None:
                    write_basin_pdbs(basins_ws, feat_for_pdb.coords_3d,
                                     top_path, sel, tag,
                                     Path(out_dir) / "basins" / "watershed",
                                     protein_coords_3d=prot_coords,
                                     protein_selection=prot_sel)
                if basins_cl is not None:
                    write_basin_pdbs(basins_cl, feat_for_pdb.coords_3d,
                                     top_path, sel, tag,
                                     Path(out_dir) / "basins" / "clustering",
                                     protein_coords_3d=prot_coords,
                                     protein_selection=prot_sel)
                n_samples = int(getattr(args_ctx, "cluster_samples", 50)
                                if args_ctx else 50)
                seed = int(getattr(args_ctx, "seed", 42) if args_ctx else 42)
                if n_samples > 0:
                    if basins_ws is not None:
                        write_basin_cluster_dcds(
                            basins_ws, feat_for_pdb.coords_3d,
                            top_path, sel, tag,
                            Path(out_dir) / "basins" / "watershed",
                            protein_coords_3d=prot_coords,
                            protein_selection=prot_sel,
                            n_samples=n_samples, seed=seed,
                        )
                    if basins_cl is not None:
                        write_basin_cluster_dcds(
                            basins_cl, feat_for_pdb.coords_3d,
                            top_path, sel, tag,
                            Path(out_dir) / "basins" / "clustering",
                            protein_coords_3d=prot_coords,
                            protein_selection=prot_sel,
                            n_samples=n_samples, seed=seed,
                        )
            save_basin_summary(basins_ws, basins_cl,
                               Path(out_dir) / "basins" / f"{tag}_basins.json")
            # Sidecar: per-frame cluster IDs from both detectors
            # columns = [watershed_id, hdbscan_id]; -1 = unassigned/noise
            n_frames = cvs.shape[0]
            clusters = np.full((n_frames, 2), -1, dtype=np.int64)
            if basins_ws is not None and "assignments" in basins_ws:
                a = np.asarray(basins_ws["assignments"])
                if a.shape[0] == n_frames:
                    clusters[:, 0] = a
            if basins_cl is not None and "assignments" in basins_cl:
                a = np.asarray(basins_cl["assignments"])
                if a.shape[0] == n_frames:
                    clusters[:, 1] = a
            clusters_path = Path(out_dir) / "basins" / f"{tag}_clusters.npy"
            clusters_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(clusters_path), clusters)
        except Exception:
            logger.warning("Basin detection failed for %s:\n%s",
                           name, traceback.format_exc())

        try:
            fes_method = getattr(args_ctx, "fes_method", "kde") if args_ctx else "kde"
            fes_kde_bw = getattr(args_ctx, "kde_bw", "scott") if args_ctx else "scott"
            fes_contour(cvs, out_dir / f"{tag}_fes.svg",
                        title=f"{name} — Free energy landscape",
                        method=fes_method,
                        kde_bw=fes_kde_bw,
                        rep_points=rep_pts or None)
        except Exception:
            logger.warning("FES contour failed for %s:\n%s",
                           name, traceback.format_exc())
        if not read_mode:
            save_cv(cvs, out_dir / f"{tag}.npy")
            # Companion: CVs widened with the two cluster-id columns, so a
            # single load gives you both projections and basin assignments.
            try:
                clusters_path = Path(out_dir) / "basins" / f"{tag}_clusters.npy"
                if clusters_path.exists():
                    cl = np.load(str(clusters_path))
                    if cl.shape[0] == cvs.shape[0]:
                        widened = np.column_stack([cvs, cl.astype(np.float64)])
                        np.save(str(out_dir / f"{tag}_with_clusters.npy"), widened)
            except Exception:
                logger.warning("Failed to write %s widened-with-clusters .npy:\n%s",
                               name, traceback.format_exc())
        try:
            feat_pub = _RUN_CTX.get("feat")
            if feat_pub is not None and getattr(feat_pub, "colorings", None):
                publication_panel(
                    cvs, feat_pub.colorings,
                    out_dir / f"{tag}_pub.svg",
                    kde_bw=getattr(args_ctx, "kde_bw", "scott") if args_ctx else "scott",
                )
        except Exception:
            logger.warning("Publication panel failed for %s:\n%s",
                           name, traceback.format_exc())
        feat = _RUN_CTX.get("feat")
        args = _RUN_CTX.get("args")
        if feat is not None:
            q = getattr(feat, "colorings", {}).get("Q (native contacts)")
            rg = getattr(feat, "colorings", {}).get("Radius of gyration (Å)")
            if q is not None and rg is not None:
                try:
                    scatter_rg_q_by_cv(
                        cvs, q, rg, name, out_dir / f"{tag}_rg_vs_q.svg",
                    )
                except Exception:
                    logger.warning("Rg-vs-Q plot failed for %s:\n%s",
                                   name, traceback.format_exc())
        if feat is not None and args is not None:
            from cv_playground.metrics import score_and_log
            score_and_log(
                name=name,
                subsection=Path(out_dir).name,
                cvs=cvs,
                feat=feat,
                args=args,
                out_root=Path(args.output_dir),
            )
        if _RUN_CTX.get("interpret_mode") == "pearson" and feat is not None:
            try:
                from cv_playground.interpret import pearson_loading_map
                pearson_loading_map(
                    cvs, feat.distances, feat.n_atoms,
                    save_path=Path(out_dir) / "interpret" / f"{tag}.svg",
                    title=f"{name} — Pearson(CV, pairwise Cα distance)",
                )
            except Exception:
                logger.warning("Pearson interpret failed for %s:\n%s",
                               name, traceback.format_exc())

        # ── universal per-Cα importance: η² (always) + PDP (when transform_fn) ──
        if feat is not None and getattr(feat, "distances", None) is not None:
            try:
                import csv
                from cv_playground.interpret import (
                    per_residue_eta2, per_residue_pdp, plot_residue_importance,
                    pearson_pair, condmean_range_pair, _pair_to_residue,
                )
                eta2_res, pair_eta2 = per_residue_eta2(
                    cvs, feat.distances, feat.n_atoms,
                )
                pair_pearson = pearson_pair(cvs, feat.distances)
                pair_cmr = condmean_range_pair(cvs, feat.distances)
                pearson_res = _pair_to_residue(pair_pearson, feat.n_atoms)
                cmr_res = _pair_to_residue(pair_cmr, feat.n_atoms)

                pdp_res = None
                if transform_fn is not None and pdp_kind == "distances":
                    top_n = min(30, pair_eta2.shape[0])
                    top_ids = np.argsort(pair_eta2.max(axis=1))[-top_n:]
                    pdp_out = per_residue_pdp(
                        transform_fn, feat.distances, feat.n_atoms,
                        top_pair_ids=top_ids,
                    )
                    if pdp_out is not None:
                        pdp_res = pdp_out[0]
                elif transform_fn is not None and pdp_kind == "cartesian":
                    from cv_playground.interpret import per_residue_pdp_cartesian
                    coords = getattr(feat, "coords_3d", None)
                    if coords is not None and coords.ndim == 3:
                        pdp_res = per_residue_pdp_cartesian(transform_fn, coords)

                interpret_dir = Path(out_dir) / "interpret"
                interpret_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(interpret_dir / f"{tag}_residue_eta2.npy"), eta2_res)
                if pdp_res is not None:
                    np.save(str(interpret_dir / f"{tag}_residue_pdp.npy"), pdp_res)

                # Tidy CSV: one row per Cα, all importance scalars side by side.
                n_a, n_cv = eta2_res.shape
                csv_path = interpret_dir / f"{tag}_residue.csv"
                with open(csv_path, "w", newline="") as fp:
                    w = csv.writer(fp)
                    cols = ["residue"]
                    for k in range(n_cv):
                        cols += [f"pearson_abs_cv{k+1}",
                                 f"condmean_range_cv{k+1}",
                                 f"eta2_cv{k+1}",
                                 f"pdp_cv{k+1}"]
                    w.writerow(cols)
                    for i in range(n_a):
                        row = [i]
                        for k in range(n_cv):
                            row.append(f"{pearson_res[i, k]:.6g}")
                            row.append(f"{cmr_res[i, k]:.6g}")
                            row.append(f"{eta2_res[i, k]:.6g}")
                            pdp_v = (pdp_res[i, k]
                                     if pdp_res is not None and k < pdp_res.shape[1]
                                     else float("nan"))
                            row.append(f"{pdp_v:.6g}")
                        w.writerow(row)
                logger.info("Saved per-Cα importance CSV → %s", csv_path)

                plot_residue_importance(
                    eta2_res, pdp_res,
                    save_path=interpret_dir / f"{tag}_residue.svg",
                    title=f"{name} — per-Cα importance",
                )
            except Exception:
                logger.warning("Residue importance failed for %s:\n%s",
                               name, traceback.format_exc())
        return cvs
    except Exception:
        logger.error("✘ %s FAILED:\n%s", name, traceback.format_exc())
        return None


# ─────────────────────────── HTML summary ───────────────────────────────────

def _img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _svg_data_uri(path: Path) -> str:
    return f"data:image/svg+xml;base64,{_img_to_base64(path)}"


_METRIC_COLUMNS = [
    ("subsection",   "subsection"),
    ("method",       "method"),
    ("tau_ns",       "τ (ns)"),
    ("vamp2",        "VAMP-2"),
    ("pearson_Q",    "|r|_Q"),
    ("pearson_RMSD", "|r|_RMSD"),
    ("pearson_Rg",   "|r|_Rg"),
    ("folded_acc",   "f/u acc"),
    ("its_ns",       "ITS (ns)"),
    ("basins",       "basins"),
]


def _render_run_details() -> str:
    args = _RUN_CTX.get("args")
    feat = _RUN_CTX.get("feat")
    if args is None:
        return ""
    tau_ns = effective_tau_ns(args)
    rows: list[tuple[str, str]] = [
        ("topology",           str(getattr(args, "top", "—"))),
        ("trajectory",         str(getattr(args, "traj", "—"))),
        ("native reference",   str(getattr(args, "native", None) or "frame 0")),
        ("atom selection",     str(getattr(args, "selection", "—"))),
        ("stride",             str(getattr(args, "stride", "—"))),
        ("dt (ps)",            str(getattr(args, "dt_ps", "—"))),
        ("lag (frames)",       str(getattr(args, "lag", "—"))),
        ("effective τ (ns)",   f"{tau_ns:.2f}"),
        ("epochs",             str(getattr(args, "epochs", "—"))),
        ("batch size",         str(getattr(args, "batch_size", "—"))),
        ("seed",               str(getattr(args, "seed", "—"))),
        ("output dir",         str(getattr(args, "output_dir", "—"))),
        ("labels override",    str(getattr(args, "labels", None) or "Q-based (auto)")),
    ]
    if feat is not None:
        n_frames = int(getattr(feat, "distances", np.zeros((0, 0))).shape[0])
        n_pairs = int(getattr(feat, "distances", np.zeros((0, 0))).shape[1]) \
            if hasattr(feat, "distances") else 0
        rows.append(("n_frames (after stride)", str(n_frames)))
        rows.append(("n_atoms",                 str(int(getattr(feat, "n_atoms", 0)))))
        rows.append(("n_pairwise distances",    str(n_pairs)))
    html = ["<h2>Run details</h2>",
            "<table class='run'><tbody>"]
    for k, v in rows:
        html.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def _render_metrics_table(output_root: Path) -> str:
    import json
    mpath = output_root / "metrics.json"
    if not mpath.exists():
        return ""
    try:
        data = json.loads(mpath.read_text())
    except Exception:
        return ""
    if not isinstance(data, dict) or not data:
        return ""
    rows = sorted(data.values(), key=lambda r: (r.get("subsection", ""),
                                                r.get("method", "")))
    html = ["<h2>Comparison metrics</h2>",
            "<table class='metrics'><thead><tr>"]
    for _, label in _METRIC_COLUMNS:
        html.append(f"<th>{label}</th>")
    html.append("</tr></thead><tbody>")
    for r in rows:
        html.append("<tr>")
        for key, _ in _METRIC_COLUMNS:
            v = r.get(key)
            if isinstance(v, float):
                cell = f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
            elif v is None:
                cell = "—"
            else:
                cell = str(v)
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


_METHOD_HYPERPARAMS: dict[str, str] = {
    "PCA on aligned Cartesians":            "n_components=2, sklearn randomized SVD, RMSD-aligned Cα coords.",
    "PCA on internal coordinates":          "n_components=2, input = distances + sin/cos dihedrals (hstacked).",
    "PCA on pairwise Cα distances":         "n_components=2 on 190-D Cα–Cα distance vector.",
    "Deep-tICA (mlcolvar)":                 "layers=[n_feat, 128, 64, 2], n_cvs=2, Adam default, epochs/batch from CLI.",
    "tICA (time-lagged independent components)": "dim=2, lag from --lag, input depends on script (distances or sin/cos dihedrals).",
    "HLDA / Fisher LDA on distances":       "S_b v = λ S_w v with ε·I regularisation (ε=1e-6); labels from Q cuts (0.3 / 0.7) with k-means(RMSD) fallback.",
    "Deep-LDA (mlcolvar)":                  "layers=[n_feat, 128, 64, min(2, n_classes−1)], n_states from Q-labels.",
    "LDA on Q-based labels":                "PCA-50 pre-reduction → sklearn LDA; labels from Q cuts (0.3 / 0.7).",
    "Diffusion maps":                       "n_evecs=2, α=0.5, k=100 (Zheng–Rohrdanz–Clementi 2013 Trp-cage); ε = 'bgh' auto and 0.5·/1·/2·median² sweep.",
    "Laplacian eigenmaps":                  "SpectralEmbedding, affinity='nearest_neighbors', n_components=2; k ∈ {5, 15, 50}.",
    "Isomap":                               "n_components=2; n_neighbors sweep k ∈ {5, 15, 50}.",
    "Locally Linear Embedding":             "n_components=2; n_neighbors sweep k ∈ {5, 15, 50}.",
    "UMAP (visualization only)":            "n_components=2, default min_dist/n_neighbors, random_state=seed.",
    "Kernel PCA (RBF) on contacts":         "n_components=2, kernel='rbf', γ = 1 / (2·median ||x−x'||²) on contact maps.",
    "Time-lagged autoencoder":              "Encoder/decoder MLPs 128→64→2 / 2→64→128→n_feat, MSE(recon)+MSE(recon_lag)+0.5·MSE(z_t, z_lag).",
    "GraphVAMPnet (SchNet encoder)":        "SchNet hidden=32, num_filters=32, 3 interactions, 16 Gaussians, cutoff=7.5 Å; head 1→32→n. Sweep n ∈ {2, 5}. Loss = −VAMP-2. Defaults from Ghorbani, Hoffmann & Ferguson 2022 (Trp-cage benchmark).",
    "GraphVAMPnet (paper SchNet)":          "Paper-faithful SchNet (Ghorbani/Hoffmann/Ferguson 2022, Table I): n_conv=4, h_a=16, 12 Gaussians in [2, 8] Å (step 0.5), residual, k-NN=min(7, n_atoms−1), n_classes=5, Adam lr=5e-4, batch_size=min(1000, max(16, --batch-size)).",
    "VAMPnet":                              "MLP lobe 5×100 (ELU) → n; deeptime VAMPNet lr=1e-3. Sweep n ∈ {2, 6}. Lobe width/depth from Mardt, Pasquali, Wu & Noé 2018 (Trp-cage benchmark).",
    "RAVE (time-lagged VAE)":               "Encoder 128→64→(μ, logσ²), predictor 2→64→128→n_feat; loss = MSE(x_{t+τ}, pred) + KL(q||N(0,I)).",
    "Variational autoencoder (bonus)":      "Encoder 128→64→(μ, logσ²), decoder 2→64→128→n_feat; ELBO = MSE(x, recon) + KL.",
    "EGNN autoencoder":                     "2× EGNNConv (node_dim=32, hidden=64), latent=2; decoder 2→64→128→n_pairs; target = upper-triangle pairwise distances.",
    "SchNet + PCA":                         "SchNet hidden=64, 3 interactions, cutoff=10 Å → 16-D embedding → sklearn PCA(n=2).",
    "Latent-space tICA on EGNN embeddings": "deeptime TICA(lag, dim=2) on 2-D EGNN latent.",
}


def _expl(what: str, usage: str, meaning: str, ref_prose: str,
          refs: list[tuple[str, str]]) -> str:
    """Compose a method explanation block: what / use in CV discovery / meaning /
    prose citation / clickable short-labels (one per cited paper)."""
    label = "References" if len(refs) > 1 else "Reference"
    links = " &middot; ".join(
        f"<a href='{url}' target='_blank'>{title}</a>" for title, url in refs
    )
    return (
        f"<b>What it is.</b> {what}<br>"
        f"<b>Role in CV discovery.</b> {usage}<br>"
        f"<b>How to read the plot.</b> {meaning}<br>"
        f"<b>{label}.</b> {ref_prose}<br>"
        f"<b>Link{'s' if len(refs) > 1 else ''}.</b> {links}"
    )


_METHOD_EXPLANATIONS: list[tuple[tuple[str, ...], str, str]] = [
    # (keyword patterns — any-match, display name, explanation)
    (("pca_cartesian",), "PCA on aligned Cartesians",
     _expl(
         "Linear projection onto directions of maximum positional variance after RMSD alignment.",
         "The oldest CV-discovery recipe: use the top eigenvectors of the Cartesian covariance as "
         "'essential' coordinates to project and bias dynamics. Easy to interpret but conflates "
         "amplitude with kinetics and depends on the alignment frame.",
         "Scatter axes are the two highest-variance collective motions; look for two clouds "
         "that separate folded from unfolded frames and track Q / RMSD.",
         "Amadei, Linssen & Berendsen 1993 — 'Essential dynamics of proteins' (Proteins). "
         "Trp-cage PCA: Juraszek & Bolhuis 2006, PNAS, 'Sampling the multiple folding mechanisms of Trp-cage in explicit solvent'.",
         [("Amadei 1993 (Proteins)", "https://doi.org/10.1002/prot.340170408"),
          ("Juraszek & Bolhuis 2006 (PNAS)", "https://doi.org/10.1073/pnas.0603229103")],
     )),
    (("pca_internal",), "PCA on internal coordinates",
     _expl(
         "Linear PCA on stacked pairwise distances + backbone sin/cos dihedrals — invariant to "
         "rotation/translation.",
         "Removes the alignment artefacts that plague Cartesian PCA, yielding CVs that can be "
         "biased in rotation-invariant ways in OPES / metadynamics.",
         "CV1/CV2 are a variance-maximising mixture of distances and torsions; the Pearson loading "
         "panel identifies which residue pairs dominate.",
         "Mu, Nguyen & Stock 2005, Proteins, 'Energy landscape of a small peptide revealed by "
         "dihedral angle principal component analysis' (dPCA). Applied to Trp-cage dihedrals by "
         "Paschek, Hempel & García 2008, PNAS.",
         [("Mu, Nguyen & Stock 2005 (Proteins)", "https://doi.org/10.1002/prot.20310"),
          ("Paschek, Hempel & García 2008 (PNAS)", "https://doi.org/10.1073/pnas.0805163105")],
     )),
    (("pca_distances",), "PCA on pairwise Cα distances",
     _expl(
         "Linear PCA of the 190-D Cα–Cα distance vector.",
         "A rotation/translation-invariant baseline CV. Widely used as a sanity check before "
         "trying nonlinear or time-aware methods.",
         "Loadings (correlation panel) highlight the residue–residue distances driving each CV — "
         "folding typically shows the N-terminal helix / C-terminal polyproline contacts loading strongly.",
         "Juraszek & Bolhuis 2006, PNAS, 'Sampling the multiple folding mechanisms of Trp-cage in "
         "explicit solvent'.",
         [("Juraszek & Bolhuis 2006 (PNAS)", "https://doi.org/10.1073/pnas.0603229103")],
     )),
    (("deep_tica",), "Deep-tICA (mlcolvar)",
     _expl(
         "Neural encoder trained to maximise auto-correlation at lag τ — a nonlinear generalisation of tICA.",
         "Central CV-discovery tool for enhanced sampling: the learned latent is differentiable, so "
         "it can drive OPES / metadynamics along the slowest modes of the sampled dynamics.",
         "Jacobian map shows which input distances the encoder relies on; CV1 should separate "
         "folded from unfolded basins with a clear transition region.",
         "Bonati, Piccini & Parrinello 2021, PNAS, 'Deep learning the slow modes for rare events "
         "sampling' — demonstrates Deep-TICA on Trp-cage folding.",
         [("Bonati, Piccini & Parrinello 2021 (PNAS)", "https://doi.org/10.1073/pnas.2113533118")],
     )),
    (("tica",), "tICA (time-lagged independent components)",
     _expl(
         "Linear slow-mode finder: maximises time-lagged auto-correlation at lag τ under a reversibility assumption.",
         "Standard preprocessor for Markov state models and a go-to linear CV for folding: the "
         "leading tICs approximate the slowest relaxing CVs of the trajectory.",
         "CV1 is the slowest coordinate — for Trp-cage it typically aligns with the folding "
         "transition; CV2 captures the next-slowest process (e.g. helix vs. cage formation).",
         "Pérez-Hernández et al. 2013, J. Chem. Phys., 'Identification of slow molecular order "
         "parameters for Markov model construction'. Trp-cage tICA: Schwantes & Pande 2013, JCTC, "
         "'Improvements in Markov state model construction reveal many non-native interactions in "
         "the folding of NTL9 and Trp-cage'.",
         [("Pérez-Hernández et al. 2013 (JCP)", "https://doi.org/10.1063/1.4811489"),
          ("Schwantes & Pande 2013 (JCTC)", "https://doi.org/10.1021/ct300878a")],
     )),
    (("hlda",), "HLDA / Fisher LDA on distances",
     _expl(
         "Supervised linear projection maximising between-class over within-class scatter with a "
         "harmonic-mean (HLDA) pooling of within-class covariances.",
         "Designed specifically as a CV for metadynamics: given metastable-state labels from short "
         "simulations, HLDA returns a single differentiable descriptor separating those states, "
         "ready to bias.",
         "If the folded/unfolded basins are cleanly discriminated along CV1, HLDA has captured the "
         "correct order parameter for the labelled transition.",
         "Mendels, Piccini & Parrinello 2018, J. Phys. Chem. Lett., 'Collective variables from local "
         "fluctuations' (HLDA). See also the Bonati et al. Trp-cage benchmark for the mlcolvar suite (2023).",
         [("Mendels, Piccini & Parrinello 2018 (JPC Lett)", "https://doi.org/10.1021/acs.jpclett.8b00733"),
          ("Bonati et al. 2023 (mlcolvar, JCP)", "https://doi.org/10.1063/5.0156343")],
     )),
    (("deep_lda",), "Deep-LDA (mlcolvar)",
     _expl(
         "Neural LDA: an MLP feeds into a Fisher-discriminant loss using Q-based folded/unfolded labels.",
         "Generalises HLDA to the nonlinear regime so a single CV can separate states that are not "
         "linearly distinguishable in the input features; used to drive enhanced-sampling biases.",
         "Latent dim = min(2, n_classes−1); a clean bimodal projection along CV1 indicates a "
         "successful state-separating coordinate.",
         "Bonati, Rizzi & Parrinello 2020, J. Phys. Chem. Lett., 'Data-driven collective variables "
         "for enhanced sampling' — uses Trp-cage as one of the validation systems.",
         [("Bonati, Rizzi & Parrinello 2020 (JPC Lett)", "https://doi.org/10.1021/acs.jpclett.0c00535")],
     )),
    (("lda",), "LDA on Q-based labels",
     _expl(
         "Linear discriminant on pairwise distances using Q-based folded/unfolded/intermediate labels.",
         "A minimal supervised CV baseline — any proposed CV should beat this projection on basin "
         "separation and kinetic content.",
         "CV1 is the direction of best linear class separation; compare VAMP-2 and Q-correlation "
         "against the more sophisticated methods in the metrics table.",
         "Piccini, Mendels & Parrinello 2018, JCTC, 'Metadynamics with discriminants: a tool for "
         "understanding chemistry'.",
         [("Piccini, Mendels & Parrinello 2018 (JCTC)", "https://doi.org/10.1021/acs.jctc.8b00634")],
     )),
    (("diffusion_maps", "diffusionmap"), "Diffusion maps",
     _expl(
         "Eigenmap of a Markov transition matrix built from an RBF kernel on distances.",
         "A nonlinear manifold-learning CV: the leading eigenvectors approximate slow modes of a "
         "diffusion process on the data manifold and have been used both as analysis CVs and (with "
         "Nyström extension) as bias-capable CVs.",
         "ε controls kernel bandwidth; the ε-sweep probes sensitivity. Look for 1-D progress "
         "along CV1 with transversal fluctuations on CV2.",
         "Rohrdanz, Zheng, Maggioni & Clementi 2011, J. Chem. Phys., 'Determination of reaction "
         "coordinates via locally scaled diffusion map'. Trp-cage application: Zheng, Rohrdanz & "
         "Clementi 2013, J. Phys. Chem. B.",
         [("Rohrdanz et al. 2011 (JCP)", "https://doi.org/10.1063/1.3569857"),
          ("Zheng, Rohrdanz & Clementi 2013 (JPC B)", "https://doi.org/10.1021/jp401911h")],
     )),
    (("laplacian",), "Laplacian eigenmaps",
     _expl(
         "Spectral embedding of the k-NN graph Laplacian — preserves local neighbourhoods.",
         "Non-parametric CV-discovery tool highlighting manifold topology; rarely used to bias "
         "directly but useful as an analysis CV and a sanity check for other manifold methods.",
         "k-sweep checks robustness of the manifold scale; large gaps between points on different "
         "basins indicate the embedding resolves the folding reaction.",
         "Das, Moll, Stamati, Kavraki & Clementi 2006, PNAS, 'Low-dimensional, free-energy "
         "landscapes of protein-folding reactions by nonlinear dimensionality reduction'.",
         [("Das et al. 2006 (PNAS)", "https://doi.org/10.1073/pnas.0603553103")],
     )),
    (("isomap",), "Isomap",
     _expl(
         "Embeds geodesic distances along the k-NN graph via classical MDS — preserves global "
         "manifold distances when the graph is connected.",
         "Analysis-oriented CV: pinpoints the global arc length of a folding pathway. Useful for "
         "diagnosing whether the reaction lives on a 1-D manifold.",
         "A quasi-1-D ribbon along CV1 indicates a well-defined folding pathway; bifurcations "
         "signal parallel folding routes.",
         "Das, Moll, Stamati, Kavraki & Clementi 2006, PNAS, 'Low-dimensional, free-energy "
         "landscapes of protein-folding reactions by nonlinear dimensionality reduction' "
         "— demonstrates Isomap on peptide folding landscapes relevant to Trp-cage-sized systems.",
         [("Das et al. 2006 (PNAS)", "https://doi.org/10.1073/pnas.0603553103")],
     )),
    (("lle",), "Locally Linear Embedding",
     _expl(
         "Each point is reconstructed from its k nearest neighbours; the low-dim embedding "
         "preserves those local reconstruction weights.",
         "Complementary manifold-learning CV: preserves local geometry rather than geodesic "
         "distances, highlighting conformational heterogeneity on each basin.",
         "Well-separated clusters along CV1/CV2 indicate distinct local geometries; compare to "
         "Isomap for global-versus-local sensitivity.",
         "Roweis & Saul 2000, Science, 'Nonlinear dimensionality reduction by Locally Linear "
         "Embedding'. Protein-folding application: Stamati, Clementi & Kavraki 2010, Proteins.",
         [("Roweis & Saul 2000 (Science)", "https://doi.org/10.1126/science.290.5500.2323"),
          ("Stamati, Clementi & Kavraki 2010 (Proteins)", "https://doi.org/10.1002/prot.22691")],
     )),
    (("umap",), "UMAP (visualization only)",
     _expl(
         "Stochastic neighbour embedding based on fuzzy simplicial sets.",
         "Visualization-only CV: non-differentiable, so it cannot bias a simulation. Included "
         "because it often produces the cleanest picture of the state decomposition.",
         "Clusters map to metastable states; the arrangement is topological — Euclidean distances "
         "in the plane do not carry free-energy meaning.",
         "McInnes, Healy & Melville 2018, arXiv:1802.03426 — 'UMAP: Uniform Manifold Approximation "
         "and Projection for Dimension Reduction'. Trp-cage application: Trozzi, Wang & Tao 2021, "
         "J. Phys. Chem. B, 'UMAP as a dimensionality reduction tool for molecular dynamics simulations of biomacromolecules'.",
         [("McInnes, Healy & Melville 2018 (arXiv)", "https://arxiv.org/abs/1802.03426"),
          ("Trozzi, Wang & Tao 2021 (JPC B)", "https://doi.org/10.1021/acs.jpcb.1c02081")],
     )),
    (("kernelpca",), "Kernel PCA (RBF) on contacts",
     _expl(
         "PCA in a feature space induced by an RBF kernel on binary contact maps.",
         "Nonlinear-PCA baseline: cheap and deterministic, often recovers qualitatively the same "
         "modes as Isomap/LLE without the need to build a k-NN graph.",
         "γ set from the median heuristic; CV1 should track global contact formation (high |r|_Q).",
         "Schölkopf, Smola & Müller 1998, Neural Computation, 'Nonlinear component analysis as a "
         "kernel eigenvalue problem'.",
         [("Schölkopf, Smola & Müller 1998 (Neural Comput.)", "https://doi.org/10.1162/089976698300017467")],
     )),
    (("tae",), "Time-lagged autoencoder",
     _expl(
         "Autoencoder with an extra term forcing encodings at t and t+τ to coincide, nudging the "
         "latent toward slow modes.",
         "Deep-learning CV that blends reconstruction with a kinetic prior; differentiable, so "
         "usable for OPES / metadynamics biasing.",
         "Jacobian map = sensitivity of latent to each input distance; compare its basin structure "
         "with Deep-TICA to check that the kinetic term is active.",
         "Wehmeyer & Noé 2018, J. Chem. Phys., 'Time-lagged autoencoders: Deep learning of slow "
         "collective variables for molecular kinetics'. Validated on Trp-cage and alanine dipeptide.",
         [("Wehmeyer & Noé 2018 (JCP)", "https://doi.org/10.1063/1.5011399")],
     )),
    (("graphvampnet_paper_schnet", "paper_schnet"), "GraphVAMPnet (paper SchNet)",
     _expl(
         "GraphVAMPNet with the original Ghorbani–Hoffmann–Ferguson modified-SchNet lobe "
         "(k-NN edges, residual interaction blocks, RBF distance expansion in [2, 8] Å).",
         "Reproduces the paper-faithful Trp-cage configuration so the benchmark scores "
         "are comparable to Ghorbani et al. 2022. Used to sanity-check our lighter-weight "
         "GraphVAMPnet (SchNet encoder) variant.",
         "Output is the 5-state soft assignment (only the first 2 columns are plotted); "
         "class concentration along CV1 should match the folded/unfolded separation.",
         "Ghorbani, Hoffmann & Ferguson 2022, J. Chem. Phys., 'GraphVAMPNet, using graph "
         "neural networks to learn coarse-grained dynamics of molecular systems' — Table I "
         "configuration for Trp-cage.",
         [("Ghorbani, Hoffmann & Ferguson 2022 (JCP)", "https://doi.org/10.1063/5.0085607")],
     )),
    (("graphvampnet",), "GraphVAMPnet (SchNet encoder)",
     _expl(
         "SchNet graph encoder trained via the VAMP-2 variational score on time-lagged frame pairs.",
         "Equivariant deep CV that consumes atom positions directly (no hand-crafted features) and "
         "simultaneously learns a state decomposition; among the most accurate CV discovery methods "
         "benchmarked on Trp-cage.",
         "n = latent / state dim; residue-importance bars show |∂CV/∂pos| per residue — for "
         "Trp-cage the Trp6 indole residue typically dominates.",
         "Ghorbani, Hoffmann & Ferguson 2022, J. Chem. Phys., 'GraphVAMPNet, using graph neural "
         "networks to learn coarse-grained dynamics of molecular systems' — benchmarked on Trp-cage.",
         [("Ghorbani, Hoffmann & Ferguson 2022 (JCP)", "https://doi.org/10.1063/5.0085607")],
     )),
    (("vampnet",), "VAMPnet",
     _expl(
         "Feed-forward encoder trained with the VAMP-2 variational score on time-lagged feature pairs.",
         "End-to-end replacement for the tICA+clustering pipeline that underpins Markov state "
         "models — simultaneously learns a CV and a soft state assignment.",
         "n = output dim; sweep compares 2-state (folding yes/no) vs. 6-state (intermediates) "
         "resolution. A high VAMP-2 score implies informative slow CVs.",
         "Mardt, Pasquali, Wu & Noé 2018, Nature Communications, 'VAMPnets for deep learning of "
         "molecular kinetics' — validates VAMPnets on Trp-cage folding trajectories.",
         [("Mardt, Pasquali, Wu & Noé 2018 (Nature Commun.)", "https://doi.org/10.1038/s41467-017-02388-1")],
     )),
    (("rave",), "RAVE (time-lagged VAE)",
     _expl(
         "VAE whose decoder predicts x_{t+τ} from z_t, combining a slow-mode bias with a Gaussian prior.",
         "Self-consistent CV discovery: the learned latent is iteratively refined inside an "
         "enhanced-sampling loop (predictive information bottleneck), yielding bias-capable CVs.",
         "Latent μ is the CV; a clear bimodal distribution indicates a successful slow coordinate.",
         "Ribeiro, Bravo, Wang & Tiwary 2018, J. Chem. Phys., 'Reweighted autoencoded variational "
         "Bayes for enhanced sampling (RAVE)'. Trp-cage application: Wang, Ribeiro & Tiwary 2019, Nature Commun.",
         [("Ribeiro et al. 2018 (JCP)", "https://doi.org/10.1063/1.5025487"),
          ("Wang, Ribeiro & Tiwary 2019 (Nature Commun.)", "https://doi.org/10.1038/s41467-019-11405-4")],
     )),
    (("vae",), "Variational autoencoder (bonus)",
     _expl(
         "Plain VAE on distance features — not time-aware.",
         "Baseline CV for the time-lagged variants (RAVE, TAE): shows what a purely geometric "
         "latent gives up when the kinetic term is removed.",
         "Expect a latent that correlates with Rg/RMSD amplitude but mixes timescales; compare to "
         "the TAE plot for the effect of the time-lag term.",
         "Hernández, Wayment-Steele, Sultan, Husic & Pande 2018, Physical Review E, 'Variational "
         "encoding of complex dynamics' — applies a time-lagged VAE to Trp-cage.",
         [("Hernández et al. 2018 (Phys. Rev. E)", "https://doi.org/10.1103/PhysRevE.97.062412")],
     )),
    (("egnn_autoencoder", "egnn"), "EGNN autoencoder",
     _expl(
         "E(n)-equivariant GNN encoder; decoder reconstructs the (invariant) pairwise-distance vector.",
         "Geometry-aware CV that guarantees rotation / translation / reflection equivariance by "
         "construction — no alignment needed, and the latent inherits the symmetry.",
         "Residue-importance bars show how strongly each atom position pulls the latent CV; "
         "expect Trp6 and the salt-bridge residues to dominate.",
         "Satorras, Hoogeboom & Welling 2021, ICML, 'E(n) Equivariant Graph Neural Networks'. "
         "Molecular CV usage: Ghorbani et al. 2022 (GraphVAMPNet) for Trp-cage.",
         [("Satorras, Hoogeboom & Welling 2021 (ICML)", "https://proceedings.mlr.press/v139/satorras21a.html"),
          ("Ghorbani, Hoffmann & Ferguson 2022 (JCP)", "https://doi.org/10.1063/5.0085607")],
     )),
    (("schnet_+_pca", "schnet_pca"), "SchNet + PCA",
     _expl(
         "SchNet graph embeddings projected to 2-D by linear PCA post-hoc.",
         "Minimal-training CV-discovery pipeline: uses a pretrained-style graph embedding as a "
         "featuriser, then takes the dominant variance directions as CVs.",
         "Loading map is Pearson |corr(CV, distance)|; tracks which residue pairs the SchNet "
         "embedding encodes most strongly.",
         "Schütt et al. 2018, J. Chem. Phys., 'SchNet – A deep learning architecture for molecules "
         "and materials'.",
         [("Schütt et al. 2018 (JCP)", "https://doi.org/10.1063/1.5019779")],
     )),
    (("latent_tica",), "Latent-space tICA on EGNN embeddings",
     _expl(
         "tICA on the EGNN latent — adds a time-lagged slow-mode filter on top of the equivariant embedding.",
         "Hybrid CV discovery: equivariant deep featuriser + linear kinetic projector, combining "
         "interpretability with kinetic relevance.",
         "Loading map is a Pearson pseudo-loading; compare CV1 basins to the raw EGNN plot to see "
         "what the tICA layer adds kinetically.",
         "Pérez-Hernández et al. 2013, J. Chem. Phys., 'Identification of slow molecular order "
         "parameters for Markov model construction' (tICA). Trp-cage GNN+tICA benchmark: "
         "Ghorbani et al. 2022 (GraphVAMPNet).",
         [("Pérez-Hernández et al. 2013 (JCP)", "https://doi.org/10.1063/1.4811489"),
          ("Ghorbani, Hoffmann & Ferguson 2022 (JCP)", "https://doi.org/10.1063/5.0085607")],
     )),
]


def _extract_variant(stem: str) -> str:
    """Pull the parenthesised sweep parameter out of a saved-image stem.

    e.g. 'isomap_(k=15)' → 'k = 15',   'vampnet_(n=6)' → 'n = 6'.
    """
    import re
    m = re.search(r"\(([^)]+)\)", stem)
    if not m:
        return ""
    raw = m.group(1).replace("_", " ").strip()
    raw = raw.replace("=", " = ")
    return f"variant: {raw}"


def _match_method_family(stem: str) -> tuple[str, str, str]:
    s = stem.lower()
    for keys, display, expl in _METHOD_EXPLANATIONS:
        if any(k in s for k in keys):
            return (display, expl, display)
    return (stem, "", stem)


def generate_summary_html(output_root: Path) -> Path:
    """Collect every .svg under *output_root* into a single summary.html.

    The first path component under *output_root* defines the top-level section
    (e.g. ``01_linear_cvs``). Files inside an ``interpret/`` subfolder are
    attached to the matching method group and rendered directly below the
    method's result plots.
    """
    html_path = output_root / "summary.html"
    output_root = Path(output_root)
    sections: dict[str, list[Path]] = {}
    for svg in sorted(output_root.rglob("*.svg")):
        rel = svg.relative_to(output_root)
        if len(rel.parts) < 2:
            continue
        section = rel.parts[0]
        sections.setdefault(section, []).append(svg)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>CV Discovery – Summary</title>",
        "<style>",
        "body{font-family:sans-serif;margin:24px;max-width:1100px}",
        "h2{border-bottom:2px solid #333;padding-bottom:4px}",
        "h3{margin:18px 0 4px;color:#223}",
        "h4.interpret-h{margin:12px 0 6px;color:#446;font-size:14px;"
        "text-transform:uppercase;letter-spacing:0.05em}",
        ".method{margin-bottom:28px}",
        ".expl{margin:4px 0 6px;color:#333;font-size:13px;line-height:1.4;"
        "background:#f6f6f9;padding:8px 12px;border-left:3px solid #667}",
        ".hparams{margin:0 0 10px;color:#223;font-size:12px;line-height:1.4;"
        "font-family:monospace;background:#eef2ee;padding:6px 12px;border-left:3px solid #494}",
        ".variant{margin:4px 0;font-size:12px;color:#335;font-family:monospace}",
        ".grid{display:flex;flex-direction:column;gap:18px;margin-bottom:16px}",
        ".card{text-align:center}",
        ".card img{max-width:100%;height:auto;border:1px solid #ccc;border-radius:6px}",
        ".card small{display:block;margin-top:4px;color:#555;font-size:12px}",
        "table.metrics{border-collapse:collapse;margin-bottom:32px;font-size:13px}",
        "table.metrics th,table.metrics td{border:1px solid #bbb;padding:4px 8px;text-align:right}",
        "table.metrics th{background:#eee;text-align:center}",
        "table.metrics td:nth-child(1),table.metrics td:nth-child(2){text-align:left}",
        "table.run{border-collapse:collapse;margin-bottom:32px;font-size:13px}",
        "table.run th,table.run td{border:1px solid #bbb;padding:4px 10px}",
        "table.run th{background:#eee;text-align:left;font-weight:600;width:220px}",
        "table.run td{text-align:left;font-family:monospace}",
        "</style></head><body>",
        "<h1>CV Discovery Playground – Results</h1>",
    ]

    run_block = _render_run_details()
    if run_block:
        parts.append(run_block)

    metrics_block = _render_metrics_table(output_root)
    if metrics_block:
        parts.append(metrics_block)

    for sec in sorted(sections):
        nice = sec.replace("_", " ").title()
        parts.append(f"<h2>{nice}</h2>")
        groups: dict[str, list[Path]] = {}
        order: list[str] = []
        group_info: dict[str, tuple[str, str]] = {}
        for svg in sections[sec]:
            display, expl, key = _match_method_family(svg.stem)
            if key not in groups:
                groups[key] = []
                order.append(key)
                group_info[key] = (display, expl)
            groups[key].append(svg)
        for key in order:
            display, expl = group_info[key]
            parts.append("<div class='method'>")
            parts.append(f"<h3>{display}</h3>")
            if expl:
                parts.append(f"<div class='expl'>{expl}</div>")
            hp = _METHOD_HYPERPARAMS.get(display)
            if hp:
                parts.append(f"<div class='hparams'>Hyperparameters: {hp}</div>")
            # Results first, then interpret plots directly below.
            mains = [p for p in groups[key] if "interpret" not in p.parts]
            interprets = [p for p in groups[key] if "interpret" in p.parts]
            parts.append("<div class='grid'>")
            for svg in mains:
                uri = _svg_data_uri(svg)
                variant = _extract_variant(svg.stem)
                variant_html = (f"<div class='variant'>{variant}</div>"
                                if variant else "")
                parts.append(
                    f"<div class='card'>{variant_html}"
                    f"<img src='{uri}'/>"
                    f"<small>{svg.stem}</small></div>"
                )
            parts.append("</div>")
            if interprets:
                parts.append("<h4 class='interpret-h'>Interpretation</h4>")
                parts.append("<div class='grid'>")
                for svg in interprets:
                    uri = _svg_data_uri(svg)
                    variant = _extract_variant(svg.stem)
                    variant_html = (f"<div class='variant'>{variant}</div>"
                                    if variant else "")
                    parts.append(
                        f"<div class='card'>{variant_html}"
                        f"<img src='{uri}'/>"
                        f"<small>{svg.stem}</small></div>"
                    )
                parts.append("</div>")
            parts.append("</div>")
    parts.append("</body></html>")

    html_path.write_text("\n".join(parts))
    logger.info("Summary HTML → %s", html_path)
    return html_path


def generate_publication_html(output_root: Path) -> Path:
    """Collect every ``*_pub.svg`` under *output_root* into
    ``summary_publication.html`` — one image per analysis, no titles or
    metrics, grouped by subsection."""
    output_root = Path(output_root)
    html_path = output_root / "summary_publication.html"
    sections: dict[str, list[Path]] = {}
    for svg in sorted(output_root.rglob("*_pub.svg")):
        rel = svg.relative_to(output_root)
        if len(rel.parts) < 2:
            continue
        sections.setdefault(rel.parts[0], []).append(svg)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>CV Discovery – Publication Panels</title>",
        "<style>",
        "body{font-family:'CMU Serif','Computer Modern Roman',serif;"
        "margin:32px;max-width:1400px}",
        "h2{border-bottom:1px solid #888;padding-bottom:4px;"
        "font-weight:normal;letter-spacing:0.02em}",
        ".panel{margin:24px 0}",
        ".panel img{max-width:100%;height:auto;display:block}",
        ".panel small{display:block;margin-top:6px;color:#444;"
        "font-family:monospace;font-size:12px}",
        "</style></head><body>",
    ]
    for sec in sorted(sections):
        nice = sec.replace("_", " ")
        parts.append(f"<h2>{nice}</h2>")
        for svg in sections[sec]:
            uri = _svg_data_uri(svg)
            parts.append(
                f"<div class='panel'><img src='{uri}'/>"
                f"<small>{svg.stem}</small></div>"
            )
    parts.append("</body></html>")
    html_path.write_text("\n".join(parts))
    logger.info("Publication HTML → %s", html_path)
    return html_path

