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
            if stat_name == "std":
                # Normalize σ by the coloring's range, then clip colorbar at 10 %.
                hb_kwargs["reduce_C_function"] = lambda a, s=scale: np.std(a) / s
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
        with timer(name):
            cvs = func()
        if cvs is None or cvs.ndim < 2 or cvs.shape[1] < 2:
            logger.warning("%s returned invalid CV array – skipping.", name)
            return None
        tag = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        scatter_cv(cvs, colorings, name, out_dir / f"{tag}.svg")
        save_cv(cvs, out_dir / f"{tag}.npy")
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
    "HLDA / Fisher LDA on distances":       "S_b v = λ S_w v with ε·I regularisation (ε=1e-6); labels from k-means(RMSD).",
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
    "VAMPnet":                              "MLP lobe 5×100 (ELU) → n; deeptime VAMPNet lr=1e-3. Sweep n ∈ {2, 6}. Lobe width/depth from Mardt, Pasquali, Wu & Noé 2018 (Trp-cage benchmark).",
    "RAVE (time-lagged VAE)":               "Encoder 128→64→(μ, logσ²), predictor 2→64→128→n_feat; loss = MSE(x_{t+τ}, pred) + KL(q||N(0,I)).",
    "Variational autoencoder (bonus)":      "Encoder 128→64→(μ, logσ²), decoder 2→64→128→n_feat; ELBO = MSE(x, recon) + KL.",
    "EGNN autoencoder":                     "2× EGNNConv (node_dim=32, hidden=64), latent=2; decoder 2→64→128→n_pairs; target = upper-triangle pairwise distances.",
    "SchNet + PCA":                         "SchNet hidden=64, 3 interactions, cutoff=10 Å → 16-D embedding → sklearn PCA(n=2).",
    "Latent-space tICA on EGNN embeddings": "deeptime TICA(lag, dim=2) on 2-D EGNN latent.",
}


def _expl(what: str, usage: str, meaning: str, ref_title: str, ref_url: str) -> str:
    """Compose a method explanation block: what / use in CV discovery / meaning / reference."""
    return (
        f"<b>What it is.</b> {what}<br>"
        f"<b>Role in CV discovery.</b> {usage}<br>"
        f"<b>How to read the plot.</b> {meaning}<br>"
        f"<b>Reference.</b> <a href='{ref_url}' target='_blank'>{ref_title}</a>"
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
         "https://doi.org/10.1073/pnas.0603229103",
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
         "https://doi.org/10.1073/pnas.0805163105",
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
         "https://doi.org/10.1073/pnas.0603229103",
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
         "https://doi.org/10.1073/pnas.2113533118",
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
         "https://doi.org/10.1021/ct300878a",
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
         "https://doi.org/10.1021/acs.jpclett.8b00733",
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
         "https://doi.org/10.1021/acs.jpclett.0c00535",
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
         "https://doi.org/10.1021/acs.jctc.8b00634",
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
         "https://doi.org/10.1021/jp401911h",
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
         "https://doi.org/10.1073/pnas.0603553103",
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
         "https://doi.org/10.1073/pnas.0603553103",
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
         "https://doi.org/10.1002/prot.22691",
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
         "https://doi.org/10.1021/acs.jpcb.1c02081",
     )),
    (("kernelpca",), "Kernel PCA (RBF) on contacts",
     _expl(
         "PCA in a feature space induced by an RBF kernel on binary contact maps.",
         "Nonlinear-PCA baseline: cheap and deterministic, often recovers qualitatively the same "
         "modes as Isomap/LLE without the need to build a k-NN graph.",
         "γ set from the median heuristic; CV1 should track global contact formation (high |r|_Q).",
         "Schölkopf, Smola & Müller 1998, Neural Computation, 'Nonlinear component analysis as a "
         "kernel eigenvalue problem'. Protein-folding application: Antoniou & Schwartz 2011, JCTC.",
         "https://doi.org/10.1162/089976698300017467",
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
         "https://doi.org/10.1063/1.5011399",
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
         "https://doi.org/10.1063/5.0085000",
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
         "https://doi.org/10.1038/s41467-017-02388-1",
     )),
    (("rave",), "RAVE (time-lagged VAE)",
     _expl(
         "VAE whose decoder predicts x_{t+τ} from z_t, combining a slow-mode bias with a Gaussian prior.",
         "Self-consistent CV discovery: the learned latent is iteratively refined inside an "
         "enhanced-sampling loop (predictive information bottleneck), yielding bias-capable CVs.",
         "Latent μ is the CV; a clear bimodal distribution indicates a successful slow coordinate.",
         "Ribeiro, Bravo, Wang & Tiwary 2018, J. Chem. Phys., 'Reweighted autoencoded variational "
         "Bayes for enhanced sampling (RAVE)'. Trp-cage application: Wang, Ribeiro & Tiwary 2019, Nature Commun.",
         "https://doi.org/10.1038/s41467-019-11405-4",
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
         "https://doi.org/10.1103/PhysRevE.97.062412",
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
         "https://proceedings.mlr.press/v139/satorras21a.html",
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
         "https://doi.org/10.1063/1.5019779",
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
         "https://doi.org/10.1063/5.0085000",
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

