"""Quantitative metrics for comparing CV methods.

For every method we compute:

* ``vamp2``          – VAMP-2 score of the 2-D CV projection at the run's τ,
                       via the whitened cross-covariance Frobenius-squared.
* ``pearson_*``      – max |Pearson r| between a CV column and
                       {Q, Cα-RMSD, Rg}.
* ``folded_acc``     – 5-fold CV logistic-regression accuracy for separating
                       folded (Q ≥ 0.7) vs unfolded (Q < 0.3) points in the
                       2-D CV plane.
* ``its_ns``         – implied timescale of the slowest mode of a 2-state MSM
                       built from k-means (k=2) on the 2-D CV projection.
* ``basins``         – number of distinct minima in the 2-D free-energy
                       surface (smoothed histogram, local-minimum filter).

Metrics are appended to ``outputs/metrics.json`` (one object, keyed by
``<subsection>/<method>``) and rendered as a table at the top of
``outputs/summary.html``.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────── helpers ──────────────────────────────────────

def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


# ───────────────────────────── metrics ──────────────────────────────────────

def vamp2_score(cvs: np.ndarray, lag: int) -> Optional[float]:
    """VAMP-2 (squared Frobenius of the whitened cross-covariance) of *cvs*
    at lag *lag* (in samples). Values range in [0, d²] where d is the CV dim;
    higher = more of the slow kinetics captured in the projection."""
    if lag <= 0 or lag >= len(cvs):
        return None
    X_t   = _center(cvs[:-lag].astype(np.float64))
    X_tau = _center(cvs[lag:].astype(np.float64))
    n, d = X_t.shape
    if n < d + 2:
        return None
    eye = np.eye(d)
    C00 = X_t.T   @ X_t   / n + 1e-6 * eye
    C11 = X_tau.T @ X_tau / n + 1e-6 * eye
    C01 = X_t.T   @ X_tau / n
    try:
        L00 = np.linalg.cholesky(C00)
        L11 = np.linalg.cholesky(C11)
        K = np.linalg.solve(L00, C01)
        K = np.linalg.solve(L11, K.T).T
    except np.linalg.LinAlgError:
        return None
    return float((K ** 2).sum())


def pearson_abs(cvs: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Max |Pearson r| across CV columns against a 1-D target."""
    if target is None or len(target) != len(cvs):
        return None
    best = 0.0
    tgt = np.asarray(target, dtype=np.float64)
    if tgt.std() == 0:
        return None
    for i in range(cvs.shape[1]):
        col = cvs[:, i].astype(np.float64)
        if col.std() == 0:
            continue
        c = np.corrcoef(col, tgt)[0, 1]
        if np.isfinite(c):
            best = max(best, abs(float(c)))
    return best


def folded_unfolded_accuracy(
    cvs: np.ndarray, q: np.ndarray, cuts=(0.3, 0.7),
) -> Optional[float]:
    """5-fold CV logistic-regression accuracy on folded vs unfolded points."""
    if q is None or len(q) != len(cvs):
        return None
    lo, hi = cuts
    mask = (q < lo) | (q >= hi)
    if mask.sum() < 20:
        return None
    y = (q[mask] >= hi).astype(int)
    if len(np.unique(y)) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=500)
        scores = cross_val_score(clf, cvs[mask], y, cv=5)
        return float(scores.mean())
    except Exception:
        return None


def msm_implied_timescale(
    cvs: np.ndarray, lag: int, dt_ns: float,
) -> Optional[float]:
    """Slowest implied timescale (ns) of a 2-state MSM built from k-means on *cvs*."""
    if lag <= 0 or lag >= len(cvs):
        return None
    try:
        from sklearn.cluster import KMeans
        states = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(cvs)
    except Exception:
        return None
    s_t = states[:-lag]
    s_tau = states[lag:]
    T = np.zeros((2, 2), dtype=np.float64)
    np.add.at(T, (s_t, s_tau), 1.0)
    rs = T.sum(axis=1, keepdims=True)
    if (rs <= 0).any():
        return None
    T = T / rs
    eig = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
    if len(eig) < 2:
        return None
    lam2 = float(eig[1])
    if not (0.0 < lam2 < 1.0):
        return None
    return float(-lag * dt_ns / np.log(lam2))


def fes_basin_count(
    cvs: np.ndarray, bins: int = 60, min_count: int = 5,
    sigma: float = 1.5, window: int = 5,
) -> Optional[int]:
    """Count local minima in the smoothed 2-D negative-log-density."""
    if cvs.shape[1] < 2:
        return None
    try:
        from scipy.ndimage import gaussian_filter, minimum_filter
    except Exception:
        return None
    H, _, _ = np.histogram2d(cvs[:, 0], cvs[:, 1], bins=bins)
    if H.sum() == 0:
        return 0
    F = -np.log(H + 1e-6)
    Fs = gaussian_filter(F, sigma=sigma)
    pop = H >= min_count
    mf = minimum_filter(Fs, size=window)
    basins = (Fs == mf) & pop
    return int(basins.sum())


# ───────────────────────────── orchestration ────────────────────────────────

def score_cv(name: str, subsection: str, cvs: np.ndarray, feat, args) -> dict:
    """Compute the full metric row for one CV method."""
    col = feat.colorings
    q   = col.get("Q (native contacts)")
    rms = col.get("Cα-RMSD to native (Å)")
    rg  = col.get("Radius of gyration (Å)")
    dt_sample_ns = float(args.stride) * float(args.dt_ps) / 1000.0
    tau_ns = float(args.lag) * dt_sample_ns
    return {
        "subsection":   subsection,
        "method":       name,
        "tau_ns":       tau_ns,
        "vamp2":        vamp2_score(cvs, int(args.lag)),
        "pearson_Q":    pearson_abs(cvs, q),
        "pearson_RMSD": pearson_abs(cvs, rms),
        "pearson_Rg":   pearson_abs(cvs, rg),
        "folded_acc":   folded_unfolded_accuracy(cvs, q),
        "its_ns":       msm_implied_timescale(cvs, int(args.lag), dt_sample_ns),
        "basins":       fes_basin_count(cvs),
    }


def append_metric_row(path: Path, row: dict) -> None:
    """Upsert a row into the JSON store keyed by ``<subsection>/<method>``."""
    path = Path(path)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    key = f"{row['subsection']}/{row['method']}"
    data[key] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def reset_metric_store(path: Path) -> None:
    path = Path(path)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            logger.warning("Could not remove stale metrics file at %s", path)


def score_and_log(name: str, subsection: str, cvs: np.ndarray,
                  feat, args, out_root: Path) -> Optional[dict]:
    """Score a CV, persist, and log a one-line summary. Never raises."""
    try:
        row = score_cv(name, subsection, cvs, feat, args)
    except Exception:
        logger.warning("Metric scoring failed for %s:\n%s",
                       name, traceback.format_exc())
        return None
    append_metric_row(Path(out_root) / "metrics.json", row)
    def fmt(x, spec):
        return format(x, spec) if isinstance(x, (int, float)) else "—"
    logger.info(
        "  metrics → VAMP-2=%s  |r|_Q=%s  |r|_RMSD=%s  |r|_Rg=%s  "
        "f/u_acc=%s  ITS=%s ns  basins=%s",
        fmt(row["vamp2"], ".3f"),
        fmt(row["pearson_Q"], ".2f"),
        fmt(row["pearson_RMSD"], ".2f"),
        fmt(row["pearson_Rg"], ".2f"),
        fmt(row["folded_acc"], ".2f"),
        fmt(row["its_ns"], ".1f"),
        fmt(row["basins"], "d"),
    )
    return row
