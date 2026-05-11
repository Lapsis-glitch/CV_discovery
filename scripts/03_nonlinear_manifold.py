#!/usr/bin/env python
"""03 – Nonlinear manifold learning.

Methods
-------
1. Diffusion maps          (pydiffmap)
2. Laplacian eigenmaps      (scikit-learn SpectralEmbedding)
3. Isomap                   (scikit-learn)
4. Locally Linear Embedding (scikit-learn)
5. UMAP                     (umap-learn)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_playground.io import featurize_cached
from cv_playground.utils import (
    attach_committor,
    base_argparser,
    run_method,
    set_run_context,
    set_seed,
    setup_logging,
)

SUBSECTION = "03_nonlinear_manifold"


def run(args, feat=None) -> None:
    set_seed(args.seed)
    if feat is None:
        feat = featurize_cached(args.top, args.traj, args.stride, args.selection, native=args.native, cache_dir=getattr(args, "cache_dir", "outputs/cache"), read_cache=getattr(args, "read", False))
    out = Path(args.output_dir) / SUBSECTION
    attach_committor(feat)
    fi = feat.colorings
    X = feat.distances          # default high-dim input for all methods here
    set_run_context(feat=feat, args=args, interpret_mode="pearson")

    # ── 1. Diffusion maps — ε sweep (bgh auto + median-heuristic scales) ───
    from scipy.spatial.distance import pdist
    _sub = X
    if len(_sub) > 2000:
        _sub = _sub[np.random.default_rng(args.seed).choice(len(_sub), 2000, replace=False)]
    _med = float(np.median(pdist(_sub))) + 1e-12

    def _make_dmap(eps):
        def _fn():
            # k=100 neighbours and α=1/2 follow Zheng, Rohrdanz & Clementi 2013
            # (locally-scaled diffusion maps on Trp-cage).
            from pydiffmap import diffusion_map as dm
            mydmap = dm.DiffusionMap.from_sklearn(
                n_evecs=2, epsilon=eps, alpha=0.5, k=100,
            )
            return mydmap.fit_transform(X)
        return _fn

    run_method("Diffusion Maps (ε=bgh)", _make_dmap("bgh"), fi, out)
    for _scale in (0.5, 1.0, 2.0):
        _eps = (_scale * _med) ** 2
        run_method(f"Diffusion Maps (ε={_scale}·median²)", _make_dmap(_eps), fi, out)

    # ── 2–4. k sweeps for Laplacian / Isomap / LLE ─────────────────────────
    def _make_laplacian(k):
        def _fn():
            from sklearn.manifold import SpectralEmbedding
            return SpectralEmbedding(
                n_components=2, affinity="nearest_neighbors",
                n_neighbors=k, random_state=args.seed,
            ).fit_transform(X)
        return _fn

    def _make_isomap(k):
        def _fn():
            from sklearn.manifold import Isomap
            m = Isomap(n_components=2, n_neighbors=k).fit(X)
            return m.transform(X), lambda Xn: m.transform(Xn)
        return _fn

    def _make_lle(k):
        def _fn():
            from sklearn.manifold import LocallyLinearEmbedding
            m = LocallyLinearEmbedding(
                n_components=2, n_neighbors=k, random_state=args.seed,
            ).fit(X)
            return m.transform(X), lambda Xn: m.transform(Xn)
        return _fn

    for _k in (5, 15, 50):
        run_method(f"Laplacian Eigenmaps (k={_k})", _make_laplacian(_k), fi, out)
        run_method(f"Isomap (k={_k})",              _make_isomap(_k),     fi, out)
        run_method(f"LLE (k={_k})",                 _make_lle(_k),        fi, out)

    # ── 5. UMAP — visualization only, no k sweep ───────────────────────────
    def umap_embed():
        import umap
        return umap.UMAP(n_components=2, random_state=args.seed).fit_transform(X)

    # UMAP is a visualization tool, not a differentiable CV: it cannot be
    # used to bias a simulation. It is kept for comparison to the manifold
    # methods above but should *not* be interpreted as a candidate CV.
    run_method("UMAP (visualization only)", umap_embed, fi, out)


def main() -> None:
    setup_logging()
    args = base_argparser("03 – Nonlinear manifold learning").parse_args()
    run(args)


if __name__ == "__main__":
    main()

