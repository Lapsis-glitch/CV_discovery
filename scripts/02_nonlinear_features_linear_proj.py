#!/usr/bin/env python
"""02 – Nonlinear features fed into linear projections.

Methods
-------
1. PCA on pairwise CA distances                  (scikit-learn)
2. tICA on dihedral sin/cos features             (deeptime)
3. Kernel PCA (RBF) on contact maps              (scikit-learn)
4. Diffusion maps → PCA on diffusion coordinates (pydiffmap + scikit-learn)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_playground.io import featurize
from cv_playground.utils import (
    base_argparser,
    log_effective_tau,
    run_method,
    set_run_context,
    set_seed,
    setup_logging,
)

SUBSECTION = "02_nonlinear_features_linear_proj"


def run(args, feat=None) -> None:
    set_seed(args.seed)
    if feat is None:
        feat = featurize(args.top, args.traj, args.stride, args.selection, native=args.native)
    out = Path(args.output_dir) / SUBSECTION
    fi = feat.colorings
    log_effective_tau(args, "script 02 tICA-dihedrals")
    set_run_context(feat=feat, args=args, interpret_mode="pearson")

    # ── 1. PCA on pairwise distances ────────────────────────────────────────
    def pca_distances():
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=args.seed).fit_transform(feat.distances)

    run_method("PCA Distances", pca_distances, fi, out)

    # ── 2. tICA on dihedral sin/cos ─���───────────────────────────────────────
    def tica_dihedrals():
        from deeptime.decomposition import TICA
        model = TICA(lagtime=args.lag, dim=2).fit_fetch(feat.dihedrals)
        return model.transform(feat.dihedrals)

    run_method("tICA Dihedrals", tica_dihedrals, fi, out)

    # ── 3. Kernel PCA (RBF) on contact maps — median-heuristic γ ───────────
    def kpca_contacts():
        from sklearn.decomposition import KernelPCA
        from scipy.spatial.distance import pdist
        sub = feat.contacts
        if len(sub) > 2000:
            idx = np.random.default_rng(args.seed).choice(len(sub), 2000, replace=False)
            sub = sub[idx]
        med_sq = float(np.median(pdist(sub, metric="sqeuclidean"))) + 1e-12
        gamma = 1.0 / (2.0 * med_sq)
        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=gamma,
                         random_state=args.seed)
        return kpca.fit_transform(feat.contacts)

    run_method("KernelPCA Contacts (median-γ)", kpca_contacts, fi, out)

    # ── 4. Diffusion map → PCA on diffusion coordinates ─────────────────────
    def diffmap_pca():
        from pydiffmap import diffusion_map as dm
        from sklearn.decomposition import PCA

        mydmap = dm.DiffusionMap.from_sklearn(
            n_evecs=10, epsilon="bgh", alpha=0.5, k=64,
        )
        dmap_coords = mydmap.fit_transform(feat.distances)
        return PCA(n_components=2, random_state=args.seed).fit_transform(dmap_coords)

    run_method("DiffusionMap + PCA", diffmap_pca, fi, out)


def main() -> None:
    setup_logging()
    args = base_argparser("02 – Nonlinear features + linear projection").parse_args()
    run(args)


if __name__ == "__main__":
    main()

