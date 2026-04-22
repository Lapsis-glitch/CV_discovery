#!/usr/bin/env python
"""01 – Linear CV discovery methods.

Methods
-------
1. PCA on aligned Cartesians            (scikit-learn)
2. tICA on Cartesian features           (deeptime)
3. LDA with k-means-on-RMSD labels      (scikit-learn)
4. HLDA / harmonic linear discriminant   (numpy / scipy)
5. PCA on internal-coordinate covariance (scikit-learn)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_playground.io import featurize
from cv_playground.utils import (
    base_argparser,
    get_or_make_labels,
    log_effective_tau,
    run_method,
    set_run_context,
    set_seed,
    setup_logging,
)

SUBSECTION = "01_linear_cvs"


def run(args, feat=None) -> None:
    set_seed(args.seed)
    if feat is None:
        feat = featurize(args.top, args.traj, args.stride, args.selection, native=args.native)
    out = Path(args.output_dir) / SUBSECTION
    fi = feat.colorings
    log_effective_tau(args, "script 01 tICA")
    set_run_context(feat=feat, args=args, interpret_mode="pearson")

    # ── 1. PCA on aligned Cartesians ────────────────────────────────────────
    def pca_cartesian():
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=args.seed).fit_transform(feat.cartesian)

    run_method("PCA Cartesian", pca_cartesian, fi, out)

    # ── 2. tICA on pairwise Cα distances (canonical) ───────────────────────
    def tica_distances():
        from deeptime.decomposition import TICA
        model = TICA(lagtime=args.lag, dim=2).fit_fetch(feat.distances)
        return model.transform(feat.distances)

    run_method("tICA (Cα distances)", tica_distances, fi, out)

    # ── 3. LDA on pairwise distances with Q-based labels ───────────────────
    def lda_q():
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        labels = get_or_make_labels(args, feat)
        n_classes = len(np.unique(labels))

        X = feat.distances
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=args.seed).fit_transform(X)

        n_comp = min(2, n_classes - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        proj = lda.fit_transform(X, labels)

        if proj.shape[1] == 1:
            proj = np.column_stack([proj[:, 0], np.zeros(len(proj))])
        return proj

    run_method("LDA (Q-based labels)", lda_q, fi, out)

    # ── 4. HLDA – harmonic linear discriminant analysis ─────────────────────
    def hlda():
        """
        Fisher LDA on distance features with state labels from k-means.
        In a true HLDA workflow (Mendels et al. 2018) the states come from
        harmonic-restraint simulations; here k-means on RMSD is used as a
        demonstration placeholder.
        """
        from scipy.linalg import eigh

        labels = get_or_make_labels(args, feat)
        X = feat.distances.astype(np.float64)
        n_classes = len(np.unique(labels))

        grand_mean = X.mean(axis=0)
        d = X.shape[1]
        S_w = np.zeros((d, d))
        S_b = np.zeros((d, d))

        for c in np.unique(labels):
            Xc = X[labels == c]
            mu_c = Xc.mean(axis=0)
            diff = Xc - mu_c
            S_w += diff.T @ diff
            dm = (mu_c - grand_mean).reshape(-1, 1)
            S_b += len(Xc) * (dm @ dm.T)

        S_w += 1e-6 * np.eye(d)

        # Solve S_b v = λ S_w v  — take the two largest eigenvalues
        n_ev = min(2, n_classes - 1, d)
        eigvals, eigvecs = eigh(S_b, S_w, subset_by_index=[d - n_ev, d - 1])
        idx = np.argsort(eigvals)[::-1][:2]
        W = eigvecs[:, idx]
        proj = X @ W

        if proj.shape[1] < 2:
            proj = np.column_stack([proj, np.zeros(len(proj))])
        return proj

    run_method("HLDA", hlda, fi, out)

    # ── 5. PCA on internal-coordinate covariance ────────────────────────────
    def pca_internal():
        from sklearn.decomposition import PCA
        internal = np.hstack([feat.distances, feat.dihedrals])
        return PCA(n_components=2, random_state=args.seed).fit_transform(internal)

    run_method("PCA Internal Coords", pca_internal, fi, out)


def main() -> None:
    setup_logging()
    args = base_argparser("01 – Linear CV discovery").parse_args()
    run(args)


if __name__ == "__main__":
    main()

