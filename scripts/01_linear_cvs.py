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

from cv_playground.io import featurize_cached
from cv_playground.utils import (
    attach_committor,
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
        feat = featurize_cached(args.top, args.traj, args.stride, args.selection, native=args.native, cache_dir=getattr(args, "cache_dir", "outputs/cache"), read_cache=getattr(args, "read", False))
    out = Path(args.output_dir) / SUBSECTION
    attach_committor(feat)
    fi = feat.colorings
    log_effective_tau(args, "script 01 tICA")
    set_run_context(feat=feat, args=args, interpret_mode="pearson")

    # ── 1. PCA on aligned Cartesians ────────────────────────────────────────
    # Native input is flattened Cartesians → axis-sweep PDP in xyz space.
    def pca_cartesian():
        from sklearn.decomposition import PCA
        m = PCA(n_components=2, random_state=args.seed).fit(feat.cartesian)

        def _tf(coords3d):
            X = np.asarray(coords3d).reshape(len(coords3d), -1)
            return m.transform(X)

        return m.transform(feat.cartesian), _tf, "cartesian"

    run_method("PCA Cartesian", pca_cartesian, fi, out)

    # ── 2. tICA on pairwise Cα distances (canonical) ───────────────────────
    def tica_distances():
        from deeptime.decomposition import TICA
        model = TICA(lagtime=args.lag, dim=2).fit_fetch(feat.distances)
        return model.transform(feat.distances), lambda X: model.transform(X)

    run_method("tICA (Cα distances)", tica_distances, fi, out)

    # ── 3. LDA on pairwise distances with Q-based labels ───────────────────
    def lda_q():
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        labels = get_or_make_labels(args, feat)
        n_classes = len(np.unique(labels))

        X_raw = feat.distances
        pca50 = None
        X = X_raw
        if X.shape[1] > 50:
            pca50 = PCA(n_components=50, random_state=args.seed).fit(X_raw)
            X = pca50.transform(X_raw)

        n_comp = min(2, n_classes - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        proj = lda.fit_transform(X, labels)

        def _transform(Xnew):
            Z = pca50.transform(Xnew) if pca50 is not None else Xnew
            p = lda.transform(Z)
            if p.shape[1] == 1:
                p = np.column_stack([p[:, 0], np.zeros(len(p))])
            return p

        if proj.shape[1] == 1:
            proj = np.column_stack([proj[:, 0], np.zeros(len(proj))])
        return proj, _transform

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

        def _transform(Xnew):
            p = Xnew.astype(np.float64) @ W
            if p.shape[1] < 2:
                p = np.column_stack([p, np.zeros(len(p))])
            return p

        return proj, _transform

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

