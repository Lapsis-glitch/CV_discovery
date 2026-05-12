# CV Discovery Playground

A self-contained environment for comparing collective-variable (CV) discovery
methods on molecular-dynamics trajectories.

## Supported Formats

| Role       | Formats                                        |
|------------|------------------------------------------------|
| Topology   | PDB, PSF, PRMTOP (AMBER), GRO, TPR            |
| Trajectory | DCD, XTC, TRR, NetCDF (`.nc`), TNG, LAMMPSTRJ  |

All I/O goes through **MDAnalysis**, which handles these natively.

---

## Installation

### Option A – uv (recommended, fastest)

```bash
cd /path/to/CV_discovery
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

> **CUDA note:** for GPU-accelerated PyTorch install the matching wheel first:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> uv pip install -e .
> ```

### Option B – pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Option C – conda / mamba

```bash
conda create -n cvplay python=3.11 -y
conda activate cvplay
pip install -e .
```

---

## Quick Start

### Run everything at once

```bash
python run_all.py --top my_system.pdb --traj my_traj.xtc
```

The trajectory + topology are loaded and featurised **once**; each subsection
script is then imported and run in-process against the shared features.
Writes:

- per-method CV maps → `outputs/<subsection>/<method>.svg`
  — a 2 × 6 figure: top row = per-bin **mean** of each physical colouring in
  CV-space (viridis), bottom row = per-bin **σ normalized by the colouring's
  value range** (magma, easily distinguished from mean plots), colorbar
  clipped at 10 % (values above show as the top "extend" arrow). The last
  column is the **committor** (when the committor CSV is available): the
  mean panel uses a blue → white → red diverging map centred at 0.5, so
  basins of attraction read off immediately.
- per-method **FES contour** → `outputs/<subsection>/<method>_fes.svg`
  — free-energy landscape F = −k_BT ln P on the CV1/CV2 plane. By default
  P(x, y) is estimated with a **Gaussian KDE** (smooth, no empty-bin
  artefacts); switch to a classical 2-D histogram with `--fes-method hist`.
  Temperature defaults to 298 K (kT ≈ 0.593 kcal/mol); contour lines are
  drawn every **0.25 kcal/mol**, colorbar clipped at **10 kcal/mol**.
- per-method **Rg-vs-Q CV maps** → `outputs/<subsection>/<method>_rg_vs_q.svg`
  — a 2 × 2 figure on the (Q, Rg) plane: columns = CV1 / CV2, rows = mean /
  normalized σ (same 10 % colorbar cap). Shows how each CV varies across the
  standard folding coordinates.
- CV arrays               → `outputs/<subsection>/<method>.npy`
- quantitative scores      → `outputs/metrics.json`
  (VAMP-2 at τ, |Pearson r| with Q / Cα-RMSD / Rg, folded-vs-unfolded
  logistic-regression accuracy, 2-state MSM implied timescale, FES basin
  count — one row per `<subsection>/<method>`)
- comparison page          → `outputs/summary.html` (ranking table + plot
  gallery + per-method *What it is / Role in CV discovery / How to read the
  plot / Reference* blocks with links to primary papers, preferring Trp-cage
  applications where available)

### Per-frame colourings used for the scatter subplots

Each method's main plot is **two rows** of five-to-six hexbin panels over the
first two CVs — top row is the per-bin *mean* (viridis), bottom row the
per-bin *relative σ* (magma, σ divided by the colouring's range, clipped at
10 %). Colourings are:

1. **Q** – fraction of native Cα–Cα contacts retained (Best–Hummer style,
   cutoff 8 Å, sequence separation ≥ 4, dₜ < 1.2 · d_native).
2. **Cα-RMSD to native** – native = frame 0 after global CA alignment.
3. **Radius of gyration** – over the CA selection.
4. **Trp6 burial** – count of non-Trp protein heavy atoms within 5 Å of the
   Trp6 sidechain (auto-detected: the first TRP in the first 10 residues).
5. **Helicity of residues 2–8** – fraction in the α-helical Ramachandran
   basin (−100 ≤ φ ≤ −30, −80 ≤ ψ ≤ 0).
6. **Committor** *(optional, 6th column)* – loaded from
   `/home/rat/Nancy_D/GNN/TRP_gnn/trp_k10.0.csv` (column `committor`) and
   aligned to the trajectory frames via `feat.frame_indices`. The mean
   panel uses a diverging **blue → white → red** colormap centred at 0.5
   (`cmap="bwr"`, `vmin=0`, `vmax=1`) so folded (1.0) vs. unfolded (0.0)
   basins separate visually; the std panel keeps the standard
   magma/σ-normalised treatment. If the CSV is missing or its length is
   inconsistent with the trajectory, the column is silently skipped.

### Run a single subsection

```bash
python scripts/01_linear_cvs.py            --top sys.pdb --traj traj.xtc
python scripts/03_nonlinear_manifold.py     --top sys.pdb --traj traj.xtc --stride 5
python scripts/04_deep_learning_cvs.py      --top sys.pdb --traj traj.dcd --epochs 100
```

### Common CLI Flags

| Flag          | Default                 | Description                                    |
|---------------|-------------------------|------------------------------------------------|
| `--top`       | *(required)*            | Topology file                                  |
| `--traj`      | *(required)*            | Trajectory file                                |
| `--native`    | *(frame 0)*             | Native reference structure (PDB/GRO/…) used for alignment, Cα-RMSD, and the native-contact map |
| `--stride`    | `1`                     | Frame stride                                   |
| `--selection` | `protein and name CA`   | MDAnalysis atom-selection string               |
| `--seed`      | `42`                    | Random seed for reproducibility                |
| `--lag`       | `100`                   | Lag in **strided frames** for tICA / time-lagged methods (≈ 20 ns at DESRES dt, matching Ghorbani–Hoffmann–Ferguson 2022 GraphVAMPNet and Bonati–Piccini–Parrinello 2021 Deep-TICA on Trp-cage) |
| `--dt-ps`     | `200`                   | Time between saved frames in ps (DESRES Trp-cage default) |
| `--labels`    | *(none)*                | `.npy` with integer state labels (optional)    |
| `--epochs`    | `50`                    | Training epochs (deep-learning scripts)        |
| `--batch-size`| `32`                    | Batch size (deep-learning scripts)             |
| `--output-dir`| `outputs`               | Root output directory                          |
| `--read`      | *(off)*                 | Skip CV training; reload each method's cached `.npy` from `--output-dir` and rebuild the plots, metrics, and `summary.html` only. Featurisation still runs (needed for the colourings + metrics). Cached outputs that aren't found are logged and skipped. GNN residue-importance maps (script 05) rely on a live forward pass and are automatically skipped in this mode; Pearson-based interpretability still runs from the cached CVs. |
| `--fes-method`| `kde`                   | Density estimator for the per-method FES contour plot. `kde` (default) uses a Gaussian KDE on a 120 × 120 grid (subsampled to 50 k frames for speed) — smooth P(x, y) with **no empty-bin artefacts**. `hist` falls back to a 2-D histogram. |

---

## Methods by Subsection

1. **Linear CVs** (`01`) – PCA, tICA, LDA, HLDA, PCA-on-internal-coords
2. **Nonlinear features → linear projection** (`02`) – PCA-distances, tICA-dihedrals, KernelPCA-contacts, DiffusionMap+PCA
3. **Nonlinear manifold** (`03`) – Diffusion maps, Laplacian eigenmaps, Isomap, LLE, UMAP
4. **Deep-learning CVs** (`04`) – TAE, VAMPnet, Deep-tICA, Deep-LDA, VAE
5. **GNN CVs** (`05`) – EGNN-AE, SchNet-PCA, GraphVAMPnet (PyG SchNet),
   **GraphVAMPnet (paper SchNet)**, Latent-tICA

> ⚠️  Deep-learning and GNN scripts default to **50 epochs** and are
> *demonstration* runs, **not** converged production models.  Increase
> `--epochs` for real analysis.

---

## Paper-Aligned Trp-Cage Defaults

Where the reference paper for a method explicitly benchmarked Trp-cage, the
script-level defaults have been aligned to the settings reported there. For
methods not validated on Trp-cage by the original authors, the defaults are
left as generic sensible values.

| Method                | Paper (Trp-cage)                                           | Aligned settings                                                                                   |
|-----------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| GraphVAMPNet (PyG SchNet)    | Ghorbani, Hoffmann & Ferguson, *J. Chem. Phys.* **156**, 184103 (2022) — [doi:10.1063/5.0085607](https://doi.org/10.1063/5.0085607) | graph cutoff **7.5 Å**; SchNet `hidden_channels = num_filters = 32`; `num_gaussians = 16`; 3 interaction blocks; state sweep **n ∈ {2, 5}** |
| GraphVAMPNet (paper SchNet)  | Ghorbani, Hoffmann & Ferguson, *J. Chem. Phys.* **156**, 184103 (2022) — [doi:10.1063/5.0085607](https://doi.org/10.1063/5.0085607) | Faithful reproduction of the paper's **modified SchNet** (Table I): K-NN graph with `num_neighbors = 7` Cα atoms, `n_conv = 4` interaction blocks, `h_a = 16`, 12 Gaussians in [2 Å, 8 Å], `n_classes = 5`, `lr = 5e-4`, residual connections. Modification vs. vanilla SchNet: the continuous-filter convolution aggregates neighbors via a **learned attention softmax** (`softmax(conv · w_nbr)`) instead of a plain sum. Implementation: `cv_playground/schnet_graphvamp.py` |
| VAMPnet               | Mardt, Pasquali, Wu & Noé, *Nature Communications* **9**, 5 (2018) — [doi:10.1038/s41467-017-02388-1](https://doi.org/10.1038/s41467-017-02388-1) | MLP lobe = **5 × 100-unit ELU layers** → n-state softmax; state sweep **n ∈ {2, 6}**                |
| Diffusion maps        | Zheng, Rohrdanz & Clementi, *J. Phys. Chem. B* **117**, 12769 (2013) — [doi:10.1021/jp401911h](https://doi.org/10.1021/jp401911h) | `k = 100` nearest neighbours, α = ½, ε via `bgh` auto + 0.5·/1·/2·median² sweep                    |
| tICA / Deep-TICA lag  | Bonati, Piccini & Parrinello, *PNAS* **118**, e2113533118 (2021) — [doi:10.1073/pnas.2113533118](https://doi.org/10.1073/pnas.2113533118); Schwantes & Pande, *JCTC* **9**, 2000 (2013) — [doi:10.1021/ct300878a](https://doi.org/10.1021/ct300878a) | default `--lag = 100` frames (= **20 ns** at `--dt-ps 200`)                                         |

Other methods' references (used in the `summary.html` method blocks) but with
defaults left generic: PCA on distances / Cartesians / internal coordinates —
Juraszek & Bolhuis 2006 ([doi:10.1073/pnas.0603229103](https://doi.org/10.1073/pnas.0603229103)), Mu, Nguyen & Stock 2005, Paschek–Hempel–García 2008
([doi:10.1073/pnas.0805163105](https://doi.org/10.1073/pnas.0805163105)); HLDA/LDA — Mendels, Piccini & Parrinello 2018 ([doi:10.1021/acs.jpclett.8b00733](https://doi.org/10.1021/acs.jpclett.8b00733)),
Piccini, Mendels & Parrinello 2018 ([doi:10.1021/acs.jctc.8b00634](https://doi.org/10.1021/acs.jctc.8b00634)); Deep-LDA — Bonati, Rizzi &
Parrinello 2020 ([doi:10.1021/acs.jpclett.0c00535](https://doi.org/10.1021/acs.jpclett.0c00535)); Laplacian / Isomap — Das et al. 2006
([doi:10.1073/pnas.0603553103](https://doi.org/10.1073/pnas.0603553103)); LLE — Roweis & Saul 2000, Stamati et al. 2010
([doi:10.1002/prot.22691](https://doi.org/10.1002/prot.22691)); UMAP — Trozzi, Wang & Tao 2021 ([doi:10.1021/acs.jpcb.1c02081](https://doi.org/10.1021/acs.jpcb.1c02081));
Kernel PCA — Schölkopf, Smola & Müller 1998 ([doi:10.1162/089976698300017467](https://doi.org/10.1162/089976698300017467));
TAE — Wehmeyer & Noé 2018 ([doi:10.1063/1.5011399](https://doi.org/10.1063/1.5011399)); RAVE — Ribeiro et al. 2018, Wang, Ribeiro & Tiwary 2019
([doi:10.1038/s41467-019-11405-4](https://doi.org/10.1038/s41467-019-11405-4)); VAE/VDE — Hernández et al. 2018
([doi:10.1103/PhysRevE.97.062412](https://doi.org/10.1103/PhysRevE.97.062412)); EGNN — Satorras, Hoogeboom & Welling 2021; SchNet — Schütt et al. 2018
([doi:10.1063/1.5019779](https://doi.org/10.1063/1.5019779)).

---

## Recent updates

### Per-Cα importance + cluster sidecars (universal across methods)
Every method now emits per-Cα importance scalars and basin assignments
alongside the CV array, with **no per-method opt-in required** for the
data-driven metrics.

- **Cluster IDs sidecar** → `outputs/<subsection>/basins/{tag}_clusters.npy`,
  shape `(n_frames, 2)` columns `[watershed_id, hdbscan_id]`, `-1` = noise /
  unassigned. A widened companion `{tag}_with_clusters.npy` =
  `[CV1, CV2, watershed_id, hdbscan_id]` is also written so a single load
  gives you projections + basins. The plain `{tag}.npy` is unchanged.
- **Per-Cα importance CSV** → `outputs/<subsection>/interpret/{tag}_residue.csv`,
  one row per Cα with four scalars per CV:
  - `pearson_abs_cv{k}` — mean |Pearson(CV, pair-distance)| over partner
    pairs; linear-baseline.
  - `condmean_range_cv{k}` — `max_bin E[CV|bin] − min_bin E[CV|bin]`
    averaged over partner pairs, in raw CV units. The conditional-mean
    "swing" estimated from data.
  - `eta2_cv{k}` — normalised variance of that conditional mean, ∈ [0, 1].
    Universal nonlinear/non-monotonic importance; equals R² in the linear
    case, exceeds Pearson² for nonlinear maps.
  - `pdp_cv{k}` — model-based partial dependence (axis-sweep PDP querying
    the trained encoder on synthetic inputs). NaN for methods whose encoder
    has no clean out-of-sample `transform()`.
- **Per-Cα importance plot** → `interpret/{tag}_residue.svg` — bar chart
  per CV, η² (blue) and PDP (red) side-by-side when both exist.
- **`.npy` artefacts** of the residue arrays:
  `interpret/{tag}_residue_eta2.npy` (always) and
  `interpret/{tag}_residue_pdp.npy` (when model PDP ran).

#### Model-PDP coverage by method

PDP queries the trained encoder at synthetic inputs, so it requires a
callable `transform_fn`. Two PDP "spaces" are wired depending on each
method's native input:

| PDP space | Methods |
|-----------|---------|
| **Distances** (perturb one pair-distance over its 5–95 % quantile grid) | tICA / LDA / HLDA (01), PCA Distances / KernelPCA-contacts via contact threshold (02), Isomap / LLE (03), TAE / VAMPnet × 2 / Deep-tICA / Deep-LDA / VAE / RAVE (04) |
| **Cartesian** (axis-sweep on each Cα's x/y/z, averaged over axes) | PCA Cartesian (01), EGNN-AE / SchNet+PCA / GraphVAMPnet (PyG) × 2 / GraphVAMPnet (paper SchNet) (05) |
| **η² only** (no callable OOS transform) | PCA Internal (01), tICA-dihedrals / DiffusionMap+PCA (02), Diffusion Maps / Laplacian Eigenmaps / UMAP (03), Latent-tICA (05) |

PDP defaults: `n_grid = 10`, `n_sub = 500`, distance PDP restricted to the
top-30 pairs by η² to bound cost. Cartesian PDP is `n_atoms × 3` batched
forward calls per method — ~1 s for PCA, ~1–3 min per GNN method on CPU.

### Free-energy landscape (FES) contours
Every method now gets a dedicated `{tag}_fes.svg` next to its hexbin figure.

- F = −k_BT ln P on the CV1 / CV2 plane, minimum shifted to 0, colorbar
  clipped at **10 kcal/mol**, kT ≈ 0.593 kcal/mol (298 K).
- **Gaussian KDE by default** (120 × 120 grid, subsampled to 50 k frames for
  speed, Scott bandwidth). The smooth density means there are **no empty-bin
  artefacts** — basin boundaries are continuous. Opt into a histogram with
  `--fes-method hist`.
- Contour lines at **0.25 kcal/mol** spacing, drawn thicker
  (`linewidths=0.8, alpha=0.7`) so they read clearly on top of the filled
  contour.

### Committor panels
When `/home/rat/Nancy_D/GNN/TRP_gnn/trp_k10.0.csv` is present, the
`committor` column is aligned to the trajectory's `frame_indices` and
injected into `feat.colorings["Committor"]` by `attach_committor(feat)` —
called by **every** script (01 – 05), not just the GNN one. That adds a
6th column to every method's main hexbin figure:

- **mean row** uses a diverging **blue → white → red** colormap
  (`cmap="bwr"`, `vmin=0`, `vmax=1`), i.e. blue at committor 0 (unfolded),
  white at 0.5 (transition), red at 1 (folded).
- **std row** uses `magma`, the same treatment as every other coloring.

Missing CSV or a frame-index mismatch is logged and silently skipped.

### `--read` replay mode
`--read` (added to `base_argparser`, so every script and `run_all.py`
accept it) skips CV training and instead reloads each method's cached
`{out_dir}/{tag}.npy`, then regenerates the hexbin plot, FES contour,
Rg-vs-Q plot, metrics, Pearson-interpretability, and summary HTML. Methods
whose cache is missing are logged and skipped. Featurisation still runs so
the per-frame colourings and metric computations remain consistent.

### Reference links & settings audit in `summary.html`
- `_expl` in `cv_playground/utils.py` now accepts a list of
  `(short-label, url)` tuples and renders one clickable link per cited
  paper, so multi-paper entries (PCA Cartesians, PCA internal, tICA, HLDA,
  Diffusion maps, LLE, UMAP, RAVE, EGNN, Latent-tICA) no longer hide
  secondary references behind prose.
- Fixed the broken GraphVAMPNet DOI `10.1063/5.0085000` (404) →
  `10.1063/5.0085607` everywhere it appears (this README, method blocks,
  secondary citations).
- Added a dedicated hyperparameter card and method block for
  **GraphVAMPnet (paper SchNet)** so its Table-I configuration shows up
  in the HTML alongside the lighter-weight PyG variant.
- `HLDA / Fisher LDA on distances` hyperparameter line updated to reflect
  that labels are now Q-based first with k-means(RMSD) fallback (the
  actual behaviour of `get_or_make_labels`).

---

## Design Notes

| Concern           | Approach                                                        |
|-------------------|-----------------------------------------------------------------|
| Format support    | MDAnalysis handles all common MD formats                        |
| Robustness        | Every method wrapped by `run_method()` – catches exceptions, logs, continues |
| GPU               | Auto-detected via `torch.cuda.is_available()`                   |
| Reproducibility   | `set_seed()` seeds Python, NumPy, and PyTorch (incl. CUDA)     |
| Timing            | Each method timed and logged                                    |
| PyEMMA            | Skipped (install broken on modern Python); deeptime covers tICA/VAMP |
| PLUMED            | Optional dep – install with `pip install -e .[plumed]`          |
| Model weights     | Everything trains from scratch; nothing downloaded              |

