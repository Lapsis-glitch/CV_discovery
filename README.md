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
  — a 2 × 5 figure: top row = per-bin **mean** of each physical colouring in
  CV-space, bottom row = per-bin **σ normalized by the colouring's value
  range**, colorbar clipped at 10 % (values above show as the top
  "extend" arrow).
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

Each method's main plot is **two rows** of five hexbin panels over the first
two CVs — top row is the per-bin *mean*, bottom row the per-bin *relative σ*
(σ divided by the colouring's range, clipped at 10 %). Colourings are:

1. **Q** – fraction of native Cα–Cα contacts retained (Best–Hummer style,
   cutoff 8 Å, sequence separation ≥ 4, dₜ < 1.2 · d_native).
2. **Cα-RMSD to native** – native = frame 0 after global CA alignment.
3. **Radius of gyration** – over the CA selection.
4. **Trp6 burial** – count of non-Trp protein heavy atoms within 5 Å of the
   Trp6 sidechain (auto-detected: the first TRP in the first 10 residues).
5. **Helicity of residues 2–8** – fraction in the α-helical Ramachandran
   basin (−100 ≤ φ ≤ −30, −80 ≤ ψ ≤ 0).

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

---

## Methods by Subsection

1. **Linear CVs** (`01`) – PCA, tICA, LDA, HLDA, PCA-on-internal-coords
2. **Nonlinear features → linear projection** (`02`) – PCA-distances, tICA-dihedrals, KernelPCA-contacts, DiffusionMap+PCA
3. **Nonlinear manifold** (`03`) – Diffusion maps, Laplacian eigenmaps, Isomap, LLE, UMAP
4. **Deep-learning CVs** (`04`) – TAE, VAMPnet, Deep-tICA, Deep-LDA, VAE
5. **GNN CVs** (`05`) – EGNN-AE, SchNet-PCA, GraphVAMPnet, Latent-tICA

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
| GraphVAMPNet          | Ghorbani, Hoffmann & Ferguson, *J. Chem. Phys.* **156**, 184103 (2022) — [doi:10.1063/5.0085000](https://doi.org/10.1063/5.0085000) | graph cutoff **7.5 Å**; SchNet `hidden_channels = num_filters = 32`; `num_gaussians = 16`; 3 interaction blocks; state sweep **n ∈ {2, 5}** |
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

