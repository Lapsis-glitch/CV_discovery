"""
Data loading and featurisation for MD trajectories.

Supports: PDB / PSF / PRMTOP topologies,  DCD / XTC / TRR / NetCDF / TNG trajectories
(anything MDAnalysis can read).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, distances
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger(__name__)


# Ordered labels for the per-frame physical colourings we compute alongside
# the feature arrays.  Keys match the dict returned in TrajectoryFeatures.colorings.
COLORING_LABELS = [
    "Q (native contacts)",
    "Cα-RMSD to native (Å)",
    "Radius of gyration (Å)",
    "Trp6 burial",
    "Helicity (res 2–8)",
]


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryFeatures:
    """Container returned by :func:`featurize`.

    Three canonical featurizations are exposed so that every CV method can be
    compared on the same input basis. Unless a method has a methodological
    reason to use something else, it should default to :attr:`distances`.

    * :attr:`distances` – **primary**. All pairwise Cα–Cα distances, shape
      ``(n_frames, n_atoms·(n_atoms-1)/2)``. Matches the input used by the
      Ferguson / Bonati benchmarks (190-D for Trp-cage).
    * :attr:`dihedrals` – backbone φ/ψ encoded as
      ``[sin φ, cos φ, sin ψ, cos ψ]`` per residue, for methods that benefit
      from angular features.
    * :attr:`contacts`  – binary contact map from the same Cα pairs at
      an 8 Å cutoff.

    :attr:`cartesian` / :attr:`coords_3d` are retained for PCA-on-Cartesians
    and for geometry-aware GNN methods that need positions.
    """

    cartesian: np.ndarray
    coords_3d: np.ndarray
    distances: np.ndarray          # canonical "pairwise Cα distances"
    dihedrals: np.ndarray          # canonical "backbone sin/cos"
    contacts: np.ndarray           # canonical "Cα contact map"
    rmsd: np.ndarray
    n_atoms: int = 0
    frame_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    colorings: dict = field(default_factory=dict)

    # ---- canonical featurization accessors (for readability in scripts) ----
    @property
    def primary(self) -> np.ndarray:
        """The canonical input: all pairwise Cα–Cα distances."""
        return self.distances

    @property
    def sincos(self) -> np.ndarray:
        """Backbone φ/ψ as sin/cos pairs."""
        return self.dihedrals

    @property
    def contact_map(self) -> np.ndarray:
        """Binary Cα contact map at 8 Å."""
        return self.contacts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def load_universe(
    top: str,
    traj: str,
    selection: str = "protein and name CA",
) -> tuple[mda.Universe, mda.AtomGroup]:
    """Load topology + trajectory and return (Universe, selected AtomGroup)."""
    u = mda.Universe(top, traj)
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(
            f"Selection '{selection}' matched 0 atoms.  "
            "Check your topology and selection string."
        )
    logger.info(
        "Loaded %d atoms (%d total), %d frames",
        len(ag), len(u.atoms), len(u.trajectory),
    )
    return u, ag


def align_trajectory(u: mda.Universe, selection: str = "protein and name CA") -> None:
    """Align every frame to the first frame **in place** (loads into memory)."""
    ref = u.copy()
    align.AlignTraj(u, ref, select=selection, in_memory=True).run()
    logger.info("Alignment complete (reference = frame 0).")


def _compute_dihedrals(u: mda.Universe, frame_indices: np.ndarray) -> np.ndarray:
    """Backbone phi/psi as [sin φ, cos φ, sin ψ, cos ψ] per residue."""
    try:
        from MDAnalysis.analysis.dihedrals import Ramachandran

        protein = u.select_atoms("protein")
        if len(protein) == 0:
            raise ValueError("No protein atoms for dihedral computation.")

        rama = Ramachandran(protein).run(frames=frame_indices)
        angles_deg = rama.results.angles
        angles_rad = np.deg2rad(angles_deg)
        sin_vals = np.sin(angles_rad)
        cos_vals = np.cos(angles_rad)
        combined = np.concatenate([sin_vals, cos_vals], axis=-1)
        return combined.reshape(len(frame_indices), -1).astype(np.float32)
    except Exception as exc:
        logger.warning("Dihedral computation failed (%s); returning zeros.", exc)
        return np.zeros((len(frame_indices), 1), dtype=np.float32)


def _prepare_native_contacts(ref_pos: np.ndarray, cutoff: float = 8.0, seq_sep: int = 4):
    """Return (pairs, native_distances) for residue pairs forming the native map."""
    n = ref_pos.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=seq_sep)
    d0 = np.linalg.norm(ref_pos[i_idx] - ref_pos[j_idx], axis=1)
    mask = d0 < cutoff
    pairs = np.column_stack([i_idx[mask], j_idx[mask]])
    return pairs, d0[mask]


def _find_trp6(u: mda.Universe):
    """Locate a Trp residue in the first 10 residues (Trp-cage's Trp6).

    Returns (sidechain_ag, other_heavy_ag) or (None, None) if absent.
    """
    trp_residues = u.select_atoms("protein and resname TRP").residues
    if len(trp_residues) == 0:
        return None, None
    trp6 = None
    for r in trp_residues:
        if int(r.resid) <= 10:
            trp6 = r
            break
    if trp6 is None:
        trp6 = trp_residues[0]
    sidechain = trp6.atoms.select_atoms(
        "not (name N or name CA or name C or name O or name HN or name H) "
        "and not type H"
    )
    others = u.select_atoms(
        f"protein and not resid {int(trp6.resid)} and not type H"
    )
    if len(sidechain) == 0 or len(others) == 0:
        return None, None
    return sidechain, others


def _compute_helicity(u: mda.Universe, frame_indices: np.ndarray,
                      first: int = 2, last: int = 8) -> np.ndarray:
    """Fraction of residues in α-helical basin (per frame) over residues *first*..*last*."""
    from MDAnalysis.analysis.dihedrals import Ramachandran
    try:
        sel = u.select_atoms(f"protein and resid {first}-{last}")
        if len(sel.residues) == 0:
            return np.zeros(len(frame_indices), dtype=np.float32)
        rama = Ramachandran(sel).run(frames=frame_indices)
        ang = rama.results.angles  # (n_frames, n_res, 2) in degrees
        phi, psi = ang[..., 0], ang[..., 1]
        helix = (phi > -100) & (phi < -30) & (psi > -80) & (psi < 0)
        return helix.mean(axis=1).astype(np.float32)
    except Exception as exc:
        logger.warning("Helicity calculation failed (%s); returning zeros.", exc)
        return np.zeros(len(frame_indices), dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_native_reference(native: str, selection: str) -> np.ndarray:
    """Load a native structure file and return positions matching *selection*."""
    ref_u = mda.Universe(native)
    ref_ag = ref_u.select_atoms(selection)
    if len(ref_ag) == 0:
        raise ValueError(
            f"Native file '{native}' + selection '{selection}' matched 0 atoms."
        )
    logger.info("Loaded native reference from %s (%d atoms).", native, len(ref_ag))
    return ref_ag.positions.copy()


def featurize(
    top: str,
    traj: str,
    stride: int = 1,
    selection: str = "protein and name CA",
    contact_cutoff: float = 8.0,
    native: str | None = None,
) -> TrajectoryFeatures:
    """Load, align, compute features *and* physical per-frame colourings in one pass.

    If *native* is given it is used as the reference structure for alignment,
    Cα-RMSD, and the native-contact map; otherwise frame 0 is used.
    """
    u, ag = load_universe(top, traj, selection)

    if native is not None:
        ref_pos_full = _load_native_reference(native, selection)
        if ref_pos_full.shape != ag.positions.shape:
            raise ValueError(
                f"Native selection has {ref_pos_full.shape[0]} atoms but trajectory "
                f"selection has {ag.positions.shape[0]} – they must match."
            )
        # Align trajectory to the native structure (in memory).
        ref_u = mda.Universe(native)
        align.AlignTraj(u, ref_u, select=selection, in_memory=True).run()
        logger.info("Alignment complete (reference = native file).")
        ref_pos = ref_pos_full
    else:
        align_trajectory(u, selection)
        u.trajectory[0]
        ref_pos = ag.positions.copy()
        logger.info("No --native supplied; using frame 0 as native reference.")

    all_indices = np.arange(0, len(u.trajectory), stride)
    n_frames = len(all_indices)
    n_atoms = len(ag)
    n_pairs = n_atoms * (n_atoms - 1) // 2

    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    dist_arr = np.empty((n_frames, n_pairs), dtype=np.float32)
    rmsd_arr = np.empty(n_frames, dtype=np.float32)
    rg_arr = np.empty(n_frames, dtype=np.float32)
    q_arr = np.empty(n_frames, dtype=np.float32)
    burial_arr = np.empty(n_frames, dtype=np.float32)

    native_pairs, native_d = _prepare_native_contacts(ref_pos, cutoff=contact_cutoff)
    has_native = len(native_pairs) > 0

    trp_side, trp_others = _find_trp6(u)
    has_trp = trp_side is not None

    for idx, fi in enumerate(all_indices):
        u.trajectory[int(fi)]
        pos = ag.positions.copy()
        coords[idx] = pos
        dist_arr[idx] = distances.self_distance_array(pos)

        diff = pos - ref_pos
        rmsd_arr[idx] = np.sqrt((diff ** 2).sum() / n_atoms)

        com = pos.mean(axis=0)
        rg_arr[idx] = np.sqrt(((pos - com) ** 2).sum() / n_atoms)

        if has_native:
            cur_d = np.linalg.norm(
                pos[native_pairs[:, 0]] - pos[native_pairs[:, 1]], axis=1
            )
            q_arr[idx] = float((cur_d < 1.2 * native_d).mean())
        else:
            q_arr[idx] = 0.0

        if has_trp:
            d = distance_array(trp_side.positions, trp_others.positions)
            burial_arr[idx] = float((d < 5.0).sum())
        else:
            burial_arr[idx] = 0.0

    cartesian_flat = coords.reshape(n_frames, -1)
    contacts = (dist_arr < contact_cutoff).astype(np.float32)
    dihedrals = _compute_dihedrals(u, all_indices)
    helicity = _compute_helicity(u, all_indices, first=2, last=8)

    colorings = {
        "Q (native contacts)":      q_arr,
        "Cα-RMSD to native (Å)":    rmsd_arr,
        "Radius of gyration (Å)":   rg_arr,
        "Trp6 burial":              burial_arr,
        "Helicity (res 2–8)":       helicity,
    }

    logger.info(
        "Features ready – cartesian %s  distances %s  dihedrals %s  contacts %s",
        cartesian_flat.shape, dist_arr.shape, dihedrals.shape, contacts.shape,
    )
    logger.info(
        "Colourings ready – Q[min/mean/max]=%.2f/%.2f/%.2f  Rg_mean=%.2f  "
        "helicity_mean=%.2f  trp_buried_mean=%.1f",
        q_arr.min(), q_arr.mean(), q_arr.max(),
        rg_arr.mean(), helicity.mean(), burial_arr.mean(),
    )

    return TrajectoryFeatures(
        cartesian=cartesian_flat,
        coords_3d=coords,
        distances=dist_arr,
        dihedrals=dihedrals,
        contacts=contacts,
        rmsd=rmsd_arr,
        n_atoms=n_atoms,
        frame_indices=all_indices,
        colorings=colorings,
    )