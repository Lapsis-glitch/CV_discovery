"""Modified SchNet + GraphVAMPNet architecture for Trp-cage.

Faithful reproduction of Ghorbani, Hoffmann & Ferguson, J. Chem. Phys. 156,
184103 (2022). The SchNet interaction block is modified: the continuous-filter
convolution replaces the standard sum aggregation over neighbors with a
learned attention softmax (see ``ContinuousFilterConv``).

Paper defaults for the 20-residue Trp-cage TC10b (Table I):
    n_conv=4, h_a=16, num_neighbors=7, dmin=2 Å, dmax=8 Å, 12 Gaussians,
    n_classes=5, batch=1000, lr=5e-4, τ=20 ns.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDistance:
    def __init__(self, dmin: float, dmax: float, step: float, var: float | None = None):
        assert dmin < dmax and dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step)
        self.num_features = len(self.filter)
        self.var = var if var is not None else step

    def expand(self, distance: torch.Tensor) -> torch.Tensor:
        f = self.filter.to(distance.device)
        return torch.exp(-((distance.unsqueeze(-1) - f) ** 2) / self.var ** 2)


class ContinuousFilterConv(nn.Module):
    """Modified CFConv: attention-softmax pooling over neighbors."""

    def __init__(self, n_gaussians: int, n_filters: int, activation=nn.Tanh()):
        super().__init__()
        self.filter_generator = nn.Sequential(
            nn.Linear(n_gaussians, n_filters), activation,
            nn.Linear(n_filters, n_filters),
        )
        self.nbr_filter = nn.Parameter(torch.empty(n_filters, 1))
        nn.init.xavier_uniform_(self.nbr_filter, gain=1.414)

    def forward(self, features, rbf, nbr_list):
        # features: [B, N, F], rbf: [B, N, M, G], nbr_list: [B, N, M]
        conv_filter = self.filter_generator(rbf)                      # [B,N,M,F]
        B, N, M = nbr_list.shape
        idx = nbr_list.long().reshape(B, N * M, 1).expand(-1, -1, features.size(2))
        nbr_feats = torch.gather(features, 1, idx).reshape(B, N, M, -1)
        conv = nbr_feats * conv_filter                                # [B,N,M,F]
        attn = F.softmax((conv @ self.nbr_filter).squeeze(-1), dim=-1)  # [B,N,M]
        out = torch.einsum("bnm,bnmf->bnf", attn, conv)               # [B,N,F]
        return out, attn


class InteractionBlock(nn.Module):
    def __init__(self, n_inputs: int, n_gaussians: int, n_filters: int,
                 activation=nn.Tanh()):
        super().__init__()
        self.initial_dense = nn.Linear(n_inputs, n_filters, bias=False)
        self.cfconv = ContinuousFilterConv(n_gaussians, n_filters, activation)
        self.output_dense = nn.Sequential(
            nn.Linear(n_filters, n_filters), activation,
            nn.Linear(n_filters, n_filters),
        )

    def forward(self, features, rbf, nbr_list):
        x = self.initial_dense(features)
        x, attn = self.cfconv(x, rbf, nbr_list)
        return self.output_dense(x), attn


class GraphVampNetSchNet(nn.Module):
    """Modified-SchNet GraphVAMPNet (paper Trp-cage defaults)."""

    def __init__(self, num_atoms: int, num_neighbors: int = 7, n_classes: int = 5,
                 n_conv: int = 4, h_a: int = 16, h_g: int | None = None,
                 dmin: float = 2.0, dmax: float = 8.0, step: float = 0.5,
                 residual: bool = True):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_neighbors = num_neighbors
        self.residual = residual
        self.gauss = GaussianDistance(dmin, dmax, step)
        n_gauss = self.gauss.num_features

        self.atom_emb = nn.Embedding(num_atoms, h_a)
        self.atom_emb.weight.data.normal_()

        self.convs = nn.ModuleList([
            InteractionBlock(h_a, n_gauss, h_a, activation=nn.Tanh())
            for _ in range(n_conv)
        ])
        self.conv_activation = nn.ReLU()

        if h_g is not None:
            self.amino_emb = nn.Linear(h_a, h_g)
            self.fc_classes = nn.Linear(h_g, n_classes)
        else:
            self.amino_emb = None
            self.fc_classes = nn.Linear(h_a, n_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: [B, N, 2M] — first half distances, second half neighbor indices
        M = data.shape[-1] // 2
        nbr_dist = data[:, :, :M]
        nbr_list = data[:, :, M:]
        B, N, _ = nbr_list.shape

        rbf = self.gauss.expand(nbr_dist)                              # [B,N,M,G]
        idx = torch.arange(N, device=data.device).expand(B, N)
        h = self.atom_emb(idx)                                         # [B,N,h_a]

        for conv in self.convs:
            h_new, _ = conv(h, rbf, nbr_list)
            h = h + h_new if self.residual else h_new

        h = self.conv_activation(h)
        g = h.mean(dim=1)                                              # [B,h_a]
        if self.amino_emb is not None:
            g = self.amino_emb(g)
        return F.softmax(self.fc_classes(g), dim=-1)                   # [B,n_classes]


def build_knn_tensor(coords_3d: np.ndarray, num_neighbors: int) -> torch.Tensor:
    """(n_frames, n_atoms, 3) → (n_frames, n_atoms, 2*num_neighbors) tensor.

    For each atom, returns the ``num_neighbors`` nearest neighbors by distance,
    concatenating [distances | neighbor indices] along the last axis — the
    input format expected by :class:`GraphVampNetSchNet`.
    """
    from scipy.spatial import cKDTree
    n_frames, n_atoms, _ = coords_3d.shape
    k = num_neighbors + 1  # skip self
    dists = np.empty((n_frames, n_atoms, num_neighbors), dtype=np.float32)
    inds = np.empty((n_frames, n_atoms, num_neighbors), dtype=np.float32)
    for f in range(n_frames):
        tree = cKDTree(coords_3d[f])
        d, i = tree.query(coords_3d[f], k=k)
        dists[f] = d[:, 1:]
        inds[f] = i[:, 1:]
    return torch.from_numpy(np.concatenate([dists, inds], axis=-1))
