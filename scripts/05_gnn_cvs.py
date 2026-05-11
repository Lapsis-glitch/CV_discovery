#!/usr/bin/env python
"""05 – GNN-based collective variables.
Methods
-------
1. EGNN autoencoder           (from-scratch equivariant message passing)
2. SchNet-style encoder + PCA (torch_geometric SchNet)
3. GraphVAMPnet               (GCN encoder + VAMP-2 loss)
4. Latent-space tICA          (deeptime tICA on EGNN embeddings)
Training runs until early stopping on training loss;
``--epochs`` is the max cap and ``--patience`` controls the stop.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_playground.io import featurize_cached
from cv_playground.utils import (
    EarlyStopping,
    attach_committor,
    base_argparser,
    get_device,
    log_effective_tau,
    run_method,
    set_run_context,
    set_seed,
    setup_logging,
)
logger = logging.getLogger(__name__)
SUBSECTION = "05_gnn_cvs"
# ─────────────────────── graph construction ─────────────────────────────────
def _build_graph_dataset(coords_3d: np.ndarray, cutoff: float = 10.0):
    """(n_frames, n_atoms, 3) → list[torch_geometric.data.Data]."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph
    dataset = []
    n_atoms = coords_3d.shape[1]
    h = torch.ones(n_atoms, 1, dtype=torch.float32)
    z = torch.zeros(n_atoms, dtype=torch.long)
    for frame in coords_3d:
        pos = torch.as_tensor(frame, dtype=torch.float32)
        edge_index = radius_graph(pos, r=cutoff, loop=False)
        dataset.append(Data(x=h.clone(), z=z.clone(), pos=pos, edge_index=edge_index))
    return dataset
def _collate_pairs(dataset, lag: int):
    return list(zip(dataset[:-lag], dataset[lag:]))
# ─────────────────────── EGNN building blocks ───────────────────────────────
def _pairwise_distances(pos_bg):
    """Per-graph upper-triangle pairwise distances, pos_bg shape (B, N, 3)."""
    import torch
    diff = pos_bg.unsqueeze(2) - pos_bg.unsqueeze(1)      # (B, N, N, 3)
    D = diff.norm(dim=-1)                                 # (B, N, N)
    n = pos_bg.size(1)
    iu, ju = torch.triu_indices(n, n, offset=1)
    return D[:, iu, ju]                                    # (B, n*(n-1)/2)


def _make_egnn_ae(n_atoms: int, node_dim: int = 32, latent_dim: int = 2):
    """Minimal E(n)-equivariant autoencoder.

    The decoder now predicts the (invariant) upper-triangle pairwise
    distance vector — rotation/translation-invariant, so the model is not
    asked to produce equivariant Cartesians from an invariant latent.
    """
    import torch
    import torch.nn as nn
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import scatter
    class EGNNConv(MessagePassing):
        def __init__(self, in_dim, hidden_dim):
            super().__init__(aggr="add")
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * in_dim + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, in_dim),
            )
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        def forward(self, h, pos, edge_index):
            row, col = edge_index
            diff = pos[row] - pos[col]
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
            edge_feat = torch.cat([h[row], h[col], dist_sq], dim=-1)
            m = self.edge_mlp(edge_feat)
            coord_w = self.coord_mlp(m)
            coord_delta = scatter(diff * coord_w, row, dim=0,
                                  reduce="mean", dim_size=h.size(0))
            pos_new = pos + coord_delta
            agg = scatter(m, row, dim=0, reduce="sum", dim_size=h.size(0))
            h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
            return h_new, pos_new
    class EGNNAutoencoder(nn.Module):
        def __init__(self, n_atoms_: int, node_dim_: int, latent_dim_: int):
            super().__init__()
            self.embed = nn.Linear(1, node_dim_)
            self.conv1 = EGNNConv(node_dim_, node_dim_ * 2)
            self.conv2 = EGNNConv(node_dim_, node_dim_ * 2)
            self.to_latent = nn.Sequential(
                nn.Linear(node_dim_, 32), nn.ReLU(), nn.Linear(32, latent_dim_),
            )
            n_pairs_ = n_atoms_ * (n_atoms_ - 1) // 2
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim_, 64), nn.ReLU(),
                nn.Linear(64, 128),         nn.ReLU(),
                nn.Linear(128, n_pairs_),
            )
        def encode(self, data):
            from torch_geometric.nn import global_mean_pool
            h = self.embed(data.x)
            h, pos = self.conv1(h, data.pos, data.edge_index)
            h, pos = self.conv2(h, pos, data.edge_index)
            batch = data.batch if data.batch is not None else \
                torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            g = global_mean_pool(h, batch)
            return self.to_latent(g)
        def forward(self, data):
            z = self.encode(data)
            return self.decoder(z), z
    return EGNNAutoencoder(n_atoms, node_dim, latent_dim)
# ─────────────────────── VAMP-2 loss ────────────────────────────────────────
def _vamp2_loss(chi_t, chi_tau):
    """Negative VAMP-2 score (minimise to maximise VAMP-2)."""
    import torch
    chi_t   = chi_t   - chi_t.mean(0)
    chi_tau = chi_tau  - chi_tau.mean(0)
    n = chi_t.shape[0]
    d = chi_t.shape[1]
    eye = torch.eye(d, device=chi_t.device)
    C00 = chi_t.T   @ chi_t   / n + 1e-6 * eye
    C11 = chi_tau.T  @ chi_tau  / n + 1e-6 * eye
    C01 = chi_t.T   @ chi_tau  / n
    L00 = torch.linalg.cholesky(C00)
    L11 = torch.linalg.cholesky(C11)
    C00_inv_half = torch.linalg.inv(L00)
    C11_inv_half = torch.linalg.inv(L11)
    K = C00_inv_half @ C01 @ C11_inv_half.T
    return -torch.trace(K.T @ K)
# ─────────────────────── main ───────────────────────────────────────────────
def run(args, feat=None) -> None:
    set_seed(args.seed)
    device = get_device()
    if feat is None:
        feat = featurize_cached(args.top, args.traj, args.stride, args.selection, native=args.native, cache_dir=getattr(args, "cache_dir", "outputs/cache"), read_cache=getattr(args, "read", False))
    out = Path(args.output_dir) / SUBSECTION
    attach_committor(feat)
    fi = feat.colorings
    n_atoms = feat.n_atoms
    log_effective_tau(args, "script 05 time-lagged GNN")
    # GNN interpretability is manual (autograd w.r.t. pos); SchNet+PCA and
    # Latent-tICA get a Pearson pseudo-loading as a post-hoc fallback.
    set_run_context(feat=feat, args=args)

    from cv_playground import interpret
    gnn_forwards: dict = {}

    def _coords_to_pyg_batch(coords_np, cutoff, dev):
        """(B, n_atoms, 3) numpy → batched PyG graph on device."""
        import torch
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import radius_graph
        chunks = []
        h = torch.ones(n_atoms, 1, dtype=torch.float32)
        z = torch.zeros(n_atoms, dtype=torch.long)
        for frame in coords_np:
            pos = torch.as_tensor(frame, dtype=torch.float32)
            ei = radius_graph(pos, r=cutoff, loop=False)
            chunks.append(Data(x=h.clone(), z=z.clone(), pos=pos, edge_index=ei))
        return Batch.from_data_list(chunks).to(dev)
    # cutoff 7.5 Å follows Ghorbani, Hoffmann & Ferguson 2022 (GraphVAMPNet)
    # for the Trp-cage benchmark. Cα graph, no self-loops.
    logger.info("Building molecular graphs (cutoff = 7.5 A, per GraphVAMPNet/Trp-cage) ...")
    graph_data = _build_graph_dataset(feat.coords_3d, cutoff=7.5)
    logger.info("Built %d graphs (%d atoms each).", len(graph_data), n_atoms)
    # ── 1. EGNN Autoencoder ─────────────────────────────────────────────────
    def egnn_ae():
        import torch, torch.nn as nn
        from torch_geometric.loader import DataLoader as GeoLoader
        model = _make_egnn_ae(n_atoms, node_dim=32, latent_dim=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = GeoLoader(graph_data, batch_size=args.batch_size, shuffle=True)
        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta, name="EGNN-AE")
        for epoch in range(args.epochs):
            total = 0.0
            for batch in loader:
                batch = batch.to(device)
                pos_bg = batch.pos.view(batch.num_graphs, n_atoms, 3)
                target = _pairwise_distances(pos_bg)
                recon, z = model(batch)
                loss = nn.functional.mse_loss(recon, target)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * batch.num_graphs
            mean_loss = total / len(graph_data)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                logger.info("  EGNN-AE epoch %d/%d  loss=%.4f",
                            epoch + 1, args.epochs, mean_loss)
            if stopper.step(mean_loss, epoch):
                break
        model.eval()
        seq = GeoLoader(graph_data, batch_size=args.batch_size, shuffle=False)
        zs = []
        with torch.no_grad():
            for batch in seq:
                batch = batch.to(device)
                _, z = model(batch)
                zs.append(z.cpu().numpy())
        gnn_forwards["EGNN Autoencoder"] = (lambda b, _m=model: _m(b)[1], device)

        def _tf(coords_np, _m=model, _dev=device):
            import torch as _t
            _m.eval()
            batch = _coords_to_pyg_batch(coords_np, cutoff=10.0, dev=_dev)
            with _t.no_grad():
                _, z = _m(batch)
            z = z.detach().cpu().numpy()
            return z[:, :2] if z.shape[1] > 2 else z

        return np.concatenate(zs, axis=0), _tf, "cartesian"
    egnn_cvs = run_method("EGNN Autoencoder", egnn_ae, fi, out)
    if egnn_cvs is not None and "EGNN Autoencoder" in gnn_forwards:
        fwd, dev = gnn_forwards["EGNN Autoencoder"]
        interpret.gnn_residue_importance(
            fwd, graph_data, n_atoms,
            save_path=out / "interpret" / "egnn_autoencoder.png",
            title="EGNN-AE — mean |∂CV/∂pos| per residue",
            device=dev,
        )
    # ── 2. SchNet-style encoder + PCA ───────────────────────────────────────
    def schnet_pca():
        import torch, torch.nn as nn
        from torch_geometric.loader import DataLoader as GeoLoader
        from torch_geometric.nn import SchNet
        from torch_geometric.utils import scatter
        from sklearn.decomposition import PCA
        embed_dim = 16
        class SchNetEmbed(nn.Module):
            def __init__(self, hidden=64, out=16):
                super().__init__()
                self.schnet = SchNet(
                    hidden_channels=hidden, num_filters=hidden,
                    num_interactions=3, num_gaussians=25, cutoff=10.0,
                )
                self.head = nn.Linear(1, out)
            def forward(self, z, pos, batch):
                energy = self.schnet(z, pos, batch).view(-1, 1)
                return self.head(energy)
        model = SchNetEmbed(hidden=64, out=embed_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = GeoLoader(graph_data, batch_size=args.batch_size, shuffle=True)
        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta, name="SchNet+PCA")
        for epoch in range(args.epochs):
            total = 0.0; n_seen = 0
            for batch in loader:
                batch = batch.to(device)
                emb = model(batch.z, batch.pos, batch.batch)
                norms = batch.pos.norm(dim=-1)
                tgt = scatter(norms, batch.batch, reduce="mean").unsqueeze(-1)
                tgt = tgt.expand_as(emb)
                loss = nn.functional.mse_loss(emb, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * batch.num_graphs; n_seen += batch.num_graphs
            if stopper.step(total / max(1, n_seen), epoch):
                break
        model.eval()
        embs = []
        seq = GeoLoader(graph_data, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in seq:
                batch = batch.to(device)
                embs.append(model(batch.z, batch.pos, batch.batch).cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        pca = PCA(n_components=2, random_state=args.seed).fit(embs)
        cvs = pca.transform(embs)

        def _tf(coords_np, _m=model, _pca=pca, _dev=device):
            import torch as _t
            _m.eval()
            batch = _coords_to_pyg_batch(coords_np, cutoff=10.0, dev=_dev)
            with _t.no_grad():
                e = _m(batch.z, batch.pos, batch.batch).cpu().numpy()
            return _pca.transform(e)

        return cvs, _tf, "cartesian"
    schnet_cvs = run_method("SchNet + PCA", schnet_pca, fi, out)
    if schnet_cvs is not None:
        interpret.pearson_loading_map(
            schnet_cvs, feat.distances, n_atoms,
            save_path=out / "interpret" / "schnet_pca.png",
            title="SchNet + PCA — Pearson |corr(CV, pairwise-distance)|",
        )
    # ── 3. GraphVAMPnet: SchNet encoder + VAMP-2 loss, sweep n_states ──────
    def _make_graph_vampnet(out_dim: int, reg_name: str):
        def _fn():
            import torch, torch.nn as nn
            from torch_geometric.loader import DataLoader as GeoLoader
            from torch_geometric.nn import SchNet
            from torch_geometric.data import Batch

            # SchNet defaults from Ghorbani, Hoffmann & Ferguson 2022 (Trp-cage):
            # hidden=32, 3 interaction blocks, 16 RBFs, cutoff 7.5 Å.
            class SchNetLobe(nn.Module):
                def __init__(self, hidden=32, out=2):
                    super().__init__()
                    self.schnet = SchNet(
                        hidden_channels=hidden, num_filters=hidden,
                        num_interactions=3, num_gaussians=16, cutoff=7.5,
                    )
                    self.head = nn.Sequential(
                        nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, out),
                    )
                def forward(self, data):
                    energy = self.schnet(data.z, data.pos, data.batch).view(-1, 1)
                    return self.head(energy)

            lobe = SchNetLobe(hidden=32, out=out_dim).to(device)
            opt = torch.optim.Adam(lobe.parameters(), lr=1e-3)
            pairs = _collate_pairs(graph_data, args.lag)
            bs = args.batch_size
            lobe.train()
            stopper = EarlyStopping(args.patience, args.min_delta, name=reg_name)
            for epoch in range(args.epochs):
                np.random.shuffle(pairs)
                epoch_loss = 0.0; n_batches = 0
                for i in range(0, len(pairs), bs):
                    chunk = pairs[i:i + bs]
                    if len(chunk) < 2:
                        continue
                    bt   = Batch.from_data_list([p[0] for p in chunk]).to(device)
                    btau = Batch.from_data_list([p[1] for p in chunk]).to(device)
                    loss = _vamp2_loss(lobe(bt), lobe(btau))
                    opt.zero_grad(); loss.backward(); opt.step()
                    epoch_loss += loss.item(); n_batches += 1
                if n_batches and stopper.step(epoch_loss / n_batches, epoch):
                    break
            lobe.eval()
            seq = GeoLoader(graph_data, batch_size=args.batch_size, shuffle=False)
            zs = []
            with torch.no_grad():
                for batch in seq:
                    zs.append(lobe(batch.to(device)).cpu().numpy())
            gnn_forwards[reg_name] = (lambda b, _l=lobe: _l(b), device)
            cvs = np.concatenate(zs, axis=0)
            cvs = cvs[:, :2] if cvs.shape[1] > 2 else cvs

            def _tf(coords_np, _l=lobe, _dev=device):
                import torch as _t
                _l.eval()
                batch = _coords_to_pyg_batch(coords_np, cutoff=7.5, dev=_dev)
                with _t.no_grad():
                    z = _l(batch).cpu().numpy()
                return z[:, :2] if z.shape[1] > 2 else z

            return cvs, _tf, "cartesian"
        return _fn

    # n ∈ {2, 5}: 5-state assignment is the headline Trp-cage result in
    # Ghorbani, Hoffmann & Ferguson 2022 (GraphVAMPNet).
    for _n_states in (2, 5):
        _gv_name = f"GraphVAMPnet SchNet (n={_n_states})"
        gvnet_cvs = run_method(_gv_name, _make_graph_vampnet(_n_states, _gv_name), fi, out)
        if gvnet_cvs is not None and _gv_name in gnn_forwards:
            fwd, dev = gnn_forwards[_gv_name]
            _tag = _gv_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            interpret.gnn_residue_importance(
                fwd, graph_data, n_atoms,
                save_path=out / "interpret" / f"{_tag}.png",
                title=f"{_gv_name} — mean |∂CV/∂pos| per residue",
                device=dev,
            )
    # ── 3b. GraphVAMPNet w/ paper-faithful modified SchNet (Trp-cage) ──────
    def graph_vampnet_paper_schnet():
        import torch
        from cv_playground.schnet_graphvamp import (
            GraphVampNetSchNet, build_knn_tensor,
        )
        # Paper (Ghorbani/Hoffmann/Ferguson 2022) Table I defaults:
        #   n_conv=4, h_a=16, num_neighbors=7, 12 Gaussians in [2,8] Å,
        #   n_classes=5, batch=1000, lr=5e-4.
        num_neighbors = min(7, n_atoms - 1)
        knn = build_knn_tensor(feat.coords_3d, num_neighbors).to(device)
        model = GraphVampNetSchNet(
            num_atoms=n_atoms, num_neighbors=num_neighbors, n_classes=5,
            n_conv=4, h_a=16, dmin=2.0, dmax=8.0, step=0.5, residual=True,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        bs = min(1000, max(16, args.batch_size))
        n_frames = knn.shape[0]
        lag = args.lag
        idx_all = np.arange(n_frames - lag)
        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta,
                                name="GraphVAMPnet (paper SchNet)")
        for epoch in range(args.epochs):
            np.random.shuffle(idx_all)
            total = 0.0; nb = 0
            for i in range(0, len(idx_all), bs):
                sel = idx_all[i:i + bs]
                if len(sel) < 2:
                    continue
                bt = knn[sel]
                btau = knn[sel + lag]
                loss = _vamp2_loss(model(bt), model(btau))
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); nb += 1
            if nb and stopper.step(total / nb, epoch):
                break
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, n_frames, bs):
                probs.append(model(knn[i:i + bs]).cpu().numpy())
        probs = np.concatenate(probs, axis=0)

        def _tf(coords_np, _m=model, _dev=device, _nn=num_neighbors):
            import torch as _t
            _m.eval()
            knn_pert = build_knn_tensor(np.asarray(coords_np, dtype=np.float32),
                                        _nn).to(_dev)
            with _t.no_grad():
                p = _m(knn_pert).cpu().numpy()
            return p[:, :2]

        return probs[:, :2], _tf, "cartesian"
    run_method("GraphVAMPnet (paper SchNet)", graph_vampnet_paper_schnet, fi, out)

    # ── 4. Latent-space tICA on EGNN embeddings ────────────────────────────
    def latent_tica():
        if egnn_cvs is None:
            raise RuntimeError("EGNN-AE produced no embeddings.")
        from deeptime.decomposition import TICA
        logger.info("Running tICA on %d-D EGNN latent.", egnn_cvs.shape[1])
        model = TICA(lagtime=args.lag, dim=2).fit_fetch(egnn_cvs)
        return model.transform(egnn_cvs)
    latent_cvs = run_method("Latent tICA (EGNN)", latent_tica, fi, out)
    if latent_cvs is not None:
        interpret.pearson_loading_map(
            latent_cvs, feat.distances, n_atoms,
            save_path=out / "interpret" / "latent_tica.png",
            title="Latent tICA (EGNN) — Pearson |corr(CV, pairwise-distance)|",
        )


def main() -> None:
    setup_logging()
    args = base_argparser("05 – GNN-based CVs").parse_args()
    run(args)


if __name__ == "__main__":
    main()
