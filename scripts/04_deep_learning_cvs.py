#!/usr/bin/env python
"""04 – Deep-learning collective variables.

Methods
-------
1. Time-lagged Autoencoder (TAE)          – pure PyTorch
2. VAMPnet                                – deeptime
3. Deep-tICA                              – mlcolvar
4. Deep-LDA  (k-means labels fallback)    – mlcolvar
5. Variational Autoencoder (VAE)          – pure PyTorch   (bonus)

Training runs until early stopping (monitor: training loss / valid_loss for
mlcolvar). ``--epochs`` is a max cap, ``--patience`` controls the stop.
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
    get_or_make_labels,
    log_effective_tau,
    run_method,
    set_run_context,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)
SUBSECTION = "04_deep_learning_cvs"


# ── helpers ─────────────────────────────────────────────────────────────────

def _time_lagged_dataset(X: np.ndarray, lag: int):
    """Return (X_t, X_{t+lag})."""
    return X[:-lag].copy(), X[lag:].copy()


def _to_tensor(arr: np.ndarray):
    import torch
    return torch.as_tensor(arr, dtype=torch.float32)


# ── main ────────────────────────────────────────────────────────────────────

def run(args, feat=None) -> None:
    set_seed(args.seed)
    device = get_device()
    if feat is None:
        feat = featurize_cached(args.top, args.traj, args.stride, args.selection, native=args.native, cache_dir=getattr(args, "cache_dir", "outputs/cache"), read_cache=getattr(args, "read", False))
    out = Path(args.output_dir) / SUBSECTION
    attach_committor(feat)
    fi = feat.colorings
    X_dist = feat.distances.astype(np.float32)
    n_feat = X_dist.shape[1]
    log_effective_tau(args, "script 04 time-lagged DL")
    # interpretability is handled manually below via autograd Jacobians on
    # each trained encoder — no automatic Pearson fallback here.
    set_run_context(feat=feat, args=args)

    from cv_playground import interpret
    encoders: dict = {}   # populated by each closure: name -> (encoder_fn, device)

    def _torch_pdp_fn(enc, dev):
        """Wrap a torch encoder into a numpy transform_fn for per_residue_pdp.

        - Disables grad.
        - Slices output to the first two CVs (matches what run_method keeps).
        - Handles encoders that return (recon, z) or similar tuples.
        """
        import torch as _torch

        def _fn(X_np):
            was_training = enc.training
            enc.eval()
            try:
                xt = _torch.as_tensor(X_np, dtype=_torch.float32, device=dev)
                with _torch.no_grad():
                    z = enc(xt)
                if isinstance(z, (tuple, list)):
                    z = z[-1] if z[-1].ndim == 2 else z[0]
                z = z.detach().cpu().numpy()
            finally:
                if was_training:
                    enc.train()
            return z[:, :2] if z.ndim == 2 and z.shape[1] > 2 else z

        return _fn

    def _interpret(name: str) -> None:
        if name not in encoders:
            return
        enc_fn, dev = encoders[name]
        tag = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        interpret.jacobian_contact_map(
            enc_fn, X_dist, feat.n_atoms,
            save_path=out / "interpret" / f"{tag}.png",
            title=f"{name} — mean |∂CV/∂pairwise-distance|",
            device=dev,
        )

    # ── 1. Time-lagged Autoencoder (TAE) ────────────────────────────────────
    def tae():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_t, X_tau = _time_lagged_dataset(X_dist, args.lag)

        class TAE(nn.Module):
            def __init__(self, d_in, d_latent=2):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(d_in, 128), nn.ReLU(),
                    nn.Linear(128, 64),  nn.ReLU(),
                    nn.Linear(64, d_latent),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(d_latent, 64), nn.ReLU(),
                    nn.Linear(64, 128),      nn.ReLU(),
                    nn.Linear(128, d_in),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        model = TAE(n_feat).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(_to_tensor(X_t), _to_tensor(X_tau))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta, name="TAE")
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for xt, xtau in dl:
                xt, xtau = xt.to(device), xtau.to(device)
                recon_t, z_t   = model(xt)
                recon_tau, z_tau = model(xtau)
                loss = (
                    nn.functional.mse_loss(recon_t, xt)
                    + nn.functional.mse_loss(recon_tau, xtau)
                    + 0.5 * nn.functional.mse_loss(z_t, z_tau)
                )
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item() * xt.size(0)
            mean_loss = epoch_loss / len(X_t)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                logger.info("  TAE epoch %d/%d  loss=%.4f",
                            epoch + 1, args.epochs, mean_loss)
            if stopper.step(mean_loss, epoch):
                break

        model.eval()
        with torch.no_grad():
            _, z = model(_to_tensor(X_dist).to(device))
        encoders["TAE"] = (lambda x, _m=model: _m.encoder(x), device)
        return z.cpu().numpy(), _torch_pdp_fn(model.encoder, device)

    run_method("TAE", tae, fi, out)
    _interpret("TAE")

    # ── 2. VAMPnet (deeptime) — sweep n_states ∈ {2, 6} ─────────────────────
    def _make_vampnet(d_out: int, reg_name: str):
        def _fn():
            import torch
            import torch.nn as nn
            from deeptime.decomposition.deep import VAMPNet
            from torch.utils.data import DataLoader, TensorDataset

            # Lobe architecture from Mardt, Pasquali, Wu & Noé 2018 Trp-cage
            # benchmark: five 100-unit ELU layers into an n-state softmax head.
            class Lobe(nn.Module):
                def __init__(self, d_in, d_out_):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d_in, 100), nn.ELU(),
                        nn.Linear(100, 100),  nn.ELU(),
                        nn.Linear(100, 100),  nn.ELU(),
                        nn.Linear(100, 100),  nn.ELU(),
                        nn.Linear(100, 100),  nn.ELU(),
                        nn.Linear(100, d_out_),
                    )
                def forward(self, x):
                    return self.net(x)

            lobe = Lobe(n_feat, d_out).to(device)
            vnet = VAMPNet(lobe=lobe, lobe_timelagged=None,
                           learning_rate=1e-3, device=device)

            X_t, X_tau = _time_lagged_dataset(X_dist, args.lag)
            ds = TensorDataset(_to_tensor(X_t), _to_tensor(X_tau))
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

            stopper = EarlyStopping(args.patience, args.min_delta, name=reg_name)
            for epoch in range(args.epochs):
                for xt, xtau in dl:
                    vnet.partial_fit((xt.to(device), xtau.to(device)))
                # deeptime exposes the per-step VAMP score array; take the
                # most-recent value and negate so "smaller is better".
                try:
                    score = float(np.asarray(vnet.train_scores)[-1, 1])
                    if stopper.step(-score, epoch):
                        break
                except Exception:
                    pass

            mdl = vnet.fetch_model()
            cvs = mdl.transform(_to_tensor(X_dist).to(device))
            if isinstance(cvs, torch.Tensor):
                cvs = cvs.detach().cpu().numpy()
            encoders[reg_name] = (lambda x, _l=lobe: _l(x), device)
            cvs = cvs[:, :2] if cvs.shape[1] > 2 else cvs
            return cvs, _torch_pdp_fn(lobe, device)
        return _fn

    # n ∈ {2, 6}: Mardt et al. 2018 reports a 6-state VAMPnet for Trp-cage
    # (five intermediates plus folded); n=2 is the folded/unfolded baseline.
    for _n_states in (2, 6):
        _name = f"VAMPnet (n={_n_states})"
        run_method(_name, _make_vampnet(_n_states, _name), fi, out)
        _interpret(_name)

    # ── 3. Deep-tICA (mlcolvar) ─────────────────────────────────────────────
    def deep_tica():
        import torch
        import lightning
        from mlcolvar.cvs import DeepTICA
        from mlcolvar.data import DictDataset, DictModule

        X_t, X_tau = _time_lagged_dataset(X_dist, args.lag)
        dataset = DictDataset({
            "data":      _to_tensor(X_t),
            "data_lag":  _to_tensor(X_tau),
            "weights":     _to_tensor(np.ones(len(X_t),   dtype=np.float32)),
            "weights_lag": _to_tensor(np.ones(len(X_tau), dtype=np.float32)),
        })
        datamodule = DictModule(dataset, batch_size=args.batch_size, shuffle=True)

        model = DeepTICA(
            layers=[n_feat, 128, 64, 2],
            n_cvs=2,
        )
        from lightning.pytorch.callbacks import EarlyStopping as LightningES
        es = LightningES(
            monitor="valid_loss", mode="min",
            patience=args.patience, min_delta=args.min_delta,
        )
        trainer = lightning.Trainer(
            max_epochs=args.epochs, accelerator="auto",
            enable_progress_bar=False, enable_checkpointing=False, logger=False,
            callbacks=[es],
        )
        trainer.fit(model, datamodule)

        model.eval()
        with torch.no_grad():
            cvs = model(_to_tensor(X_dist).to(model.device))
        encoders["Deep-tICA"] = (lambda x, _m=model: _m(x), model.device)
        return cvs.cpu().numpy(), _torch_pdp_fn(model, model.device)

    run_method("Deep-tICA", deep_tica, fi, out)
    _interpret("Deep-tICA")

    # ── 4. Deep-LDA (mlcolvar) ──────────────────────────────────────────────
    def deep_lda():
        import torch
        import lightning
        from mlcolvar.cvs import DeepLDA
        from mlcolvar.data import DictDataset, DictModule

        labels = get_or_make_labels(args, feat)
        n_classes = int(len(np.unique(labels)))
        n_cvs = min(2, n_classes - 1)

        dataset = DictDataset({
            "data":    _to_tensor(X_dist),
            "labels":  torch.as_tensor(labels, dtype=torch.float32),
            "weights": _to_tensor(np.ones(len(X_dist), dtype=np.float32)),
        })
        datamodule = DictModule(dataset, batch_size=args.batch_size, shuffle=True)

        model = DeepLDA(
            layers=[n_feat, 128, 64, n_cvs],
            n_states=n_classes,
        )
        from lightning.pytorch.callbacks import EarlyStopping as LightningES
        es = LightningES(
            monitor="valid_loss", mode="min",
            patience=args.patience, min_delta=args.min_delta,
            check_finite=True,   # stop cleanly on NaN/Inf instead of crashing
        )
        trainer = lightning.Trainer(
            max_epochs=args.epochs, accelerator="auto",
            enable_progress_bar=False, enable_checkpointing=False, logger=False,
            callbacks=[es],
            gradient_clip_val=1.0,   # LDA scatter matrices are sensitive to exploding grads
        )
        try:
            trainer.fit(model, datamodule)
        except ValueError as e:
            # mlcolvar's Cholesky-LDA raises when a batch yields a degenerate
            # (NaN) scatter matrix; fall back to whatever weights we have.
            logger.warning("Deep-LDA training aborted early due to LDA instability: %s", e)

        model.eval()
        with torch.no_grad():
            cvs = model(_to_tensor(X_dist).to(model.device))
        encoders["Deep-LDA"] = (lambda x, _m=model: _m(x), model.device)
        cvs = cvs.cpu().numpy()
        if cvs.shape[1] < 2:
            cvs = np.column_stack([cvs, np.zeros(len(cvs))])

        def _tf(X_np, _m=model):
            import torch as _t
            _m.eval()
            with _t.no_grad():
                z = _m(_t.as_tensor(X_np, dtype=_t.float32, device=_m.device))
            z = z.detach().cpu().numpy()
            if z.ndim == 2 and z.shape[1] < 2:
                z = np.column_stack([z, np.zeros(len(z))])
            return z[:, :2]

        return cvs, _tf

    run_method("Deep-LDA", deep_lda, fi, out)
    _interpret("Deep-LDA")

    # ── 5. VAE on distance features (bonus) ─────────────────────────────────
    def vae():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class VAE(nn.Module):
            def __init__(self, d_in, d_latent=2):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(d_in, 128), nn.ReLU(),
                    nn.Linear(128, 64),  nn.ReLU(),
                )
                self.fc_mu     = nn.Linear(64, d_latent)
                self.fc_logvar = nn.Linear(64, d_latent)
                self.dec = nn.Sequential(
                    nn.Linear(d_latent, 64), nn.ReLU(),
                    nn.Linear(64, 128),      nn.ReLU(),
                    nn.Linear(128, d_in),
                )

            def encode(self, x):
                h = self.enc(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.dec(z), mu, logvar

        model = VAE(n_feat).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(_to_tensor(X_dist))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta, name="VAE")
        for epoch in range(args.epochs):
            epoch_loss = 0.0; n_seen = 0
            for (x,) in dl:
                x = x.to(device)
                recon, mu, logvar = model(x)
                recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item(); n_seen += x.size(0)
            if stopper.step(epoch_loss / max(1, n_seen), epoch):
                break

        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(_to_tensor(X_dist).to(device))
        encoders["VAE"] = (lambda x, _m=model: _m.encode(x)[0], device)

        def _tf(X_np, _m=model, _dev=device):
            import torch as _t
            _m.eval()
            with _t.no_grad():
                m, _ = _m.encode(_t.as_tensor(X_np, dtype=_t.float32, device=_dev))
            m = m.detach().cpu().numpy()
            return m[:, :2] if m.shape[1] > 2 else m

        return mu.cpu().numpy(), _tf

    run_method("VAE", vae, fi, out)
    _interpret("VAE")

    # ── 6. RAVE-style time-lagged VAE ───────────────────────────────────────
    # Encoder q(z|x_t), predictive decoder p(x_{t+τ}|z_t) + KL to N(0, I).
    def rave():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class RAVE(nn.Module):
            def __init__(self, d_in, d_latent=2):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(d_in, 128), nn.ReLU(),
                    nn.Linear(128, 64),  nn.ReLU(),
                )
                self.fc_mu     = nn.Linear(64, d_latent)
                self.fc_logvar = nn.Linear(64, d_latent)
                self.pred = nn.Sequential(
                    nn.Linear(d_latent, 64), nn.ReLU(),
                    nn.Linear(64, 128),      nn.ReLU(),
                    nn.Linear(128, d_in),
                )

            def encode(self, x):
                h = self.enc(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def forward(self, xt):
                mu, logvar = self.encode(xt)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                return self.pred(z), mu, logvar

        X_t, X_tau = _time_lagged_dataset(X_dist, args.lag)
        model = RAVE(n_feat).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(_to_tensor(X_t), _to_tensor(X_tau))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

        model.train()
        stopper = EarlyStopping(args.patience, args.min_delta, name="RAVE")
        for epoch in range(args.epochs):
            epoch_loss = 0.0; n_seen = 0
            for xt, xtau in dl:
                xt, xtau = xt.to(device), xtau.to(device)
                pred, mu, logvar = model(xt)
                pred_loss = nn.functional.mse_loss(pred, xtau, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = pred_loss + kl
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item(); n_seen += xt.size(0)
            if stopper.step(epoch_loss / max(1, n_seen), epoch):
                break

        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(_to_tensor(X_dist).to(device))
        encoders["RAVE (time-lagged VAE)"] = (lambda x, _m=model: _m.encode(x)[0], device)

        def _tf(X_np, _m=model, _dev=device):
            import torch as _t
            _m.eval()
            with _t.no_grad():
                m, _ = _m.encode(_t.as_tensor(X_np, dtype=_t.float32, device=_dev))
            m = m.detach().cpu().numpy()
            return m[:, :2] if m.shape[1] > 2 else m

        return mu.cpu().numpy(), _tf

    run_method("RAVE (time-lagged VAE)", rave, fi, out)
    _interpret("RAVE (time-lagged VAE)")


def main() -> None:
    setup_logging()
    args = base_argparser("04 – Deep-learning CVs").parse_args()
    run(args)


if __name__ == "__main__":
    main()

