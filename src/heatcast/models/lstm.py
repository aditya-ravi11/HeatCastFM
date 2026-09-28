"""LSTM trained with the pinball loss: a supervised deep learning baseline.

Inputs per day over a 60 day window: Tmax anomaly, Tmin anomaly, sin and cos of day of year.
Output: H x K anomaly quantiles, added to the daily normal of each target day.
"""
import numpy as np
import torch
from torch import nn

from ..probability import LEVELS

WINDOW = 60


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class QuantileLSTM(nn.Module):
    def __init__(self, n_feat: int, horizon: int, n_q: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon * n_q)
        self.horizon, self.n_q = horizon, n_q

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.head(h[:, -1]).view(-1, self.horizon, self.n_q)
        return torch.sort(out, dim=-1).values  # keep quantiles from crossing


def pinball(pred, y, levels):
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(levels * diff, (levels - 1) * diff).mean()


class LSTMForecaster:
    name = "Quantile-LSTM"

    def __init__(self, epochs: int = 8, batch: int = 1024, lr: float = 2e-3, seed: int = 0):
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed

    def _features(self, store):
        ta = store.tmax_in - store.normal
        tn = store.tmin_in - store.normal_tmin
        n_s, n_t = ta.shape
        sin = np.broadcast_to(store.doy_sin, (n_s, n_t))
        cos = np.broadcast_to(store.doy_cos, (n_s, n_t))
        self.scale = np.array([ta.std(), tn.std(), 1.0, 1.0], dtype="float32")
        return np.stack([ta, tn, sin, cos], -1).astype("float32") / self.scale, ta.astype("float32")

    def _windows(self, feats, anom, s_idx, t_pos, H):
        offs = np.arange(-WINDOW + 1, 1)
        X = feats[s_idx[:, None], t_pos[:, None] + offs]
        Y = anom[s_idx[:, None], t_pos[:, None] + np.arange(1, H + 1)]
        return X, Y

    def fit(self, store, cfg):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        H = cfg["forecast"]["horizon"]
        self.H = H
        self.feats, self.anom = self._features(store)
        years, months = store.dates.year.values, store.dates.month.values
        n_s, n_t = self.anom.shape
        valid_t = (np.arange(n_t) >= WINDOW) & (np.arange(n_t) < n_t - H)
        tr_t = np.where(valid_t & (years <= cfg["years"]["train_end"]) & np.isin(months, [2, 3, 4, 5, 6, 7]))[0]
        va_t = np.where(valid_t & np.isin(years, cfg["years"]["val"]) & np.isin(months, [3, 4, 5, 6]))[0]
        tr = np.array(np.meshgrid(np.arange(n_s), tr_t)).reshape(2, -1).T
        va = np.array(np.meshgrid(np.arange(n_s), va_t)).reshape(2, -1).T
        va = va[rng.choice(len(va), min(20000, len(va)), replace=False)]

        dev = _device()
        self.model = QuantileLSTM(4, H, len(LEVELS)).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lv = torch.tensor(LEVELS, dtype=torch.float32, device=dev)
        Xv, Yv = self._windows(self.feats, self.anom, va[:, 0], va[:, 1], H)
        Xv, Yv = torch.tensor(Xv, device=dev), torch.tensor(Yv, device=dev)
        best, best_state = np.inf, None
        self.history = []
        for ep in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(tr))[:300_000]  # a fresh subsample each epoch
            for i in range(0, len(perm), self.batch):
                b = tr[perm[i:i + self.batch]]
                X, Y = self._windows(self.feats, self.anom, b[:, 0], b[:, 1], H)
                loss = pinball(self.model(torch.tensor(X, device=dev)), torch.tensor(Y, device=dev), lv)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            self.model.eval()
            with torch.no_grad():
                vl = float(np.mean([pinball(self.model(Xv[j:j + 4096]), Yv[j:j + 4096], lv).item()
                                    for j in range(0, len(Xv), 4096)]))
            self.history.append({"epoch": ep + 1, "train_last_batch": float(loss.item()), "val_pinball": vl})
            print(f"  LSTM epoch {ep + 1}: val pinball {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return self

    def predict(self, store, s_idx, t_pos, H):
        dev = _device()
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(s_idx), 4096):
                X, _ = self._windows(self.feats, self.anom, s_idx[i:i + 4096], t_pos[i:i + 4096], H)
                out.append(self.model(torch.tensor(X, device=dev)).cpu().numpy())
        q = np.concatenate(out)
        tgt = t_pos[:, None] + np.arange(1, H + 1)[None]
        return q + store.normal[s_idx[:, None], tgt][..., None]
