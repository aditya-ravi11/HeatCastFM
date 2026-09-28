"""Reference forecasts: persistence, climatology and an autoregressive anomaly model.

Every predict() returns quantiles of Tmax on probability.LEVELS with shape (n, H, K).
"""
import numpy as np
from scipy.stats import norm

from ..climatology import doy365
from ..probability import LEVELS


def _train_mask(store, cfg, first=None):
    years = store.dates.year
    first = first or cfg["years"]["first"]
    return (years >= first) & (years <= cfg["years"]["train_end"])


def _season_mask(store, months=(3, 4, 5, 6, 7)):
    return np.isin(store.dates.month, months)


class Persistence:
    """Tomorrow looks like today, with the training error distribution per series, lead and month."""

    name = "Persistence"

    def fit(self, store, cfg):
        H = cfg["forecast"]["horizon"]
        tr = _train_mask(store, cfg) & _season_mask(store)
        x = store.tmax_in
        n_s = x.shape[0]
        self.err_q = np.zeros((n_s, H, 12, len(LEVELS)), dtype="float32")
        months = store.dates.month.values
        for h in range(1, H + 1):
            err = x[:, h:] - x[:, :-h]
            ok = tr[:-h]
            for m in range(3, 8):
                sel = ok & (months[:-h] == m)
                self.err_q[:, h - 1, m - 1] = np.quantile(err[:, sel], LEVELS, axis=1).T
        # months outside Mar-Jul fall back to March
        for m in list(range(0, 2)) + list(range(7, 12)):
            self.err_q[:, :, m] = self.err_q[:, :, 2]
        return self

    def predict(self, store, s_idx, t_pos, H):
        last = store.tmax_in[s_idx, t_pos]
        month = store.dates.month.values[t_pos] - 1
        eq = self.err_q[s_idx, :, month]  # (n, H, K)
        return last[:, None, None] + eq[:, :H]


class Climatology:
    """Daily normal plus the spread of anomalies seen within +/-15 days in 1985-2014."""

    name = "Climatology"

    def fit(self, store, cfg):
        tr = _train_mask(store, cfg, first=cfg["years"]["clim_start"])
        anom = store.tmax_in - store.normal
        doy = doy365(store.dates)
        self.anom_q = np.zeros((anom.shape[0], 366, len(LEVELS)), dtype="float32")
        for d in range(1, 366):
            dist = np.abs(((doy - d + 182) % 365) - 182)
            sel = tr & (dist <= 15)
            self.anom_q[:, d] = np.quantile(anom[:, sel], LEVELS, axis=1).T
        self.doy = doy
        return self

    def predict(self, store, s_idx, t_pos, H):
        tgt = t_pos[:, None] + np.arange(1, H + 1)[None]
        normal = store.normal[s_idx[:, None], tgt]
        return normal[..., None] + self.anom_q[s_idx[:, None], self.doy[tgt]]


class ARAnomaly:
    """AR(p) on daily anomalies, p chosen by AIC per series, Gaussian predictive spread."""

    name = "AR-anomaly"

    def __init__(self, max_p: int = 14):
        self.max_p = max_p

    def fit(self, store, cfg):
        tr = _train_mask(store, cfg)
        season = _season_mask(store)
        anom = (store.tmax_in - store.normal).astype("float64")
        self.coef, self.sigma, self.order = [], [], []
        for s in range(anom.shape[0]):
            a = anom[s, tr]
            best = None
            for p in range(1, self.max_p + 1):
                X = np.column_stack([np.ones(len(a) - p)] + [a[p - k - 1:len(a) - k - 1] for k in range(p)])
                y = a[p:]
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ beta
                aic = len(y) * np.log(res.var()) + 2 * (p + 1)
                if best is None or aic < best[0]:
                    best = (aic, p, beta)
            _, p, beta = best
            # residual sigma measured in the hot season, where the forecasts are issued
            full = anom[s]
            X = np.column_stack([np.ones(len(full) - p)] + [full[p - k - 1:len(full) - k - 1] for k in range(p)])
            res = full[p:] - X @ beta
            m = (tr & season)[p:]
            self.coef.append(beta)
            self.sigma.append(res[m].std())
            self.order.append(p)
        return self

    def predict(self, store, s_idx, t_pos, H):
        z = norm.ppf(LEVELS)
        out = np.empty((len(s_idx), H, len(LEVELS)), dtype="float32")
        anom = store.tmax_in - store.normal
        for i, (s, t) in enumerate(zip(s_idx, t_pos)):
            beta, p, sig = self.coef[s], self.order[s], self.sigma[s]
            hist = list(anom[s, t - p + 1:t + 1][::-1])  # most recent first
            phi = beta[1:]
            # psi weights for the forecast error variance
            psi = [1.0]
            for j in range(1, H):
                psi.append(sum(phi[k] * psi[j - k - 1] for k in range(min(p, j))))
            psi = np.array(psi)
            for h in range(H):
                nxt = beta[0] + np.dot(phi, hist[:p])
                hist.insert(0, nxt)
                sd = sig * np.sqrt(np.sum(psi[:h + 1] ** 2))
                out[i, h] = store.normal[s, t + h + 1] + nxt + z * sd
        return out
