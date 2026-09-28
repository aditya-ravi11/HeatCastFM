"""Verification scores for quantile forecasts and event probabilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .probability import LEVELS


def quantile_score(q: np.ndarray, y: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """Pinball loss per level, shape (..., K)."""
    diff = y[..., None] - q
    return np.maximum(levels * diff, (levels - 1) * diff)


def crps_from_quantiles(q: np.ndarray, y: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """CRPS = 2 * integral of pinball loss over levels, trapezoid rule on the level grid.

    The tails outside [0.01, 0.99] are left out, which slightly understates CRPS for every
    model in the same way.
    """
    qs = quantile_score(np.sort(q, -1), y, levels)
    return 2.0 * np.trapezoid(qs, levels, axis=-1)


def interval_stats(q: np.ndarray, y: np.ndarray, lo: float = 0.1, hi: float = 0.9,
                   levels: np.ndarray = LEVELS):
    i_lo = int(np.argmin(abs(levels - lo)))
    i_hi = int(np.argmin(abs(levels - hi)))
    inside = (y >= q[..., i_lo]) & (y <= q[..., i_hi])
    width = q[..., i_hi] - q[..., i_lo]
    return inside, width


def brier(p: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((p - o) ** 2))


def brier_skill(p: np.ndarray, p_ref: np.ndarray, o: np.ndarray) -> float:
    ref = brier(p_ref, o)
    return float(1 - brier(p, o) / ref) if ref > 0 else np.nan


def auc(p: np.ndarray, o: np.ndarray) -> float:
    o = o.astype(int)
    if o.min() == o.max():
        return np.nan
    return float(roc_auc_score(o, p))


def contingency(warn: np.ndarray, o: np.ndarray) -> dict:
    warn, o = warn.astype(bool), o.astype(bool)
    hits = int((warn & o).sum())
    misses = int((~warn & o).sum())
    fa = int((warn & ~o).sum())
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + misses + fa) if hits + misses + fa else np.nan
    return {"hits": hits, "misses": misses, "false_alarms": fa, "POD": pod, "FAR": far, "CSI": csi}


def reliability(p: np.ndarray, o: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "p_mean": float(p[m].mean()), "o_freq": float(o[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)


def block_bootstrap_diff(a: np.ndarray, b: np.ndarray, blocks: np.ndarray,
                         n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% CI from resampling whole blocks (e.g. region-year)."""
    rng = np.random.default_rng(seed)
    d = a - b
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        boots[i] = sums[pick].sum() / counts[pick].sum()
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
