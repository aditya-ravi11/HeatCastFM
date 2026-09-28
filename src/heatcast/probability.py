"""Turning forecast quantiles or sample paths into event probabilities."""
import numpy as np

# Common quantile grid used to store every model's forecast (Chronos-2's native grid)
LEVELS = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])


def _tail_scales(q: np.ndarray, levels: np.ndarray):
    """Exponential tail scales matched to the two outermost quantiles on each side."""
    eps = 1e-3
    up = (q[..., -1] - q[..., -2]) / np.log((1 - levels[-2]) / (1 - levels[-1]))
    lo = (q[..., 1] - q[..., 0]) / np.log(levels[1] / levels[0])
    return np.maximum(lo, eps), np.maximum(up, eps)


def cdf_from_quantiles(x, q: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """F(x) from quantiles q[..., k] at levels[k]; linear inside, exponential tails.

    x broadcasts against q[..., 0].
    """
    q = np.sort(q, axis=-1)
    x_b, _ = np.broadcast_arrays(np.asarray(x, dtype=float), q[..., 0])
    q = np.broadcast_to(q, x_b.shape + q.shape[-1:])
    lo_s, up_s = _tail_scales(q, levels)

    # interior: position of x among the quantiles
    idx = (q <= x_b[..., None]).sum(-1)  # 0 .. K
    K = len(levels)
    i0 = np.clip(idx - 1, 0, K - 2)
    q0 = np.take_along_axis(q, i0[..., None], -1)[..., 0]
    q1 = np.take_along_axis(q, (i0 + 1)[..., None], -1)[..., 0]
    l0, l1 = levels[i0], levels[i0 + 1]
    w = np.where(q1 > q0, (x_b - q0) / np.where(q1 > q0, q1 - q0, 1.0), 1.0)
    F = l0 + np.clip(w, 0, 1) * (l1 - l0)

    below = x_b < q[..., 0]
    above = x_b > q[..., -1]
    # exponents are clipped at 0: the clipped side is only used where the tail applies
    F = np.where(below, levels[0] * np.exp(np.minimum((x_b - q[..., 0]) / lo_s, 0)), F)
    F = np.where(above, 1 - (1 - levels[-1]) * np.exp(-np.maximum((x_b - q[..., -1]) / up_s, 0)), F)
    return np.clip(F, 0.0, 1.0)


def exceed_prob(threshold, q: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    return 1.0 - cdf_from_quantiles(threshold, q, levels)


def quantiles_from_cdf_grid(q_src: np.ndarray, levels_src: np.ndarray,
                            levels_dst: np.ndarray = LEVELS, n_grid: int = 400) -> np.ndarray:
    """Re-express quantiles on another level grid via the tail-extended CDF."""
    lo = q_src.min(-1, keepdims=True) - 6.0
    hi = q_src.max(-1, keepdims=True) + 6.0
    t = np.linspace(0, 1, n_grid)
    xs = lo + (hi - lo) * t  # (..., n_grid)
    F = cdf_from_quantiles(np.moveaxis(xs, -1, 0), q_src[None], levels_src)  # (n_grid, ...)
    F = np.moveaxis(F, 0, -1)
    out = np.empty(q_src.shape[:-1] + (len(levels_dst),))
    flatF = F.reshape(-1, n_grid)
    flatX = xs.reshape(-1, n_grid)
    flatO = out.reshape(-1, len(levels_dst))
    for i in range(flatF.shape[0]):
        flatO[i] = np.interp(levels_dst, np.maximum.accumulate(flatF[i]), flatX[i])
    return out


def quantiles_from_samples(samples: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """samples (..., S) to quantiles (..., K)."""
    return np.moveaxis(np.quantile(samples, levels, axis=-1), 0, -1)


def exceed_prob_samples(threshold, samples: np.ndarray) -> np.ndarray:
    return (samples >= np.asarray(threshold)[..., None]).mean(-1)
