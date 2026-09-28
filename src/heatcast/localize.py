"""Objective 3: check grid forecasts against point observations and correct them.

The station proxy (Open-Meteo, or real AWS data through aws_ingest) gives Tmax at the
exact city location. A per region, per month linear map fitted on 1990-2014
observations, point = a + b * grid, turns grid forecasts into point forecasts.
"""
import numpy as np
import pandas as pd

from . import climatology as clim
from . import metrics as M
from .probability import LEVELS, exceed_prob

FIT_YEARS = (1990, 2014)


def point_panel(cfg) -> pd.DataFrame:
    proc = cfg["paths"]["processed"]
    sp = pd.read_parquet(proc / "station_proxy.parquet")
    normals = clim.compute_normals(sp, *FIT_YEARS)
    sp = clim.attach_normals(sp, normals)
    types = {r["name"]: r["type"] for r in cfg["regions"]}
    return clim.add_labels(sp, types, cfg["criteria"])


def fit_maps(grid: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    m = grid[["region", "date", "tmax"]].merge(point[["region", "date", "tmax"]], on=["region", "date"],
                                               suffixes=("_grid", "_point")).dropna()
    m = m[m["date"].dt.year.between(*FIT_YEARS) & m["date"].dt.month.isin([3, 4, 5, 6, 7])]
    rows = []
    for (region, month), g in m.groupby([m["region"], m["date"].dt.month]):
        b, a = np.polyfit(g["tmax_grid"], g["tmax_point"], 1)
        res = g["tmax_point"] - (a + b * g["tmax_grid"])
        rows.append({"region": region, "month": month, "a": a, "b": b, "sigma": res.std(),
                     "r": np.corrcoef(g["tmax_grid"], g["tmax_point"])[0, 1],
                     "bias_point_minus_grid": (g["tmax_point"] - g["tmax_grid"]).mean(), "n": len(g)})
    return pd.DataFrame(rows)


def correct_quantiles(q: np.ndarray, a: float, b: float, sigma: float) -> np.ndarray:
    """Linear map of the median, spread widened by the regression residual sigma."""
    i50 = list(LEVELS).index(0.5)
    med = q[..., i50:i50 + 1]
    s = np.maximum((q[..., -3:-2] - q[..., 2:3]) / 2.563, 1e-3)  # sd implied by the 0.1/0.9 quantiles
    factor = np.sqrt(1 + sigma ** 2 / (b * s) ** 2)
    return a + b * med + (q - med) * b * factor


def evaluate(pred_path, store, grid_panel, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = point_panel(cfg)
    maps = fit_maps(grid_panel[grid_panel["group"] == "focus"], point)
    z = np.load(pred_path)
    q, s_idx, t_pos = z["q"], z["s_idx"], z["t_pos"]
    H = q.shape[1]
    names = np.array(store.names)[s_idx]
    keep = np.isin(names, [r["name"] for r in cfg["regions"]])
    q, s_idx, t_pos, names = q[keep], s_idx[keep], t_pos[keep], names[keep]
    tgt_dates = pd.DatetimeIndex(store.dates.values[t_pos]).values[:, None] + \
        np.arange(1, H + 1).astype("timedelta64[D]")[None]

    pt = point.set_index(["region", "date"])
    rows = []
    for i in range(len(s_idx)):
        for h in range(H):
            d = pd.Timestamp(tgt_dates[i, h])
            key = (names[i], d)
            if key not in pt.index:
                continue
            rec = pt.loc[key]
            mp = maps[(maps["region"] == names[i]) & (maps["month"] == d.month)]
            if mp.empty or np.isnan(rec["tmax"]):
                continue
            a, b, sg = mp[["a", "b", "sigma"]].values[0]
            qc = correct_quantiles(q[i, h][None], a, b, sg)[0]
            rows.append({
                "region": names[i], "target": d, "lead": h + 1, "obs_point": rec["tmax"],
                "raw_median": q[i, h, 10], "cor_median": qc[10],
                "raw_crps": float(M.crps_from_quantiles(q[i, h], np.array(rec["tmax"]))),
                "cor_crps": float(M.crps_from_quantiles(qc, np.array(rec["tmax"]))),
                "thr_hw_point": rec["thr_hw"], "o_hw_point": int(rec["cls"] >= 1),
                "raw_p_hw": float(exceed_prob(rec["thr_hw"], q[i, h])),
                "cor_p_hw": float(exceed_prob(rec["thr_hw"], qc)),
            })
    return pd.DataFrame(rows), maps
