"""Rolling-origin backtest: one forecast per series per day of the hot season."""
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .features import season_origins
from .probability import LEVELS, exceed_prob, exceed_prob_samples


def origin_grid(store, cfg, years, group: str | None = None):
    origins = season_origins(store.dates, years, cfg)
    t_pos = np.array([store.pos[d] for d in origins])
    series = np.arange(len(store.names))
    if group:
        series = series[store.group == group]
    S, T = np.meshgrid(series, t_pos, indexing="ij")
    return S.ravel(), T.ravel()


CHUNK = 2048


def _free_gpu():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run(model, store, cfg, years, group=None, tag=None):
    """Forecast in chunks; each chunk is saved so an interrupted run resumes where it stopped."""
    H = cfg["forecast"]["horizon"]
    s_idx, t_pos = origin_grid(store, cfg, years, group)
    name = tag or model.name
    part_dir = cfg["paths"]["predictions"] / "partial" / f"{name}__{years[0]}_{years[-1]}__{group or 'all'}"
    part_dir.mkdir(parents=True, exist_ok=True)
    qs, smps, secs = [], [], 0.0
    n_chunks = (len(s_idx) + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        f = part_dir / f"{c:04d}.npz"
        if f.exists():
            z = np.load(f)
        else:
            sl = slice(c * CHUNK, (c + 1) * CHUNK)
            t0 = time.time()
            q = model.predict(store, s_idx[sl], t_pos[sl], H).astype("float32")
            dt = time.time() - t0
            smp = getattr(model, "last_samples", None)
            np.savez(f, q=q, seconds=dt, **({"samples": smp} if smp is not None else {}))
            _free_gpu()
            print(f"    {name} chunk {c + 1}/{n_chunks}: {len(q)} in {dt:.1f}s", flush=True)
            z = np.load(f)
        qs.append(z["q"])
        secs += float(z["seconds"])
        if "samples" in z:
            smps.append(z["samples"])
    q = np.concatenate(qs)
    extra = {}
    if smps:
        smp = np.concatenate(smps).astype("float32")
        tgt = t_pos[:, None] + np.arange(1, H + 1)[None]
        extra["p_hw_samples"] = exceed_prob_samples(store.thr_hw[s_idx[:, None], tgt], smp).astype("float32")
        extra["p_sev_samples"] = exceed_prob_samples(store.thr_sev[s_idx[:, None], tgt], smp).astype("float32")
        extra["p_p90_samples"] = exceed_prob_samples(store.thr_p90[s_idx[:, None], tgt], smp).astype("float32")
    path = cfg["paths"]["predictions"] / f"{name}__{years[0]}_{years[-1]}__{group or 'all'}.npz"
    np.savez_compressed(path, q=q, s_idx=s_idx, t_pos=t_pos, seconds=secs, **extra)
    print(f"  {name}: {len(s_idx)} forecasts in {secs:.1f}s -> {path.name}", flush=True)
    return path


def to_frame(path, store, model_name: str) -> pd.DataFrame:
    """Long table, one row per (series, origin, lead), with scores and event probabilities."""
    z = np.load(path)
    q, s_idx, t_pos = z["q"], z["s_idx"], z["t_pos"]
    n, H, _ = q.shape
    tgt = t_pos[:, None] + np.arange(1, H + 1)[None]
    obs = store.tmax_obs[s_idx[:, None], tgt]
    thr_hw = store.thr_hw[s_idx[:, None], tgt]
    thr_sev = store.thr_sev[s_idx[:, None], tgt]
    thr_p90 = store.thr_p90[s_idx[:, None], tgt]
    y = np.where(np.isnan(obs), 0.0, obs)
    crps = M.crps_from_quantiles(q, y)
    inside, width = M.interval_stats(q, y)
    df = pd.DataFrame({
        "model": model_name,
        "series": np.repeat(np.array(store.names)[s_idx], H),
        "group": np.repeat(store.group[s_idx], H),
        "rtype": np.repeat(store.rtype[s_idx], H),
        "origin": np.repeat(store.dates[t_pos], H),
        "lead": np.tile(np.arange(1, H + 1), n),
        "obs": obs.ravel(),
        "median": q[..., list(LEVELS).index(0.5)].ravel(),
        "q10": q[..., list(LEVELS).index(0.1)].ravel(),
        "q90": q[..., list(LEVELS).index(0.9)].ravel(),
        "crps": crps.ravel(),
        "in80": inside.ravel(),
        "width80": width.ravel(),
        "thr_hw": thr_hw.ravel(),
        "thr_sev": thr_sev.ravel(),
        "p_hw": exceed_prob(thr_hw, q).ravel(),
        "p_sev": exceed_prob(thr_sev, q).ravel(),
        "thr_p90": thr_p90.ravel(),
        "p_p90": exceed_prob(thr_p90, q).ravel(),
    })
    if "p_hw_samples" in z:
        df["p_hw_samples"] = z["p_hw_samples"].ravel()
        df["p_sev_samples"] = z["p_sev_samples"].ravel()
        df["p_p90_samples"] = z["p_p90_samples"].ravel()
    df["target"] = df["origin"] + pd.to_timedelta(df["lead"], unit="D")
    df["o_hw"] = (df["obs"] >= df["thr_hw"]).astype(int)
    df["o_sev"] = (df["obs"] >= df["thr_sev"]).astype(int)
    df["o_p90"] = (df["obs"] >= df["thr_p90"]).astype(int)
    return df[df["obs"].notna()].reset_index(drop=True)
