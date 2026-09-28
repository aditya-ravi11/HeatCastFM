"""Export cached backtest forecasts for the 7 focus regions to compact JSON for the
static web viewer in docs/ (served by GitHub Pages).

Temperatures are stored as integers in tenths of a degree, probabilities as whole percent.
"""
import json

import numpy as np
import pandas as pd

from heatcast.config import ROOT, load_config
from heatcast.features import SeriesStore, load_panel
from heatcast.probability import LEVELS, exceed_prob

OUT = ROOT / "docs" / "data"
MODELS = ["Chronos-2+cov", "Chronos-2+cov+RH", "Chronos-2", "Chronos-Bolt", "Chronos-T5",
          "Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]
Q_KEEP = [0.1, 0.25, 0.5, 0.75, 0.9]


def slug(name: str) -> str:
    return name.replace("+", "_").replace("-", "").lower()


def tenths(a):
    return [None if not np.isfinite(v) else int(round(v * 10)) for v in np.asarray(a, dtype=float)]


if __name__ == "__main__":
    cfg = load_config()
    store = SeriesStore(load_panel(cfg))
    regions = [r["name"] for r in cfg["regions"]]
    years = cfg["years"]["val"] + cfg["years"]["test"]
    H = cfg["forecast"]["horizon"]
    qi = [list(LEVELS).index(q) for q in Q_KEEP]
    OUT.mkdir(parents=True, exist_ok=True)

    # daily observations and thresholds, 1 Feb to 10 Jul of each year
    keep = np.zeros(len(store.dates), dtype=bool)
    for y in years:
        keep |= (store.dates >= f"{y}-02-01") & (store.dates <= f"{y}-07-10")
    day_idx = np.where(keep)[0]
    days = [d.strftime("%Y-%m-%d") for d in store.dates[day_idx]]
    pos_in_days = {int(t): i for i, t in enumerate(day_idx)}

    series = {}
    for r in regions:
        s = store.index(r)
        series[r] = {
            "type": str(store.rtype[s]),
            "tmax": tenths(store.tmax_obs[s, day_idx]),
            "thr_hw": tenths(store.thr_hw[s, day_idx]),
            "thr_sev": tenths(store.thr_sev[s, day_idx]),
            "thr_hot": tenths(store.thr_p90[s, day_idx]),
        }

    thresholds = json.loads((cfg["paths"]["metrics"] / "warning_thresholds.json").read_text())
    models_out = []
    for m in MODELS:
        files = sorted(cfg["paths"]["predictions"].glob(f"{m}__*.npz"))
        files = [f for f in files if f.stem.split("__")[0] == m]
        if not files:
            continue
        per_region = {r: {"origins": [], "q": [], "p": []} for r in regions}
        for f in files:
            z = np.load(f)
            q, s_idx, t_pos = z["q"], z["s_idx"], z["t_pos"]
            names = np.array(store.names)[s_idx]
            for r in regions:
                sel = np.where(names == r)[0]
                if not len(sel):
                    continue
                s = store.index(r)
                for k in sel:
                    t = int(t_pos[k])
                    tgt = np.arange(t + 1, t + H + 1)
                    qq = q[k]
                    p_hot = exceed_prob(store.thr_p90[s, tgt], qq)
                    p_hw = exceed_prob(store.thr_hw[s, tgt], qq)
                    p_sev = exceed_prob(store.thr_sev[s, tgt], qq)
                    per_region[r]["origins"].append(pos_in_days[t])
                    per_region[r]["q"].append([tenths(qq[:, j]) for j in qi])
                    per_region[r]["p"].append([[int(round(v * 100)) for v in arr] for arr in (p_hot, p_hw, p_sev)])
        for r in regions:
            d = per_region[r]
            if not d["origins"]:
                continue
            order = np.argsort(d["origins"])
            payload = {k: [d[k][i] for i in order] for k in ("origins", "q", "p")}
            (OUT / f"{slug(r)}__{slug(m)}.json").write_text(json.dumps(payload, separators=(",", ":")))
        models_out.append({"name": m, "slug": slug(m),
                           "thresholds": thresholds.get(m, {"t_watch": 0.3, "t_alert": 0.25, "t_severe": 0.5})})
        print("exported", m)

    meta = {
        "days": days,
        "regions": [{"name": r, "slug": slug(r), **{k: v for k, v in series[r].items()}} for r in regions],
        "models": models_out,
        "horizon": H,
        "quantiles": Q_KEEP,
        "years": years,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    total = sum(f.stat().st_size for f in OUT.glob("*.json"))
    print(f"{len(list(OUT.glob('*.json')))} files, {total / 1e6:.1f} MB")
