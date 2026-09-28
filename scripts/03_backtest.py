"""Run every model over the validation and test hot seasons and cache predictions.

usage: python scripts/03_backtest.py [model ...] [--force]
"""
import json
import sys

from heatcast.backtest import run
from heatcast.config import load_config
from heatcast.features import SeriesStore, load_panel
from heatcast.models.baselines import ARAnomaly, Climatology, Persistence

ALL = ["Persistence", "Climatology", "AR-anomaly", "Quantile-LSTM", "Chronos-Bolt",
       "Chronos-2", "Chronos-2+cov", "Chronos-2+cov+RH", "Chronos-T5"]
FOCUS_ONLY = {"Chronos-2+cov+RH", "Chronos-T5", "Chronos-2+cov-FT"}


def build(name, cfg):
    m = cfg["models"]
    L = cfg["forecast"]["context"]
    if name == "Persistence":
        return Persistence()
    if name == "Climatology":
        return Climatology()
    if name == "AR-anomaly":
        return ARAnomaly()
    if name == "Quantile-LSTM":
        from heatcast.models.lstm import LSTMForecaster
        return LSTMForecaster()
    from heatcast.models import chronos_models as C
    if name == "Chronos-Bolt":
        return C.ChronosBolt(m["bolt"], context=L)
    if name == "Chronos-T5":
        return C.ChronosT5(m["t5"], context=L, num_samples=cfg["forecast"]["t5_samples"])
    cov = {"Chronos-2": None, "Chronos-2+cov": "basic", "Chronos-2+cov+RH": "rh"}[name]
    return C.Chronos2(m["chronos2"], covariates=cov, context=L)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    names = args or ALL
    cfg = load_config()
    store = SeriesStore(load_panel(cfg))
    val, test = cfg["years"]["val"], cfg["years"]["test"]
    for name in names:
        group = "focus" if name in FOCUS_ONLY else None
        outs = [cfg["paths"]["predictions"] / f"{name}__{y[0]}_{y[-1]}__{group or 'all'}.npz" for y in (val, test)]
        if all(o.exists() for o in outs) and not force:
            print(f"skip {name} (cached)")
            continue
        print(f"== {name}", flush=True)
        model = build(name, cfg).fit(store, cfg)
        if hasattr(model, "history"):
            (cfg["paths"]["metrics"] / "lstm_history.json").write_text(json.dumps(model.history, indent=1))
        if hasattr(model, "order"):
            (cfg["paths"]["metrics"] / "ar_orders.json").write_text(
                json.dumps(dict(zip(store.names, map(int, model.order))), indent=1))
        for years, o in zip((val, test), outs):
            if o.exists() and not force:
                print(f"  skip {o.name} (cached)")
                continue
            run(model, store, cfg, years, group=group, tag=name)
        del model
