"""Stretch goal: LoRA fine-tune of Chronos-2 (with covariates) on 1951-2014, then backtest it.

usage: python scripts/03b_finetune.py [num_steps]
"""
import json
import sys
import time

import numpy as np

from heatcast.backtest import run
from heatcast.config import load_config
from heatcast.features import SeriesStore, load_panel
from heatcast.models.chronos_models import Chronos2

KNOWN = ["doy_sin", "doy_cos", "normal"]

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfg = load_config()
    store = SeriesStore(load_panel(cfg))
    H, L = cfg["forecast"]["horizon"], cfg["forecast"]["context"]
    train_end = np.where(store.dates.year <= cfg["years"]["train_end"])[0][-1] + 1
    n_s = len(store.names)
    sin = np.broadcast_to(store.doy_sin, (n_s, len(store.dates)))
    cos = np.broadcast_to(store.doy_cos, (n_s, len(store.dates)))

    from chronos.chronos2.preprocess import from_list_of_dicts

    items = [{
        "target": store.tmax_in[s, :train_end],
        "past_covariates": {"tmin": store.tmin_in[s, :train_end], "doy_sin": sin[s, :train_end],
                            "doy_cos": cos[s, :train_end], "normal": store.normal[s, :train_end]},
    } for s in range(n_s)]
    prepared = from_list_of_dicts(items, prediction_length=H, known_covariates_names=KNOWN)

    base = Chronos2(cfg["models"]["chronos2"], covariates="basic", context=L)
    t0 = time.time()
    tuned_pipe = base.pipe.fit(
        prepared, prediction_length=H, finetune_mode="lora", learning_rate=1e-5,
        num_steps=steps, batch_size=32, context_length=L,
        output_dir=str(cfg["paths"]["predictions"].parent / "chronos2_lora"),
        logging_steps=25, report_to=[], use_cpu=True, dataloader_num_workers=0,
    )
    minutes = (time.time() - t0) / 60
    print(f"fine-tuned {steps} steps in {minutes:.1f} min", flush=True)
    (cfg["paths"]["metrics"] / "finetune.json").write_text(json.dumps(
        {"steps": steps, "batch_size": 32, "lr": 1e-5, "mode": "lora", "minutes": minutes}, indent=1))

    tuned = Chronos2(None, covariates="basic", context=L, pipeline=tuned_pipe, label="Chronos-2+cov-FT")
    # scored on the 7 focus regions, like Chronos-T5 and the +RH variant, to keep laptop run time down
    for years in (cfg["years"]["val"], cfg["years"]["test"]):
        run(tuned, store, cfg, years, group="focus", tag="Chronos-2+cov-FT")
