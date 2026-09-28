"""IMD style colour warnings from forecast probabilities.

Green   no warning
Yellow  heat watch       P(hot day, Tmax above local 90th percentile) >= t_watch
Orange  heatwave alert   P(IMD heatwave) >= t_alert
Red     severe heatwave  P(IMD severe heatwave) >= t_severe

t_watch and t_alert are chosen for the best CSI on the validation years. Severe events are
too rare on the 1 degree grid to tune on, so t_severe is fixed at 0.5.
"""
import numpy as np
import pandas as pd

from . import metrics as M

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
COLOURS = ["Green", "Yellow", "Orange", "Red"]


def best_csi(p, o):
    scores = [(t, M.contingency(p >= t, o)["CSI"]) for t in GRID]
    scores = [(t, c) for t, c in scores if not np.isnan(c)]
    return max(scores, key=lambda x: x[1])[0] if scores else 0.5


def tune(val: pd.DataFrame, leads=(1, 2, 3)) -> dict:
    v = val[val["lead"].isin(leads)]
    return {
        "t_watch": float(best_csi(v["p_p90"].values, v["o_p90"].values)),
        "t_alert": float(best_csi(v["p_hw"].values, v["o_hw"].values)),
        "t_severe": 0.5,
    }


def level(p_p90, p_hw, p_sev, th: dict) -> np.ndarray:
    lvl = np.zeros(len(p_hw), dtype=int)
    lvl[p_p90 >= th["t_watch"]] = 1
    lvl[p_hw >= th["t_alert"]] = 2
    lvl[p_sev >= th["t_severe"]] = 3
    return lvl


def evaluate(df: pd.DataFrame, th: dict) -> dict:
    lvl = level(df["p_p90"].values, df["p_hw"].values, df["p_sev"].values, th)
    out = {}
    for name, warn, obs in (
        ("yellow_vs_hot90", lvl >= 1, df["o_p90"].values),
        ("orange_vs_imd_hw", lvl >= 2, df["o_hw"].values),
        ("red_vs_imd_severe", lvl >= 3, df["o_sev"].values),
    ):
        out[name] = M.contingency(warn, obs)
    return out
