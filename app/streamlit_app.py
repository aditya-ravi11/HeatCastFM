"""HeatCast-FM viewer: 7 day Tmax forecast, heatwave probability and warning colour.

run: .venv/bin/streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heatcast import warning as W  # noqa: E402
from heatcast.config import load_config  # noqa: E402
from heatcast.features import SeriesStore, load_panel  # noqa: E402
from heatcast.probability import LEVELS, exceed_prob  # noqa: E402

LEVEL_STYLE = {
    0: ("Green", "#1f7a3a", "No warning"),
    1: ("Yellow", "#b58100", "Heat watch: be aware"),
    2: ("Orange", "#c4540c", "Heatwave alert: be prepared"),
    3: ("Red", "#b3261e", "Severe heatwave: take action"),
}
MODELS = ["Chronos-2+cov-FT", "Chronos-2+cov", "Chronos-2+cov+RH", "Chronos-2", "Chronos-Bolt", "Chronos-T5",
          "Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]

st.set_page_config(page_title="HeatCast-FM", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:1rem}</style>", unsafe_allow_html=True)


@st.cache_resource
def load():
    cfg = load_config()
    panel = load_panel(cfg)
    store = SeriesStore(panel)
    preds = {}
    for p in sorted(cfg["paths"]["predictions"].glob("*.npz")):
        model, years, grp = p.stem.split("__")
        z = np.load(p)
        preds.setdefault(model, []).append((z["q"], z["s_idx"], z["t_pos"]))
    th = {}
    tf = cfg["paths"]["metrics"] / "warning_thresholds.json"
    if tf.exists():
        import json
        th = json.loads(tf.read_text())
    return cfg, store, preds, th


@st.cache_resource
def chronos2(model_id):
    from heatcast.models.chronos_models import Chronos2
    return Chronos2(model_id, covariates="basic")


def cached_forecast(preds, model, s, t):
    for q, s_idx, t_pos in preds.get(model, []):
        hit = np.where((s_idx == s) & (t_pos == t))[0]
        if len(hit):
            return q[hit[0]]
    return None


cfg, store, preds, thresholds = load()
H = cfg["forecast"]["horizon"]
regions = [r["name"] for r in cfg["regions"]]

st.title("HeatCast-FM")
st.caption("7 day maximum temperature forecast and IMD heatwave warning for Maharashtra, "
           "from Chronos time-series foundation models. Data: IMD 1 degree gridded Tmax.")

c1, c2, c3 = st.columns([1, 1, 1])
# optional URL parameters, e.g. ?region=Nagpur&date=2019-05-31&model=Chronos-2%2Bcov
qp = st.query_params
model_list = [m for m in MODELS if m in preds] or MODELS
region = c1.selectbox("Region", regions,
                      index=regions.index(qp["region"]) if qp.get("region") in regions else 0)
model = c2.selectbox("Model", model_list,
                     index=model_list.index(qp["model"]) if qp.get("model") in model_list else 0)
years = cfg["years"]["val"] + cfg["years"]["test"]
default = pd.Timestamp(qp.get("date", f"{cfg['years']['test'][-1]}-05-10"))
date = pd.Timestamp(c3.date_input("Forecast issued on (last observed day)", default,
                                  min_value=pd.Timestamp(f"{years[0]}-01-01"),
                                  max_value=store.dates[-H - 1]))

s = store.index(region)
t = store.pos[date]
q = cached_forecast(preds, model, s, t)
source = "cached backtest"
if q is None:
    with st.spinner("No cached forecast for this date. Running Chronos-2 with covariates live..."):
        q = chronos2(cfg["models"]["chronos2"]).predict(store, np.array([s]), np.array([t]), H)[0]
    source = "live Chronos-2+cov run"
    model = "Chronos-2+cov"

tgt = np.arange(t + 1, t + H + 1)
thr_hw, thr_sev, thr_p90 = store.thr_hw[s, tgt], store.thr_sev[s, tgt], store.thr_p90[s, tgt]
p_hw = exceed_prob(thr_hw, q)
p_sev = exceed_prob(thr_sev, q)
p_p90 = exceed_prob(thr_p90, q)
th = thresholds.get(model, {"t_watch": 0.4, "t_alert": 0.3, "t_severe": 0.5})
lvl = W.level(p_p90, p_hw, p_sev, th)
worst = int(lvl.max())
name, colour, text = LEVEL_STYLE[worst]

st.markdown(
    f"<div style='padding:12px 16px;border-left:6px solid {colour};background:rgba(0,0,0,0.03)'>"
    f"<b style='color:{colour}'>{name}</b> for {region}, next {H} days: {text}. "
    f"Highest heatwave probability {p_hw.max():.0%} ({source}).</div>", unsafe_allow_html=True)

left, right = st.columns([5, 1])
with left:
    hist = np.arange(t - 20, t + 1)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    d_hist, d_tgt = store.dates[hist], store.dates[tgt]
    ax.plot(d_hist, store.tmax_obs[s, hist], color="#52514e", lw=1.8, label="Observed Tmax")
    i = {lv: k for k, lv in enumerate(LEVELS)}
    ax.fill_between(d_tgt, q[:, i[0.1]], q[:, i[0.9]], color="#2a78d6", alpha=0.18, lw=0, label="80% interval")
    ax.fill_between(d_tgt, q[:, i[0.25]], q[:, i[0.75]], color="#2a78d6", alpha=0.28, lw=0, label="50% interval")
    ax.plot(d_tgt, q[:, i[0.5]], color="#2a78d6", lw=2, label="Median forecast")
    obs_future = store.tmax_obs[s, tgt]
    if np.isfinite(obs_future).any():
        ax.plot(d_tgt, obs_future, "o", color="#0b0b0b", ms=4, label="Observed (verification)")
    ax.step(d_tgt, thr_hw, where="mid", color="#c4540c", lw=1.2, ls="--", label="IMD heatwave threshold")
    ax.step(d_tgt, thr_p90, where="mid", color="#b58100", lw=1.0, ls=":", label="Hot day (local 90th pct)")
    ax.set_ylabel("Tmax (deg C)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    st.pyplot(fig)

with right:
    st.metric("Warning level", name)
    st.metric("Max P(heatwave)", f"{p_hw.max():.0%}")
    st.metric("Max P(hot day)", f"{p_p90.max():.0%}")

table = pd.DataFrame({
    "Date": [d.strftime("%a %d %b") for d in store.dates[tgt]],
    "Median": np.round(q[:, 10], 1),
    "10-90%": [f"{a:.1f} to {b:.1f}" for a, b in zip(q[:, 2], q[:, 18])],
    "HW threshold": np.round(thr_hw, 1),
    "P(hot day)": [f"{p:.0%}" for p in p_p90],
    "P(heatwave)": [f"{p:.0%}" for p in p_hw],
    "P(severe)": [f"{p:.0%}" for p in p_sev],
    "Level": [LEVEL_STYLE[v][0] for v in lvl],
})
if np.isfinite(obs_future).any():
    table["Observed"] = [f"{v:.1f}" if np.isfinite(v) else "" for v in obs_future]
st.dataframe(table, hide_index=True, use_container_width=True, height=36 * (len(table) + 1) + 3)

st.caption(f"Region type: {store.rtype[s]}. IMD rule used: "
           + ("Tmax at least 37 deg C and at least 4.5 deg C above normal (coastal)."
              if store.rtype[s] == "coastal" else
              "Tmax at least 40 deg C and at least 4.5 deg C above normal, or Tmax at least 45 deg C (plains)."))
