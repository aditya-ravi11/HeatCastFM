"""Figures for the report (PDF for LaTeX, PNG for quick viewing)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from heatcast import plots as P
from heatcast.config import load_config
from heatcast.features import SeriesStore, load_panel
from heatcast.probability import LEVELS

KEY4 = ["Chronos-2+cov", "Chronos-2", "Quantile-LSTM", "AR-anomaly"]
ALL7 = ["Chronos-2+cov", "Chronos-2", "Chronos-Bolt", "Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]


def fig_thresholds(panel, out):
    """Normal Tmax and heatwave thresholds through the hot season, with 2024 observations."""
    regions = [("Mumbai", "coastal"), ("Pune", "plains"), ("Nagpur", "plains")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
    for ax, (r, t) in zip(axes, regions):
        g = panel[(panel["region"] == r) & (panel["date"].dt.year == 2019) & panel["date"].dt.month.isin([3, 4, 5, 6])]
        ax.plot(g["date"], g["tmax"], color=P.INK2, lw=1.0, label="Observed Tmax, 2019")
        ax.plot(g["date"], g["normal_tmax"], color="#6da7ec", lw=1.4, label="Normal (1981-2010)")
        ax.plot(g["date"], g["thr_hw"], color="#eb6834", lw=1.4, ls="--", label="Heatwave threshold")
        ax.plot(g["date"], g["thr_sev"], color="#b3261e", lw=1.2, ls=":", label="Severe threshold")
        hw = g[g["cls"] >= 1]
        ax.scatter(hw["date"], hw["tmax"], s=9, color="#b3261e", zorder=3, label="Heatwave day")
        ax.set_title(f"{r} ({t} rule)")
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
        P.clean(ax)
    axes[0].set_ylabel("Tmax (deg C)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    P.save(fig, out / "fig_thresholds.pdf")


def fig_skill_by_lead(met, out):
    d = pd.read_csv(met / "continuous_by_lead_all.csv")
    e = pd.read_csv(met / "events_p90_by_lead_all.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    for m in ALL7 + ["Chronos-2+cov-FT"]:
        x = d[d["model"] == m]
        if x.empty or m not in P.MODEL_COLOURS:
            continue
        P.line(axes[0], x["lead"], x["CRPS"], m, marker="o", ms=3)
        P.line(axes[1], x["lead"], x["Cov80"], m, marker="o", ms=3)
        y = e[e["model"] == m]
        if m != "Climatology":
            P.line(axes[2], y["lead"], y["BSS"], m, marker="o", ms=3)
    axes[0].set_ylabel("CRPS (deg C), lower is better")
    axes[1].set_ylabel("80% interval coverage")
    axes[1].axhline(0.8, color=P.INK2, lw=0.8, ls="--")
    axes[2].set_ylabel("BSS, hot day (vs climatology)")
    axes[2].axhline(0, color=P.INK2, lw=0.8)
    for ax in axes:
        ax.set_xlabel("Lead time (days)")
        ax.set_xticks(range(1, 8))
        P.clean(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 1))
    P.save(fig, out / "fig_skill_by_lead.pdf")


def fig_reliability_roc(met, out):
    rel = pd.read_csv(met / "reliability_p90_lead1-3.csv")
    long = pd.read_parquet(met / "test_long.parquet")
    long = long[(long["group"].isin(["focus", "box"])) & (long["lead"] <= 3)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=P.INK2, lw=0.8, ls="--")
    for m in KEY4:
        r = rel[rel["model"] == m]
        P.line(ax, r["p_mean"], r["o_freq"], m, marker="o", ms=3)
    ax.set_xlabel("Forecast probability of a hot day")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability, hot day, leads 1-3")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    P.clean(ax, "both")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=P.INK2, lw=0.8, ls="--")
    for m in KEY4:
        x = long[long["model"] == m]
        fpr, tpr, _ = roc_curve(x["o_hw"], x["p_hw"])
        P.line(ax, fpr, tpr, m)
    ax.set_xlabel("False alarm rate (POFD)")
    ax.set_ylabel("Hit rate (POD)")
    ax.set_title("ROC, IMD heatwave, leads 1-3")
    P.clean(ax, "both")
    fig.tight_layout(w_pad=2)
    P.save(fig, out / "fig_reliability_roc.pdf")


def fig_cases(met, cfg, store, out):
    ep = pd.read_csv(met / "case_episodes.csv", parse_dates=["start", "end"])
    models = ["Chronos-2+cov", "AR-anomaly"]
    preds = {m: np.load(cfg["paths"]["predictions"] / f"{m}__{cfg['years']['test'][0]}_{cfg['years']['test'][-1]}__all.npz")
             for m in models}
    fig, axes = plt.subplots(1, len(ep), figsize=(7.1, 2.25), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (_, e) in zip(axes, ep.iterrows()):
        s = store.index(e["region"])
        t0 = store.pos[e["start"]] - 2  # forecast issued two days before the first heatwave day
        hist = np.arange(t0 - 12, t0 + 1)
        tgt = np.arange(t0 + 1, t0 + 8)
        ax.plot(store.dates[np.r_[hist, tgt]], store.tmax_obs[s, np.r_[hist, tgt]], color=P.INK2, lw=1.1,
                label="Observed")
        ax.plot(store.dates[tgt], store.thr_hw[s, tgt], color="#eb6834", lw=1.1, ls="--", label="Heatwave threshold")
        for m in models:
            z = preds[m]
            k = np.where((z["s_idx"] == s) & (z["t_pos"] == t0))[0]
            if not len(k):
                continue
            q = z["q"][k[0]]
            c = P.MODEL_COLOURS[m]
            ax.fill_between(store.dates[tgt], q[:, 2], q[:, 18], color=c, alpha=0.15, lw=0)
            ax.plot(store.dates[tgt], q[:, 10], color=c, lw=1.5, label=f"{m} median, 80% band")
        ax.axvline(store.dates[t0], color=P.INK2, lw=0.6, ls=":")
        ax.set_title(f"{e['region']}, issued {store.dates[t0]:%d %b %Y}")
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
        P.clean(ax)
    axes[0].set_ylabel("Tmax (deg C)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1, rect=(0, 0.07, 1, 1))
    P.save(fig, out / "fig_cases.pdf")


def fig_humid(met, out):
    d = pd.read_csv(met / "humid_heat_by_year.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), sharey=True)
    for ax, r in zip(axes, ["Mumbai", "Ratnagiri", "Pune"]):
        x = d[d["region"] == r]
        w = 0.42
        ax.bar(x["year"] - w / 2, x["tmax_rule_hw"], width=w, color="#2a78d6", label="Heatwave days, IMD Tmax rule")
        ax.bar(x["year"] + w / 2, x["humid_missed"], width=w, color="#eb6834",
               label="Heat index 39.4+ deg C (NOAA Danger), no Tmax heatwave")
        ax.set_title(r + (" (inland)" if r == "Pune" else " (coast)"))
        P.clean(ax)
    axes[0].set_ylabel("Days in March to June")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    P.save(fig, out / "fig_humid_heat.pdf")


def fig_localization(met, out):
    d = pd.read_csv(met / "localization_lead1-3.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["MAE_raw"], w, color="#86b6ef", label="Grid forecast vs point")
    ax.bar(x + w / 2, d["MAE_corrected"], w, color="#1c5cab", label="After point correction")
    ax.set_xticks(x, d["region"], rotation=35, ha="right")
    ax.set_ylabel("MAE at the city point (deg C)")
    P.clean(ax)
    ax.legend(loc="upper left")
    P.save(fig, out / "fig_localization.pdf")


def fig_lstm(met, out):
    import json
    h = pd.DataFrame(json.loads((met / "lstm_history.json").read_text()))
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    ax.plot(h["epoch"], h["val_pinball"], color=P.MODEL_COLOURS["Quantile-LSTM"], marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation pinball loss")
    P.clean(ax)
    P.save(fig, out / "fig_lstm_training.pdf")


if __name__ == "__main__":
    P.setup()
    cfg = load_config()
    met, out = cfg["paths"]["metrics"], cfg["paths"]["figures"]
    panel = load_panel(cfg)
    store = SeriesStore(panel)
    fig_thresholds(panel, out)
    fig_skill_by_lead(met, out)
    fig_reliability_roc(met, out)
    fig_cases(met, cfg, store, out)
    fig_humid(met, out)
    if (met / "localization_lead1-3.csv").exists():
        fig_localization(met, out)
    if (met / "lstm_history.json").exists():
        fig_lstm(met, out)
    print("figures written to", out)
