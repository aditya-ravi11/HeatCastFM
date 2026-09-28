"""Score every cached prediction and write the tables used by the report and the app."""
import json

import numpy as np
import pandas as pd

from heatcast import heat_index as HI
from heatcast import localize
from heatcast import metrics as M
from heatcast import warning as W
from heatcast.backtest import to_frame
from heatcast.config import load_config
from heatcast.features import SeriesStore, load_panel

ORDER = ["Chronos-2+cov-FT", "Chronos-2+cov", "Chronos-2+cov+RH", "Chronos-2", "Chronos-Bolt", "Chronos-T5",
         "Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]
BASELINES = ["Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]


def load_all(cfg, store, years):
    frames = []
    for name in ORDER:
        for grp in ("all", "focus"):
            p = cfg["paths"]["predictions"] / f"{name}__{years[0]}_{years[-1]}__{grp}.npz"
            if p.exists():
                frames.append(to_frame(p, store, name))
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["target"].dt.year
    return df


def continuous_table(df, by=("model",)):
    g = df.groupby(list(by))
    t = pd.DataFrame({
        "MAE": g.apply(lambda x: (x["median"] - x["obs"]).abs().mean()),
        "RMSE": g.apply(lambda x: np.sqrt(((x["median"] - x["obs"]) ** 2).mean())),
        "CRPS": g["crps"].mean(),
        "Cov80": g["in80"].mean(),
        "Width80": g["width80"].mean(),
        "n": g.size(),
    })
    return t


def event_table(df, clim_p, event="hw"):
    rows = []
    for model, x in df.groupby("model"):
        ref = clim_p.loc[x.index]
        p, o = x[f"p_{event}"].values, x[f"o_{event}"].values
        rows.append({"model": model, "Brier": M.brier(p, o), "BSS": M.brier_skill(p, ref.values, o),
                     "AUC": M.auc(p, o), "events": int(o.sum()), "n": len(o)})
    return pd.DataFrame(rows).set_index("model")


def attach_clim(df):
    """Climatology probability for the same (series, origin, lead), used as the BSS reference."""
    key = ["series", "origin", "lead"]
    c = df[df["model"] == "Climatology"].set_index(key)
    joined = df.join(c[["p_hw", "p_sev", "p_p90"]], on=key, rsuffix="_clim")
    return joined


def sort_models(t):
    return t.reindex([m for m in ORDER if m in t.index])


def episodes(panel, years, regions, top=3):
    """Longest runs of heatwave days at focus regions in the test years."""
    p = panel[(panel["group"] == "focus") & panel["date"].dt.year.isin(years)].sort_values(["region", "date"])
    out = []
    for r, g in p.groupby("region"):
        hw = (g["cls"] >= 1).values
        d = g["date"].values
        i = 0
        while i < len(hw):
            if hw[i]:
                j = i
                while j + 1 < len(hw) and hw[j + 1]:
                    j += 1
                seg = g.iloc[i:j + 1]
                out.append({"region": r, "start": pd.Timestamp(d[i]), "end": pd.Timestamp(d[j]), "days": j - i + 1,
                            "peak_tmax": float(seg["tmax"].max()), "peak_dep": float(seg["dep"].max()),
                            "severe_days": int((seg["cls"] == 2).sum())})
                i = j + 1
            else:
                i += 1
    ep = pd.DataFrame(out).sort_values(["days", "peak_dep"], ascending=False)
    # one episode per region so the case studies cover different places
    return ep.drop_duplicates("region").head(top), pd.DataFrame(out)


if __name__ == "__main__":
    cfg = load_config()
    out = cfg["paths"]["metrics"]
    panel = load_panel(cfg)
    store = SeriesStore(panel)
    val_years, test_years = cfg["years"]["val"], cfg["years"]["test"]

    test = attach_clim(load_all(cfg, store, test_years))
    val = attach_clim(load_all(cfg, store, val_years))
    test.to_parquet(out / "test_long.parquet", index=False)

    # common evaluation sets: all 43 series (7 models) and 7 focus regions (all models)
    FOCUS_ONLY = ("Chronos-2+cov+RH", "Chronos-T5", "Chronos-2+cov-FT")
    all_models = [m for m in ORDER if m not in FOCUS_ONLY]
    sets = {
        "all": test[test["model"].isin(all_models)],
        "focus": test[test["group"] == "focus"],
    }
    summary = {}
    for sname, d in sets.items():
        ct = sort_models(continuous_table(d))
        clim_crps = ct.loc["Climatology", "CRPS"]
        ct["CRPSS"] = 1 - ct["CRPS"] / clim_crps
        ct.round(4).to_csv(out / f"continuous_{sname}.csv")
        bylead = continuous_table(d, ("model", "lead")).reset_index()
        bylead.to_csv(out / f"continuous_by_lead_{sname}.csv", index=False)
        early = d[d["lead"] <= 3]
        et = sort_models(event_table(early, early["p_hw_clim"], "hw"))
        e9 = sort_models(event_table(early, early["p_p90_clim"], "p90"))
        es = sort_models(event_table(early, early["p_sev_clim"], "sev"))
        et.round(4).to_csv(out / f"events_hw_lead1-3_{sname}.csv")
        e9.round(4).to_csv(out / f"events_p90_lead1-3_{sname}.csv")
        es.round(4).to_csv(out / f"events_sev_lead1-3_{sname}.csv")
        for ev in ("hw", "p90"):
            ev_lead = []
            for lead, x in d.groupby("lead"):
                t = event_table(x, x[f"p_{ev}_clim"], ev).reset_index()
                t["lead"] = lead
                ev_lead.append(t)
            pd.concat(ev_lead).to_csv(out / f"events_{ev}_by_lead_{sname}.csv", index=False)
        summary[sname] = {"continuous": ct.round(4).to_dict(), "events_hw": et.round(4).to_dict(),
                          "events_p90": e9.round(4).to_dict()}

    # validation-year skill and test skill per year (covariate effect by season)
    vset = val[val["model"].isin(all_models)]
    sort_models(continuous_table(vset)).round(4).to_csv(out / "continuous_val_all.csv")
    by_year = sets["all"].groupby(["model", "year"])["crps"].mean().unstack().round(4)
    by_year.to_csv(out / "crps_by_year_all.csv")

    # warnings: thresholds tuned on validation years per model, scored on test years
    warn_rows, thresholds = [], {}
    for model in ORDER:
        v = val[val["model"] == model]
        t = sets["focus"][sets["focus"]["model"] == model] if model in FOCUS_ONLY \
            else sets["all"][sets["all"]["model"] == model]
        if v.empty or t.empty:
            continue
        th = W.tune(v)
        thresholds[model] = th
        early = t[t["lead"] <= 3]
        res = W.evaluate(early, th)
        for k, c in res.items():
            warn_rows.append({"model": model, "check": k, **c})
    pd.DataFrame(warn_rows).to_csv(out / "warnings_test_lead1-3.csv", index=False)
    # same thresholds, every model scored on the 7 focus regions so all rows are comparable
    focus_rows = []
    for model, th in thresholds.items():
        t = sets["focus"][(sets["focus"]["model"] == model) & (sets["focus"]["lead"] <= 3)]
        for k, c in W.evaluate(t, th).items():
            focus_rows.append({"model": model, "check": k, **c})
    pd.DataFrame(focus_rows).to_csv(out / "warnings_focus_lead1-3.csv", index=False)
    (out / "warning_thresholds.json").write_text(json.dumps(thresholds, indent=1))

    # reliability (leads 1-3, all series set)
    for ev in ("hw", "p90"):
        rel = []
        for model, x in sets["all"][sets["all"]["lead"] <= 3].groupby("model"):
            r = M.reliability(x[f"p_{ev}"].values, x[f"o_{ev}"].values)
            r["model"] = model
            rel.append(r)
        pd.concat(rel).to_csv(out / f"reliability_{ev}_lead1-3.csv", index=False)

    # bootstrap: best foundation model vs best baseline (by CRPS, all series)
    ct = pd.read_csv(out / "continuous_all.csv", index_col=0)
    fm = ct.loc[[m for m in ct.index if m.startswith("Chronos")], "CRPS"].idxmin()
    bl = ct.loc[[m for m in ct.index if m in BASELINES], "CRPS"].idxmin()
    a = sets["all"][sets["all"]["model"] == fm].set_index(["series", "origin", "lead"]).sort_index()
    b = sets["all"][sets["all"]["model"] == bl].set_index(["series", "origin", "lead"]).sort_index()
    a, b = a.align(b, join="inner", axis=0)
    blocks = (a.reset_index()["series"] + "_" + a["year"].astype(str).values).values
    boot = {"fm": fm, "baseline": bl}
    for metric, fa, fb in (("CRPS", a["crps"].values, b["crps"].values),
                           ("Brier_hw", (a["p_hw"] - a["o_hw"]).values ** 2, (b["p_hw"] - b["o_hw"]).values ** 2),
                           ("Brier_p90", (a["p_p90"] - a["o_p90"]).values ** 2, (b["p_p90"] - b["o_p90"]).values ** 2)):
        mean, lo, hi = M.block_bootstrap_diff(fa, fb, blocks)
        boot[metric] = {"diff_mean": mean, "ci95": [lo, hi]}
    for lead in (1, 3, 7):
        m = a.reset_index()["lead"].values == lead
        mean, lo, hi = M.block_bootstrap_diff(a["crps"].values[m], b["crps"].values[m], blocks[m])
        boot[f"CRPS_lead{lead}"] = {"diff_mean": mean, "ci95": [lo, hi]}
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=1))

    # per region (focus) for the headline model and the best baseline
    reg = continuous_table(sets["focus"][sets["focus"]["lead"] <= 3], ("series", "model")).reset_index()
    reg.to_csv(out / "by_region_lead1-3.csv", index=False)

    # T5: probabilities from raw sample paths vs from the fitted quantiles
    t5 = test[(test["model"] == "Chronos-T5") & (test["lead"] <= 3)]
    if not t5.empty and "p_hw_samples" in t5:
        (out / "t5_sample_vs_quantile.json").write_text(json.dumps({
            "hw_brier_samples": M.brier(t5["p_hw_samples"].values, t5["o_hw"].values),
            "hw_brier_quantiles": M.brier(t5["p_hw"].values, t5["o_hw"].values),
            "p90_brier_samples": M.brier(t5["p_p90_samples"].values, t5["o_p90"].values),
            "p90_brier_quantiles": M.brier(t5["p_p90"].values, t5["o_p90"].values),
            "p90_corr": float(np.corrcoef(t5["p_p90_samples"], t5["p_p90"])[0, 1]),
        }, indent=1))

    # event counts in the test years
    hot = panel[panel["date"].dt.year.isin(test_years) & panel["date"].dt.month.isin([3, 4, 5, 6])]
    counts = hot.groupby(["group", "region"]).agg(hw_days=("cls", lambda s: int((s >= 1).sum())),
                                                  severe_days=("cls", lambda s: int((s == 2).sum())),
                                                  hot90_days=("hot90", "sum"),
                                                  hw_2day=("hw_2day", "sum")).reset_index()
    counts.to_csv(out / "event_counts_test.csv", index=False)
    top, allep = episodes(panel, test_years, None)
    top.to_csv(out / "case_episodes.csv", index=False)
    allep.to_csv(out / "all_episodes_test.csv", index=False)

    # Objective 3: grid forecasts against the station proxy
    best_path = cfg["paths"]["predictions"] / f"Chronos-2+cov__{test_years[0]}_{test_years[-1]}__all.npz"
    if best_path.exists():
        loc, maps = localize.evaluate(best_path, store, panel, cfg)
        maps.to_csv(out / "grid_point_maps.csv", index=False)
        loc.to_csv(out / "localization_long.csv", index=False)
        early = loc[loc["lead"] <= 3]
        rows = []
        for region, x in early.groupby("region"):
            rows.append({
                "region": region,
                "MAE_raw": (x["raw_median"] - x["obs_point"]).abs().mean(),
                "MAE_corrected": (x["cor_median"] - x["obs_point"]).abs().mean(),
                "CRPS_raw": x["raw_crps"].mean(), "CRPS_corrected": x["cor_crps"].mean(),
                "Brier_raw": M.brier(x["raw_p_hw"].values, x["o_hw_point"].values),
                "Brier_corrected": M.brier(x["cor_p_hw"].values, x["o_hw_point"].values),
                "point_hw_days": int(x.drop_duplicates("target")["o_hw_point"].sum()),
            })
        pd.DataFrame(rows).round(4).to_csv(out / "localization_lead1-3.csv", index=False)

    # humid heat at the coast (station proxy)
    point = localize.point_panel(cfg)
    hh = HI.humid_heat_table(point, ["Mumbai", "Ratnagiri", "Pune"])
    hh.to_parquet(out / "humid_heat_days.parquet", index=False)
    per_year = hh.groupby(["region", "year"]).agg(
        tmax_rule_hw=("tmax_rule_hw", "sum"), humid_missed=("humid_missed", "sum"),
        hi_danger=("hi_danger", "sum"), hi_max=("hi", "max"),
        imd_orange_or_red=("hi", lambda s: int((s >= 45.5).sum()))).reset_index()
    per_year.to_csv(out / "humid_heat_by_year.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(pd.read_csv(out / "continuous_all.csv", index_col=0))
    print(pd.read_csv(out / "events_hw_lead1-3_all.csv", index_col=0))
    print(pd.read_csv(out / "events_p90_lead1-3_all.csv", index_col=0))
    print(json.dumps(boot, indent=1))
