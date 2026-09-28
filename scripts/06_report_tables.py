"""Write LaTeX tables and number macros for the report straight from the metric files,
so every number in the PDF matches outputs/metrics."""
import json

import numpy as np
import pandas as pd

from heatcast.config import load_config
from heatcast.features import load_panel

cfg = load_config()
MET = cfg["paths"]["metrics"]
REP = cfg["paths"]["metrics"].parents[1] / "report"
ORDER = ["Chronos-2+cov-FT", "Chronos-2+cov", "Chronos-2+cov+RH", "Chronos-2", "Chronos-Bolt", "Chronos-T5",
         "Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]
BASELINES = ["Quantile-LSTM", "AR-anomaly", "Persistence", "Climatology"]


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def bold_best(vals, lower=True, d=3):
    s = pd.Series(vals, dtype=float)
    best = s.min() if lower else s.max()
    return [("\\textbf{" + f(v, d) + "}") if np.isclose(v, best) else f(v, d) for v in s]


def table(path, caption, label, header, rows, colspec, wide=False, note=None):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\fitbox{%",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}}"]
    if note:
        lines += [f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {note}}}"]
    lines += [f"\\end{{{env}}}", ""]
    (REP / path).write_text("\n".join(lines))


macros = {}
panel = load_panel(cfg)
test_years = cfg["years"]["test"]

# regions
cells = pd.read_csv(cfg["paths"]["processed"] / "region_cells.csv")
counts = pd.read_csv(MET / "event_counts_test.csv")
normals = pd.read_parquet(cfg["paths"]["processed"] / "normals.parquet")
rows = []
for _, r in cells.iterrows():
    n_may = normals[(normals["region"] == r["name"]) & normals["doy"].between(121, 151)]["normal_tmax"].mean()
    c = counts[counts["region"] == r["name"]].iloc[0]
    rows.append([r["name"], r["zone"], r["type"], f"{r['lat']:.2f}, {r['lon']:.2f}",
                 f"{r['cell_lat']:.1f}, {r['cell_lon']:.1f}", f"{n_may:.1f}", str(int(c["hot90_days"])),
                 str(int(c["hw_days"])), str(int(c["severe_days"]))])
box = counts[counts["group"] == "box"]
rows.append(["36 box cells", "Interior", "plains", "--", "16.5-21.5N, 74.5-79.5E", "--",
             str(int(box["hot90_days"].sum())), str(int(box["hw_days"].sum())), str(int(box["severe_days"].sum()))])
table("tab_regions.tex",
      "Focus regions, their IMD grid cells, mean May normal Tmax (1981-2010) and event days in the March-June test seasons 2019-2024.",
      "tab:regions", ["Region", "Zone", "Rule", "City (N, E)", "Cell (N, E)", "May $N$", "Hot", "HW", "Sev"],
      rows, "lllllrrrr", wide=True)

hot = panel[panel["date"].dt.month.isin([3, 4, 5, 6])]
tst = hot[hot["date"].dt.year.isin(test_years)]
val = hot[hot["date"].dt.year.isin(cfg["years"]["val"])]
ev_rows = []
for name, d in (("Validation 2015-2018", val), ("Test 2019-2024", tst)):
    ev_rows.append([name, f"{len(d):,}", f"{int(d['hot90'].sum()):,}", f"{d['hot90'].mean():.1%}".replace("%", "\\%"),
                    str(int((d['cls'] >= 1).sum())), str(int(d['hw_2day'].sum())), str(int((d['cls'] == 2).sum()))])
table("tab_events.tex", "Event days over all 43 series (March-June).", "tab:events",
      ["Period", "Series-days", "Hot days", "Rate", "HW", "HW 2-day", "Severe"], ev_rows, "lrrrrrr")
macros["NHWTest"] = str(int((tst["cls"] >= 1).sum()))
macros["NSevTest"] = str(int((tst["cls"] == 2).sum()))
macros["HotRateTest"] = f"{tst['hot90'].mean() * 100:.1f}\\%"
macros["NSeries"] = str(panel["region"].nunique())

# continuous skill, all series
ca = pd.read_csv(MET / "continuous_all.csv", index_col=0)
bl = pd.read_csv(MET / "continuous_by_lead_all.csv")
ca = ca.reindex([m for m in ORDER if m in ca.index])
lead_crps = bl.pivot(index="model", columns="lead", values="CRPS")
cols = {k: bold_best(ca[k].values, lower=(k not in ("CRPSS",))) for k in ("MAE", "RMSE", "CRPS", "CRPSS")}
l1 = bold_best(lead_crps.reindex(ca.index)[1].values)
l3 = bold_best(lead_crps.reindex(ca.index)[3].values)
l7 = bold_best(lead_crps.reindex(ca.index)[7].values)
rows = [[m, cols["MAE"][i], cols["RMSE"][i], cols["CRPS"][i], cols["CRPSS"][i], f(ca["Cov80"].iloc[i], 2),
         f(ca["Width80"].iloc[i], 2), l1[i], l3[i], l7[i]] for i, m in enumerate(ca.index)]
table("tab_continuous.tex",
      "Tmax forecast skill over all 43 series, test seasons 2019-2024, leads 1-7 pooled (deg C). CRPSS is relative to climatology. Cov80 is the share of observations inside the 10-90\\% interval (target 0.80). Best value in bold.",
      "tab:continuous", ["Model", "MAE", "RMSE", "CRPS", "CRPSS", "Cov80", "Width80", "CRPS d1", "CRPS d3", "CRPS d7"],
      rows, "lrrrrrrrrr", wide=True)
macros["NTestPairs"] = f"{int(ca['n'].iloc[0]):,}".replace(",", "{,}")
macros["NTestForecasts"] = f"{int(ca['n'].iloc[0]) // 7:,}".replace(",", "{,}")
fm = ca.loc[[m for m in ca.index if m.startswith("Chronos")], "CRPS"]
bb = ca.loc[[m for m in ca.index if m in BASELINES], "CRPS"]
macros["BestFM"] = fm.idxmin()
macros["CRPSbestFM"] = f(fm.min())
macros["BestBaseline"] = bb.idxmin()
macros["CRPSbaseline"] = f(bb.min())
for m in ca.index:
    key = m.replace("-", "").replace("+", "").replace(" ", "")
    macros[f"CRPS{key}"] = f(ca.loc[m, "CRPS"])
    macros[f"CRPSS{key}"] = f(ca.loc[m, "CRPSS"])
    macros[f"MAE{key}"] = f(ca.loc[m, "MAE"], 2)
    macros[f"Cov{key}"] = f(ca.loc[m, "Cov80"], 2)
    for ld in (1, 3, 7):
        macros[f"CRPS{key}L{['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][ld]}"] = f(lead_crps.loc[m, ld])

# validation CRPS and CRPS by test year
cv = pd.read_csv(MET / "continuous_val_all.csv", index_col=0)
for m in cv.index:
    macros[f"ValCRPS{m.replace('-', '').replace('+', '')}"] = f(cv.loc[m, "CRPS"])
by = pd.read_csv(MET / "crps_by_year_all.csv", index_col=0)
by.columns = [int(c) for c in by.columns]
better = int((by.loc["Chronos-2+cov"] < by.loc["Chronos-2"]).sum())
macros["CovBetterYears"] = str(better)
macros["CovYearsTotal"] = str(len(by.columns))
for y in by.columns:
    for m in ("Chronos-2+cov", "Chronos-2"):
        macros[f"CRPS{m.replace('-', '').replace('+', '')}Y{y}"] = f(by.loc[m, y])

# events, all series
e9 = pd.read_csv(MET / "events_p90_lead1-3_all.csv", index_col=0).reindex(ca.index)
eh = pd.read_csv(MET / "events_hw_lead1-3_all.csv", index_col=0).reindex(ca.index)
b9, a9 = bold_best(e9["BSS"].values, False), bold_best(e9["AUC"].values, False)
bh, ah = bold_best(eh["BSS"].values, False), bold_best(eh["AUC"].values, False)
rows = [[m, f(e9["Brier"].iloc[i], 4), b9[i], a9[i], f(eh["Brier"].iloc[i], 4), bh[i], ah[i]]
        for i, m in enumerate(ca.index) if m != "Climatology"]
table("tab_eventskill.tex",
      f"Event probability skill over all 43 series, leads 1-3, test seasons ({int(e9['events'].iloc[0]):,} hot-day and {int(eh['events'].iloc[0])} heatwave forecast-observation pairs out of {int(e9['n'].iloc[0]):,}). BSS is relative to climatology.",
      "tab:eventskill", ["Model", "BS hot", "BSS hot", "AUC hot", "BS HW", "BSS HW", "AUC HW"], rows, "lrrrrrr")
for m in ca.index:
    key = m.replace("-", "").replace("+", "")
    macros[f"BSShot{key}"] = f(e9.loc[m, "BSS"])
    macros[f"BSShw{key}"] = f(eh.loc[m, "BSS"])
    macros[f"AUChw{key}"] = f(eh.loc[m, "AUC"])
    macros[f"AUChot{key}"] = f(e9.loc[m, "AUC"])

# focus regions, all models
cf = pd.read_csv(MET / "continuous_focus.csv", index_col=0)
cf = cf.reindex([m for m in ORDER if m in cf.index])
e9f = pd.read_csv(MET / "events_p90_lead1-3_focus.csv", index_col=0).reindex(cf.index)
c1, c2, c3 = bold_best(cf["CRPS"].values), bold_best(cf["MAE"].values), bold_best(e9f["BSS"].values, False)
rows = [[m, c2[i], c1[i], f(cf["CRPSS"].iloc[i]), f(cf["Cov80"].iloc[i], 2), c3[i], f(e9f["AUC"].iloc[i])]
        for i, m in enumerate(cf.index)]
table("tab_focus.tex",
      "Skill at the 7 focus regions (all models, test seasons). MAE and CRPS pool leads 1-7. Hot-day BSS and AUC use leads 1-3.",
      "tab:focus", ["Model", "MAE", "CRPS", "CRPSS", "Cov80", "BSS hot", "AUC hot"], rows, "lrrrrrr")
for m in cf.index:
    key = m.replace("-", "").replace("+", "")
    macros[f"FocusCRPS{key}"] = f(cf.loc[m, "CRPS"])

# warnings (focus regions, every model on the same forecasts)
w_all = pd.read_csv(MET / "warnings_test_lead1-3.csv")
w = pd.read_csv(MET / "warnings_focus_lead1-3.csv")
th = json.loads((MET / "warning_thresholds.json").read_text())
rows = []
for m in [x for x in ORDER if x in th]:
    y = w[(w["model"] == m) & (w["check"] == "yellow_vs_hot90")].iloc[0]
    o = w[(w["model"] == m) & (w["check"] == "orange_vs_imd_hw")].iloc[0]
    rows.append([m, f(th[m]["t_watch"], 2), f(y["POD"], 2), f(y["FAR"], 2), f(y["CSI"], 2),
                 f(th[m]["t_alert"], 2), f(o["POD"], 2), f(o["FAR"], 2), f(o["CSI"], 2)])
table("tab_warnings.tex",
      "Warning verification at the 7 focus regions, leads 1-3, test seasons. Thresholds $t_w$ and $t_a$ were tuned for best CSI on 2015-2018. Yellow is checked against hot days, Orange against IMD heatwave days.",
      "tab:warnings", ["Model", "$t_w$", "POD", "FAR", "CSI", "$t_a$", "POD", "FAR", "CSI"], rows, "lrrrrrrrr",
      wide=True)
for m in th:
    key = m.replace("-", "").replace("+", "")
    o = w[(w["model"] == m) & (w["check"] == "orange_vs_imd_hw")].iloc[0]
    y = w[(w["model"] == m) & (w["check"] == "yellow_vs_hot90")].iloc[0]
    macros[f"CSIorange{key}"] = f(o["CSI"], 2)
    macros[f"CSIyellow{key}"] = f(y["CSI"], 2)
    macros[f"PODorange{key}"] = f(o["POD"], 2)
    macros[f"FARorange{key}"] = f(o["FAR"], 2)
for m in w_all["model"].unique():
    key = m.replace("-", "").replace("+", "")
    o = w_all[(w_all["model"] == m) & (w_all["check"] == "orange_vs_imd_hw")].iloc[0]
    y = w_all[(w_all["model"] == m) & (w_all["check"] == "yellow_vs_hot90")].iloc[0]
    macros[f"AllCSIorange{key}"] = f(o["CSI"], 2)
    macros[f"AllCSIyellow{key}"] = f(y["CSI"], 2)
sev = w_all[w_all["check"] == "red_vs_imd_severe"]
macros["NSevPairs"] = str(int(sev["hits"].iloc[0] + sev["misses"].iloc[0]))

# bootstrap
bs = json.loads((MET / "bootstrap.json").read_text())
macros["BootFM"], macros["BootBL"] = bs["fm"], bs["baseline"]
for k in ("CRPS", "Brier_hw", "Brier_p90", "CRPS_lead1", "CRPS_lead3", "CRPS_lead7"):
    kk = k.replace("_", "")
    d = 4 if "Brier" in k else 3
    macros[f"Boot{kk}"] = f(bs[k]["diff_mean"], d)
    macros[f"Boot{kk}lo"] = f(bs[k]["ci95"][0], d)
    macros[f"Boot{kk}hi"] = f(bs[k]["ci95"][1], d)

# localization
lp = MET / "localization_lead1-3.csv"
if lp.exists():
    lo = pd.read_csv(lp)
    maps = pd.read_csv(MET / "grid_point_maps.csv")
    r_ = maps.groupby("region")[["r", "bias_point_minus_grid"]].mean()
    rows = [[x["region"], f(r_.loc[x["region"], "r"], 2), f(r_.loc[x["region"], "bias_point_minus_grid"], 2),
             f(x["MAE_raw"], 2), f(x["MAE_corrected"], 2), f(x["CRPS_raw"], 2), f(x["CRPS_corrected"], 2),
             str(int(x["point_hw_days"]))] for _, x in lo.iterrows()]
    table("tab_localization.tex",
          "Grid forecasts (Chronos-2+cov) checked against city-point Tmax, leads 1-3, test seasons. $r$ and bias (point minus grid, deg C) are from 1990-2014 March-July observations. HW is the number of point-level IMD heatwave days.",
          "tab:localization", ["Region", "$r$", "Bias", "MAE grid", "MAE corr.", "CRPS grid", "CRPS corr.", "HW"],
          rows, "lrrrrrrr", wide=True)
    macros["LocMAEraw"] = f(lo["MAE_raw"].mean(), 2)
    macros["LocMAEcor"] = f(lo["MAE_corrected"].mean(), 2)
    macros["LocCRPSraw"] = f(lo["CRPS_raw"].mean(), 2)
    macros["LocCRPScor"] = f(lo["CRPS_corrected"].mean(), 2)
    macros["LocPointHW"] = str(int(lo["point_hw_days"].sum()))

# humid heat
hh = pd.read_csv(MET / "humid_heat_by_year.csv")
for r in ("Mumbai", "Ratnagiri", "Pune"):
    x = hh[hh["region"] == r]
    macros[f"Humid{r}"] = str(int(x["humid_missed"].sum()))
    macros[f"Tmaxhw{r}"] = str(int(x["tmax_rule_hw"].sum()))
    recent = x[x["year"] >= 2015]
    macros[f"HumidRecent{r}"] = str(int(recent["humid_missed"].sum()))
    macros[f"HumidPerYear{r}"] = f"{x['humid_missed'].mean():.1f}"
    macros[f"HImax{r}"] = f"{x['hi_max'].max():.1f}"
    macros[f"IMDOrange{r}"] = str(int(x["imd_orange_or_red"].sum()))

# T5 sample vs quantile
tp = MET / "t5_sample_vs_quantile.json"
if tp.exists():
    t5 = json.loads(tp.read_text())
    macros["TFiveCorr"] = f(t5["p90_corr"], 3)
    macros["TFiveBSsamp"] = f(t5["p90_brier_samples"], 4)
    macros["TFiveBSquant"] = f(t5["p90_brier_quantiles"], 4)
ftp = MET / "finetune.json"
if ftp.exists():
    ft = json.loads(ftp.read_text())
    macros["FTsteps"] = str(ft["steps"])
    macros["FTminutes"] = f"{ft['minutes']:.0f}"
macros["TFiveSamples"] = str(cfg["forecast"]["t5_samples"])

# timing
import numpy as np  # noqa: E402
times = {}
for p in cfg["paths"]["predictions"].glob("*__2019_2024__*.npz"):
    z = np.load(p)
    times[p.stem.split("__")[0]] = (float(z["seconds"]), len(z["s_idx"]))
for m, (sec, n) in times.items():
    key = m.replace("-", "").replace("+", "")
    macros[f"Speed{key}"] = f"{n / max(sec, 1e-9):.0f}" if sec > 0.5 else "$>$1000"

lines = []
for k, v in macros.items():
    k2 = "".join(ch for ch in k if ch.isalpha())
    num = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine"}
    k2 = "".join(num.get(ch, ch) for ch in k if ch.isalnum())
    k2 = "".join(ch for ch in k2 if ch.isalpha())
    lines.append(f"\\newcommand{{\\{k2}}}{{{v}}}")
(REP / "results_macros.tex").write_text("\n".join(lines) + "\n")
print(f"{len(macros)} macros, tables written to {REP}")
