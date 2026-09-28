"""Daily normals, departures and IMD heatwave labels.

IMD criteria (FAQ on Heat Wave, IMD) for a station, written as thresholds on Tmax:
  plains   heatwave if Tmax >= 40 and departure >= 4.5, or Tmax >= 45
           severe   if Tmax >= 40 and departure >= 6.5, or Tmax >= 47
  coastal  heatwave if Tmax >= 37 and departure >= 4.5
           severe   if Tmax >= 37 and departure >= 6.5 (our extension, IMD gives no coastal severe rule)

Because departure = Tmax - normal, each rule reduces to Tmax >= threshold(normal),
which makes probability of an event a single exceedance probability.
"""
import numpy as np
import pandas as pd

SMOOTH_DAYS = 31


def doy365(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365 day calendar (29 Feb shares 28 Feb's index)."""
    d = pd.DatetimeIndex(dates)
    doy = d.dayofyear.values.copy()
    leap_after_feb = d.is_leap_year & ((d.month > 2) | ((d.month == 2) & (d.day == 29)))
    doy[leap_after_feb] -= 1
    return doy


def _circular_smooth(values: np.ndarray, window: int = SMOOTH_DAYS) -> np.ndarray:
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


P90_HALF_WINDOW = 7


def compute_normals(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Per region, over [start, end]: smoothed day-of-year mean Tmax and Tmin, and the
    90th percentile of Tmax within +/-7 days of each calendar day (hot-day threshold)."""
    base = df[(df["date"].dt.year >= start) & (df["date"].dt.year <= end)].copy()
    base["doy"] = doy365(base["date"])
    out = []
    for region, g in base.groupby("region"):
        m = g.groupby("doy")[["tmax", "tmin"]].mean().reindex(range(1, 366)).interpolate()
        doy, tmax = g["doy"].values, g["tmax"].values
        p90 = np.empty(365)
        for d in range(1, 366):
            dist = np.abs(((doy - d + 182) % 365) - 182)
            p90[d - 1] = np.nanpercentile(tmax[dist <= P90_HALF_WINDOW], 90)
        out.append(pd.DataFrame({
            "region": region,
            "doy": np.arange(1, 366),
            "normal_tmax": _circular_smooth(m["tmax"].values),
            "normal_tmin": _circular_smooth(m["tmin"].values),
            "normal_p90": _circular_smooth(p90, 15),
        }))
    return pd.concat(out, ignore_index=True)


def attach_normals(df: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doy"] = doy365(df["date"])
    df = df.merge(normals, on=["region", "doy"], how="left")
    df["dep"] = df["tmax"] - df["normal_tmax"]
    return df


def thresholds(normal_tmax, region_type: str | np.ndarray, criteria: dict):
    """Tmax thresholds for heatwave (or worse) and for severe heatwave."""
    normal_tmax = np.asarray(normal_tmax, dtype=float)
    rtype = np.broadcast_to(np.asarray(region_type), normal_tmax.shape)
    thr_hw = np.full(normal_tmax.shape, np.nan)
    thr_sev = np.full(normal_tmax.shape, np.nan)

    p = criteria["plains"]
    m = rtype == "plains"
    thr_hw[m] = np.minimum(np.maximum(p["tmax_min"], normal_tmax[m] + p["hw_dep"]), p["hw_abs"])
    thr_sev[m] = np.minimum(np.maximum(p["tmax_min"], normal_tmax[m] + p["severe_dep"]), p["severe_abs"])

    c = criteria["coastal"]
    m = rtype == "coastal"
    thr_hw[m] = np.maximum(c["tmax_min"], normal_tmax[m] + c["hw_dep"])
    thr_sev[m] = np.maximum(c["tmax_min"], normal_tmax[m] + c["severe_dep"])
    return thr_hw, thr_sev


def label(tmax, thr_hw, thr_sev) -> np.ndarray:
    """0 = normal, 1 = heatwave, 2 = severe heatwave (NaN Tmax gives -1)."""
    tmax = np.asarray(tmax, dtype=float)
    cls = (tmax >= thr_hw).astype(int) + (tmax >= thr_sev).astype(int)
    cls[np.isnan(tmax)] = -1
    return cls


def add_labels(df: pd.DataFrame, region_types: dict, criteria: dict) -> pd.DataFrame:
    df = df.copy()
    df["rtype"] = df["region"].map(region_types).fillna("plains")
    df["thr_hw"], df["thr_sev"] = thresholds(df["normal_tmax"].values, df["rtype"].values, criteria)
    df["cls"] = label(df["tmax"].values, df["thr_hw"].values, df["thr_sev"].values)
    df["thr_p90"] = df["normal_p90"]
    df["hot90"] = (df["tmax"] >= df["thr_p90"]).astype(int)
    # 2-day persistence variant: a heatwave day must touch another heatwave day
    df = df.sort_values(["region", "date"])
    hw = (df["cls"] >= 1)
    g = hw.groupby(df["region"])
    df["hw_2day"] = hw & (g.shift(1, fill_value=False) | g.shift(-1, fill_value=False))
    return df.reset_index(drop=True)
