"""Heat index (NOAA Rothfusz regression) and humid heat days missed by the Tmax rule.

IMD experimental heat index bands: Green <= 35, Yellow 36-45, Orange 46-55, Red > 55 (deg C).
NOAA bands: Extreme caution 32.2-39.4, Danger 39.4-51.1, Extreme danger above 51.1 (deg C).
With RH at the time of Tmax approximated by the daily minimum RH, the proxy data never
reaches IMD Orange, so the NOAA Danger level is used to count humid heat days.
"""
import numpy as np
import pandas as pd

IMD_HI_BINS = [-np.inf, 35.5, 45.5, 55.5, np.inf]
IMD_HI_LABELS = ["Green", "Yellow", "Orange", "Red"]
NOAA_DANGER_C = 39.4  # 103 deg F


def heat_index_c(t_c, rh) -> np.ndarray:
    """NOAA heat index with the Rothfusz regression and its standard adjustments."""
    t = np.asarray(t_c, dtype=float) * 9 / 5 + 32
    rh = np.asarray(rh, dtype=float)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
          - 6.83783e-3 * t ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh
          + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2)
    adj_low = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95.0)) / 17, 0, None))
    hi = np.where((rh < 13) & (t >= 80) & (t <= 112), hi - adj_low, hi)
    adj_high = ((rh - 85) / 10) * ((87 - t) / 5)
    hi = np.where((rh > 85) & (t >= 80) & (t <= 87), hi + adj_high, hi)
    hi = np.where((simple + t) / 2 < 80, simple, hi)
    return (hi - 32) * 5 / 9


def humid_heat_table(point: pd.DataFrame, regions: list[str], years=(1991, 2024)) -> pd.DataFrame:
    """Hot-season days per year by IMD heat index category, and humid heat days missed by the Tmax rule.

    RH at the time of Tmax is approximated by the daily minimum RH.
    """
    p = point[point["region"].isin(regions) & point["date"].dt.year.between(*years)
              & point["date"].dt.month.isin([3, 4, 5, 6])].copy()
    p["hi"] = heat_index_c(p["tmax"], p["rh_min"])
    p["hi_cat"] = pd.cut(p["hi"], IMD_HI_BINS, labels=IMD_HI_LABELS)
    p["tmax_rule_hw"] = p["cls"] >= 1
    p["hi_danger"] = p["hi"] >= NOAA_DANGER_C
    p["humid_missed"] = p["hi_danger"] & ~p["tmax_rule_hw"]
    p["year"] = p["date"].dt.year
    return p
