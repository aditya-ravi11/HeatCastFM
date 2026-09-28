"""Loader for real IoT Automated Weather Station (AWS) readings.

Expected CSV columns (one file per station, any sampling interval):
    timestamp   ISO 8601 local time, e.g. 2025-04-12T14:30:00
    temp_c      air temperature in deg C
    rh_pct      relative humidity in percent

Readings are quality checked, then aggregated to the daily fields used by the
station proxy (tmax, tmin, rh_mean, rh_min), so the output can replace
station_proxy.parquet rows for that region.
"""
from pathlib import Path

import pandas as pd

TEMP_RANGE = (-5.0, 55.0)
RH_RANGE = (1.0, 100.0)
MAX_STEP_C = 8.0          # largest believable change between consecutive readings
MIN_COVERAGE = 0.75       # share of expected readings needed to keep a day


def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df.loc[~df["temp_c"].between(*TEMP_RANGE), "temp_c"] = pd.NA
    df.loc[~df["rh_pct"].between(*RH_RANGE), "rh_pct"] = pd.NA
    jump = df["temp_c"].diff().abs() > MAX_STEP_C
    df.loc[jump, "temp_c"] = pd.NA
    return df


def load_aws_csv(path: str | Path, region: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = quality_check(df)
    step = df["timestamp"].diff().median()
    expected = pd.Timedelta("1D") / step if pd.notna(step) and step > pd.Timedelta(0) else 1
    g = df.set_index("timestamp").resample("1D")
    daily = pd.DataFrame({
        "tmax": g["temp_c"].max(),
        "tmin": g["temp_c"].min(),
        "rh_mean": g["rh_pct"].mean(),
        "rh_min": g["rh_pct"].min(),
        "coverage": g["temp_c"].count() / expected,
    })
    daily.loc[daily["coverage"] < MIN_COVERAGE, ["tmax", "tmin", "rh_mean", "rh_min"]] = pd.NA
    daily = daily.reset_index().rename(columns={"timestamp": "date"})
    daily["region"] = region
    return daily
