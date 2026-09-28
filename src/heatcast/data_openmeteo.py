"""Open-Meteo historical archive (ERA5 based) at exact city points.

Used as a stand-in for the IoT Automated Weather Station (AWS) record and as the
source of the relative humidity covariate. Swap in real AWS data with aws_ingest.py.
"""
import time

import pandas as pd
import requests

from .config import load_config

URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "apparent_temperature_max",
]
RENAME = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "relative_humidity_2m_mean": "rh_mean",
    "relative_humidity_2m_min": "rh_min",
    "apparent_temperature_max": "app_tmax",
}


def fetch_point(lat: float, lon: float, start: str, end: str, retries: int = 70) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(retries):
        resp = requests.get(URL, params=params, timeout=120)
        if resp.status_code == 200:
            d = resp.json()["daily"]
            df = pd.DataFrame(d).rename(columns={"time": "date", **RENAME})
            df["date"] = pd.to_datetime(df["date"])
            return df
        # 429 means the per-minute or per-hour quota is used up
        time.sleep(65 if resp.status_code == 429 else 10 * (attempt + 1))
    resp.raise_for_status()


def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = cfg["paths"]["processed"] / "station_proxy.parquet"
    # 1990 onward covers model context for 2015+, the 1991-2014 bias fit and the humid heat study
    y0, y1 = cfg["station_proxy"]["first"], cfg["years"]["last"]
    cache = cfg["paths"]["raw"] / "openmeteo"
    cache.mkdir(exist_ok=True)
    frames = []
    for r in cfg["regions"]:
        f = cache / f"{r['name']}.parquet"
        if not f.exists():
            parts = []
            for a in range(y0, y1 + 1, 12):  # 12-year chunks keep each request light
                b = min(a + 11, y1)
                parts.append(fetch_point(r["lat"], r["lon"], f"{a}-01-01", f"{b}-12-31"))
                time.sleep(20)
            df = pd.concat(parts, ignore_index=True)
            df["region"] = r["name"]
            df.to_parquet(f, index=False)
        frames.append(pd.read_parquet(f))
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(out, index=False)
    return data
