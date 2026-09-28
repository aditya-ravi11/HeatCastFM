"""IMD 1 degree gridded Tmax/Tmin (Srivastava et al., 2009) via imdlib.

Produces two parquet files:
  imd_regions.parquet  daily Tmax/Tmin for the nearest valid cell of each focus region
  imd_box.parquet      daily Tmax/Tmin for every valid cell inside the pooled box
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import load_config

MISSING_THRESHOLD = 90.0  # IMD uses 99.9 as the missing value for temperature


def download(cfg: dict | None = None, retries: int = 6, variables=("tmax", "tmin"), years=None) -> None:
    """Year by year download with retries; years already on disk are skipped."""
    import socket
    import time

    import imdlib as imd

    socket.setdefaulttimeout(90)  # imdlib sets no timeout, and the IMD server can stall forever
    cfg = cfg or load_config()
    raw = cfg["paths"]["raw"]
    y0, y1 = years or (cfg["years"]["first"], cfg["years"]["last"])
    for var in variables:
        for year in range(y0, y1 + 1):
            f = raw / var / f"{year}.GRD"
            if f.exists() and f.stat().st_size > 1_000_000:
                continue
            for attempt in range(retries):
                try:
                    imd.get_data(var, year, year, fn_format="yearwise", file_dir=str(raw))
                    break
                except Exception as exc:  # server often times out
                    print(f"{var} {year}: attempt {attempt + 1} failed ({exc.__class__.__name__})", flush=True)
                    time.sleep(15 * (attempt + 1))


def open_var(var: str, cfg: dict | None = None) -> xr.DataArray:
    import imdlib as imd

    cfg = cfg or load_config()
    y0, y1 = cfg["years"]["first"], cfg["years"]["last"]
    data = imd.open_data(var, y0, y1, "yearwise", str(cfg["paths"]["raw"]))
    da = data.get_xarray()[var]
    return da.where(da < MISSING_THRESHOLD)


def _valid_fraction(da: xr.DataArray) -> xr.DataArray:
    return da.notnull().mean("time")


def nearest_valid_cell(valid: xr.DataArray, lat: float, lon: float, min_frac: float = 0.95):
    """Nearest cell centre (lat, lon) that has at least min_frac non-missing days."""
    lats, lons = np.meshgrid(valid["lat"].values, valid["lon"].values, indexing="ij")
    ok = valid.transpose("lat", "lon").values >= min_frac
    dist = np.hypot(lats - lat, (lons - lon) * np.cos(np.radians(lat)))
    dist[~ok] = np.inf
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(lats[i, j]), float(lons[i, j]), float(dist[i, j])


def build(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    tmax = open_var("tmax", cfg)
    tmin = open_var("tmin", cfg)
    valid = _valid_fraction(tmax)

    rows, cells = [], []
    for r in cfg["regions"]:
        clat, clon, dist = nearest_valid_cell(valid, r["lat"], r["lon"])
        cells.append({**r, "cell_lat": clat, "cell_lon": clon, "cell_dist_deg": round(dist, 3)})
        df = pd.DataFrame({
            "date": pd.to_datetime(tmax["time"].values),
            "tmax": tmax.sel(lat=clat, lon=clon).values,
            "tmin": tmin.sel(lat=clat, lon=clon).values,
        })
        df["region"] = r["name"]
        rows.append(df)
    regions = pd.concat(rows, ignore_index=True)

    b = cfg["box"]
    sub_max = tmax.sel(lat=slice(*b["lat"]), lon=slice(*b["lon"]))
    sub_min = tmin.sel(lat=slice(*b["lat"]), lon=slice(*b["lon"]))
    box = (
        xr.Dataset({"tmax": sub_max, "tmin": sub_min})
        .to_dataframe()
        .reset_index()
        .rename(columns={"time": "date"})
    )
    frac = box.groupby(["lat", "lon"])["tmax"].transform(lambda s: s.notna().mean())
    box = box[frac >= 0.95].copy()
    box["region"] = "cell_" + box["lat"].map("{:.1f}".format) + "_" + box["lon"].map("{:.1f}".format)

    out = cfg["paths"]["processed"]
    regions.to_parquet(out / "imd_regions.parquet", index=False)
    box.to_parquet(out / "imd_box.parquet", index=False)
    pd.DataFrame(cells).to_csv(out / "region_cells.csv", index=False)
    return regions, box
