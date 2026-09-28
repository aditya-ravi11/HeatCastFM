"""Panel assembly (observations, normals, labels) and model input arrays."""
import numpy as np
import pandas as pd

from . import climatology as clim
from .config import load_config


def region_types(cfg: dict) -> dict:
    return {r["name"]: r["type"] for r in cfg["regions"]}


def _fill(df: pd.DataFrame) -> pd.DataFrame:
    """Gap filling for model inputs only: interpolate anomalies up to 5 days, else normal."""
    df = df.sort_values(["region", "date"]).copy()
    for col, ncol in (("tmax", "normal_tmax"), ("tmin", "normal_tmin")):
        anom = (df[col] - df[ncol]).groupby(df["region"]).transform(
            lambda s: s.interpolate(limit=5, limit_area="inside"))
        df[f"{col}_in"] = (df[ncol] + anom.fillna(0.0)).astype("float32")
    return df


def build_panel(cfg: dict | None = None) -> pd.DataFrame:
    """One row per (series, date). Series are the focus regions plus box cells."""
    cfg = cfg or load_config()
    proc = cfg["paths"]["processed"]
    regions = pd.read_parquet(proc / "imd_regions.parquet")
    box = pd.read_parquet(proc / "imd_box.parquet")[["region", "date", "tmax", "tmin"]]
    regions["group"] = "focus"
    box["group"] = "box"
    df = pd.concat([regions, box], ignore_index=True)

    normals = clim.compute_normals(df, cfg["years"]["normal_start"], cfg["years"]["normal_end"])
    normals.to_parquet(proc / "normals.parquet", index=False)
    df = clim.attach_normals(df, normals)
    df = clim.add_labels(df, region_types(cfg), cfg["criteria"])
    df = _fill(df)

    # relative humidity (station proxy) for focus regions, used only by the "+RH" variant
    sp = proc / "station_proxy.parquet"
    if sp.exists():
        rh = pd.read_parquet(sp)[["region", "date", "rh_mean"]]
        df = df.merge(rh, on=["region", "date"], how="left")
        df["rh_in"] = df.groupby("region")["rh_mean"].transform(
            lambda s: s.interpolate(limit_area="inside").bfill().ffill()).astype("float32")
    df.to_parquet(proc / "panel.parquet", index=False)
    return df


def load_panel(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(cfg["paths"]["processed"] / "panel.parquet")


def season_origins(dates: pd.DatetimeIndex, years: list[int], cfg: dict) -> pd.DatetimeIndex:
    (m0, d0), (m1, d1) = cfg["season"]["start"], cfg["season"]["end"]
    out = []
    for y in years:
        out.append(pd.date_range(f"{y}-{m0:02d}-{d0:02d}", f"{y}-{m1:02d}-{d1:02d}", freq="D"))
    idx = pd.DatetimeIndex(np.concatenate(out))
    return idx[idx.isin(dates)]


class SeriesStore:
    """Wide arrays (series x days) for fast slicing during backtests."""

    def __init__(self, panel: pd.DataFrame):
        self.dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        self.names = list(dict.fromkeys(panel.sort_values(["group", "region"], ascending=[False, True])["region"]))
        self.group = panel.drop_duplicates("region").set_index("region")["group"].reindex(self.names).values
        self.rtype = panel.drop_duplicates("region").set_index("region")["rtype"].reindex(self.names).values

        def wide(col):
            if col not in panel:
                return None
            return (panel.pivot(index="region", columns="date", values=col)
                    .reindex(index=self.names, columns=self.dates).values.astype("float32"))

        self.tmax_in = wide("tmax_in")
        self.tmin_in = wide("tmin_in")
        self.tmax_obs = wide("tmax")
        self.normal = wide("normal_tmax")
        self.normal_tmin = wide("normal_tmin")
        self.thr_hw = wide("thr_hw")
        self.thr_sev = wide("thr_sev")
        self.thr_p90 = wide("thr_p90")
        self.cls = wide("cls")
        self.rh_in = wide("rh_in")
        doy = clim.doy365(self.dates)
        self.doy_sin = np.sin(2 * np.pi * doy / 365.0).astype("float32")
        self.doy_cos = np.cos(2 * np.pi * doy / 365.0).astype("float32")
        self.pos = {d: i for i, d in enumerate(self.dates)}

    def index(self, name: str) -> int:
        return self.names.index(name)
