"""Extract region cells, compute normals and labels, write panel.parquet."""
from heatcast import data_imd
from heatcast.config import load_config
from heatcast.features import build_panel

if __name__ == "__main__":
    cfg = load_config()
    regions, box = data_imd.build(cfg)
    print("focus regions:", regions["region"].nunique(), " box cells:", box["region"].nunique())
    panel = build_panel(cfg)
    hot = panel[panel["date"].dt.month.isin([3, 4, 5, 6])]
    print(hot.groupby("group")["cls"].value_counts().unstack())
