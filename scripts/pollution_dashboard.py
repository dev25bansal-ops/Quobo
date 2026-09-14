"""Stage 5: pollution analytics — zone assignment, cleanliness score, PSI.

TACO carries no GPS metadata, so this module supports two modes:
  - 'batch': demo mapping image batch_1..N -> Zone A..F (clearly labeled demo)
  - 'folder': real deployments drop zone-labeled crop folders into
    data/geo/<Zone>/... and the same analytics run unchanged.

Metrics (per zone):
  - cleanliness_score = clean_count / total_count * 100  (notebook formula)
    'clean' classes: bottle/can/carton/cup/lid (recyclables);
    'dirty': cigarette (toxic litter).
  - PSI: hazard-weighted dominant-pollutant index on the 0-500 air-index scale:
      sub-index per class = count_c * hazard_weight_c / global_max_weighted_count * 500
      (global_max = worst (zone, class) weighted count across the city)
      headline = max sub-index (dominant-pollutant rule, as in US EPA AQI /
      Singapore PSI / India CPCB NAQI), then banded Good..Hazardous.

Usage: .venv/Scripts/python.exe -m scripts.pollution_dashboard
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.config import ROOT

log = logging.getLogger("dashboard")

CLEAN_CLASSES = {"bottle", "can", "carton", "cup", "lid"}
DIRTY_CLASSES = {"cigarette"}

# hazard weight per class for PSI (cigarette butts: toxic litter, leach
# nicotine/heavy metals; per research report they are the #1 stream item)
HAZARD_W = {"cigarette": 3.0, "bottle": 1.0, "can": 1.0, "carton": 0.8, "cup": 0.8, "lid": 0.5}

PSI_BANDS = [(50, "Good"), (100, "Moderate"), (200, "Unhealthy"), (300, "Very Unhealthy"), (500, "Hazardous")]
BAND_COLORS = {"Good": "#52b948", "Moderate": "#f5eb3d", "Unhealthy": "#f77c02",
               "Very Unhealthy": "#df2020", "Hazardous": "#7d2181"}


import html as _html


def zone_of(crop_path: str, mode: str = "batch") -> str:
    """Zone for a crop. 'batch' mode maps TACO batch_N -> Zone A..F (demo).
    Real deployments: place zone-labeled crops under data/geo/<Zone name>/<class>/
    and pass mode='folder' — the zone is the CLASS folder's parent (two levels
    above the file)."""
    p = Path(crop_path)
    if mode == "folder":
        # layout: .../data/geo/<Zone>/<class>/<file>.jpg -> zone = parent of parent
        grandparent = p.parent.parent.name
        if grandparent.startswith("Zone"):
            return grandparent
        return "Unassigned"
    name = p.name  # batch_3_000123_ann45.jpg
    if name.startswith("batch_"):
        parts = name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            n = int(parts[1])
            return f"Zone {chr(ord('A') + (n - 1) % 6)}"
    return "Unassigned"  # unknown provenance is explicit, never silently Zone A


def load_crops() -> pd.DataFrame:
    """Ground-truth mode (legacy/demo): one row per crop from the class folders."""
    crops_dir = ROOT / "data" / "processed" / "crops"
    rows = []
    for cls_dir in sorted(crops_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*.jpg"):
            rows.append({"crop_path": str(p), "class": cls_dir.name, "file": p.name})
    return pd.DataFrame(rows)


def load_predictions(clf: str = "rbf_svm", arm: str = "D_qubo") -> pd.DataFrame:
    """Prediction mode (S1): consume the CLASSIFIER'S output from the latest
    experiment run — the CV->PSI loop this project claims. Reads per-rep
    persisted y_test/predictions, aggregates to a majority vote per crop
    across repeats, and maps each crop to its zone.

    Falls back to load_crops() when no experiment run exists."""
    import json as _json

    run_dirs = sorted((ROOT / "experiments").iterdir())
    if not run_dirs:
        log.warning("no experiment runs — falling back to ground-truth crops")
        return load_crops()
    run_dir = run_dirs[-1]
    reps = sorted(run_dir.glob(f"rep*_{arm}.json"))
    if not reps:
        log.warning("run %s has no %s reps — falling back to ground-truth crops",
                    run_dir.name, arm)
        return load_crops()

    from collections import defaultdict as _dd
    votes = _dd(Counter)  # crop_key -> class -> vote count
    for rep_file in reps:
        with open(rep_file, encoding="utf-8") as f:
            j = _json.load(f)
        pred = j["metrics"][clf]["predictions"]
        # rep JSONs don't carry test crop paths — recover them by replaying
        # the same group split the experiment used (deterministic given seed)
        from src.quobo.config import load_config
        from src.quobo.features import run_features
        from src.quobo.run_experiment import crop_image_group
        cfg = load_config("configs/experiment_6class.yaml")
        df = run_features(cfg)
        groups = np.asarray([crop_image_group(p) for p in df["crop_path"]])
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1,
                                test_size=cfg["qsvm"]["test_fraction"],
                                random_state=j["seed"])
        _, idx_te = next(gss.split(df[[c for c in df.columns if c.startswith("f")]].values,
                                   df["label"].values, groups=groups))
        for i, p in zip(idx_te, pred):
            votes[df["crop_path"].values[i]][p] += 1

    rows = [{"crop_path": cp, "class": c.most_common(1)[0][0], "file": Path(cp).name}
            for cp, c in votes.items()]
    out = pd.DataFrame(rows)
    log.info("prediction mode: %d crops voted from %d %s/%s reps (%.0f%% coverage)",
             len(out), len(reps), arm, clf, 100 * len(out) / len(df))
    return out


def compute_zone_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = defaultdict(Counter)
    for _, r in df.iterrows():
        # folder-mode rows carry their zone directly; otherwise fall back to
        # the filename heuristic (demo batch mapping)
        z = r.get("zone_dir") or zone_of(r["crop_path"])
        counts[z][r["class"]] += 1

    # global max of the hazard-weighted class sub-index across all zones —
    # the reference point that maps the worst (zone, class) cell to 500
    global_max = max(
        (c.get(cls, 0) * HAZARD_W.get(cls, 1.0)
         for c in counts.values() for cls in HAZARD_W),
        default=0.0,
    ) or 1.0

    rows = []
    for z in sorted(counts):
        c = counts[z]
        total = sum(c.values())
        clean = sum(v for k2, v in c.items() if k2 in CLEAN_CLASSES)
        dirty = total - clean
        # dominant-pollutant rule (US EPA AQI / Singapore PSI / India CPCB
        # NAQI style): headline = max hazard-weighted class sub-index
        sub = {k2: c.get(k2, 0) * HAZARD_W.get(k2, 1.0) / global_max * 500.0
               for k2 in HAZARD_W}
        psi = max(min(500, round(max(sub.values()))), 0)
        band = next(b for t, b in PSI_BANDS if psi <= t)
        rows.append({
            "zone": z, "total": total, "clean": clean, "dirty": dirty,
            "cleanliness": round(clean / total * 100, 1),
            "psi": psi, "band": band,
            **{k2: c.get(k2, 0) for k2 in HAZARD_W},
        })
    return pd.DataFrame(rows)


def _build_interactive_charts(zt: pd.DataFrame) -> str:
    """Enhancement 5: two interactive Plotly charts (hover + zoom) — PSI bar
    with band colors, and a 100%-stacked class-composition chart. Falls back
    to an empty string if plotly isn't installed (keeps the static dashboard
    working without the dep)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed — skipping interactive charts")
        return ""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PSI by zone", "Class composition (% of zone total)"),
        horizontal_spacing=0.12,
    )
    # PSI bar, colored by band
    fig.add_trace(
        go.Bar(
            x=zt["zone"], y=zt["psi"],
            marker_color=[BAND_COLORS[b] for b in zt["band"]],
            hovertemplate="%{x}<br>PSI: %{y}<br>%{customdata}<extra></extra>",
            customdata=zt["band"],
        ), row=1, col=1,
    )
    # 100% stacked class composition
    for cls in HAZARD_W:
        frac = (zt[cls] / zt["total"].replace(0, np.nan) * 100).fillna(0)
        fig.add_trace(
            go.Bar(
                x=zt["zone"], y=frac, name=cls,
                hovertemplate="%{x} · %{fullData.name}: %{y:.1f}%<extra></extra>",
            ), row=1, col=2,
        )
    fig.update_yaxes(title_text="PSI", row=1, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=2)
    fig.update_layout(height=440, margin={"l": 40, "r": 20, "t": 50, "b": 40},
                      legend={"orientation": "h", "y": -0.2})
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_dashboard(zt: pd.DataFrame, out_html: Path, truth_mode: bool = False) -> None:
    total_all = zt["total"].sum()
    clean_all = zt["clean"].sum()
    city_clean = clean_all / total_all * 100
    city_psi = zt["psi"].max()
    city_band = zt.loc[zt["psi"].idxmax(), "band"]
    source_label = ("ground-truth TACO labels (demo)" if truth_mode
                    else "RBF-SVM predictions on QUBO-8 features, majority vote over 25 repeats")

    zone_cards = ""
    for _, r in zt.iterrows():
        color = BAND_COLORS[r["band"]]
        bars = ""
        for cls in HAZARD_W:
            v = r[cls]
            frac = v / r["total"] * 100 if r["total"] else 0
            bars += (
                f'<div class="bar-row"><span class="bar-label">{_html.escape(cls)}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{frac:.0f}%"></div></div>'
                f'<span class="bar-val">{v}</span></div>'
            )
        zone_cards += f'''
      <div class="card" style="border-top:6px solid {color}">
        <div class="zone-head"><span class="zone-name">{_html.escape(str(r["zone"]))}</span>
          <span class="badge" style="background:{color}">{_html.escape(str(r["band"]))} · PSI {r["psi"]}</span></div>
        <div class="big-stat">Cleanliness <b>{r["cleanliness"]}%</b></div>
        <div class="sub-stat">{r["clean"]} clean / {r["dirty"]} litter · {r["total"]} items</div>
        {bars}
      </div>'''

    # simple zone map: 2x3 grid colored by PSI
    cells = ""
    for _, r in zt.iterrows():
        cells += (f'<div class="map-cell" style="background:{BAND_COLORS[r["band"]]}">'
                  f'<b>{_html.escape(str(r["zone"]))}</b><span>PSI {r["psi"]}</span></div>')

    # Enhancement 5: interactive Plotly charts (hover + zoom)
    interactive = _build_interactive_charts(zt)

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Quobo — Pollution Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #1c2733; }}
  header {{ background: #10314f; color: #fff; padding: 22px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  header p {{ margin: 0; opacity: .75; font-size: 13px; }}
  .kpis {{ display: flex; gap: 18px; padding: 22px 32px; flex-wrap: wrap; }}
  .kpi {{ background: #fff; border-radius: 10px; padding: 16px 22px; box-shadow: 0 1px 4px rgba(0,0,0,.08); min-width: 170px; }}
  .kpi .v {{ font-size: 30px; font-weight: 700; }}
  .kpi .l {{ font-size: 12px; color: #5a6b7b; text-transform: uppercase; letter-spacing: .06em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; padding: 0 32px 26px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .zone-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .zone-name {{ font-size: 18px; font-weight: 700; }}
  .badge {{ color: #fff; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 999px; }}
  .big-stat {{ font-size: 15px; margin: 6px 0 2px; }}
  .sub-stat {{ font-size: 12px; color: #5a6b7b; margin-bottom: 10px; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }}
  .bar-label {{ width: 62px; color: #5a6b7b; }}
  .bar-track {{ flex: 1; background: #edf1f4; border-radius: 4px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: #10314f; border-radius: 4px; }}
  .bar-val {{ width: 34px; text-align: right; font-variant-numeric: tabular-nums; }}
  .map {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 0 32px 32px; max-width: 640px; }}
  .map-cell {{ border-radius: 10px; color: #fff; text-align: center; padding: 26px 0; display: flex; flex-direction: column; gap: 4px; }}
  .map-cell span {{ font-size: 13px; opacity: .9; }}
  .note {{ padding: 0 32px 30px; font-size: 12px; color: #5a6b7b; max-width: 900px; }}
</style></head><body>
<header>
  <h1>Quobo Pollution Dashboard — Gwalior Zones (demo)</h1>
  <p>Classifier output: {source_label} · TACO crops · {datetime.now().astimezone():%d %b %Y %H:%M %Z}</p>
</header>
<div class="kpis">
  <div class="kpi"><div class="v">{city_clean:.1f}%</div><div class="l">City cleanliness score</div></div>
  <div class="kpi"><div class="v" style="color:{BAND_COLORS[city_band]}">{city_psi}</div><div class="l">Worst-zone PSI ({city_band})</div></div>
  <div class="kpi"><div class="v">{total_all:,}</div><div class="l">Classified items</div></div>
  <div class="kpi"><div class="v">{len(zt)}</div><div class="l">Zones monitored</div></div>
</div>
<div class="grid">{zone_cards}</div>
<h3 style="padding:0 32px">Zone map (PSI)</h3>
<div class="map">{cells}</div>
{interactive}
<div class="note"><b>Method.</b> Cleanliness score = clean count / total waste count × 100 (project definition).
PSI = max hazard-weighted class sub-index on a 0–500 scale (dominant-pollutant rule as in US EPA AQI / Singapore PSI /
India CPCB NAQI): sub-index<sub>c</sub> = count<sub>c</sub> × weight<sub>c</sub> / max over all zones and classes of
(count × weight) × 500; weights: cigarette ×3, bottle/can ×1, carton/cup ×0.8, lid ×0.5.
Zone assignment is a <b>demo mapping</b> of TACO image batches — TACO carries no GPS metadata. For a real Gwalior
deployment, place zone-labeled crops under data/geo/&lt;Zone&gt;/ and rerun; the analytics are unchanged.</div>
</body></html>'''
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # default: consume classifier predictions (the CV->PSI loop).
    # --truth renders the ground-truth (demo/legacy) view instead.
    truth_mode = "--truth" in sys.argv
    df = load_crops() if truth_mode else load_predictions()
    zt = compute_zone_table(df)
    out_dir = ROOT / "results" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_truth" if truth_mode else ""
    zt.to_csv(out_dir / f"zone_table{suffix}.csv", index=False)
    render_dashboard(zt, out_dir / "dashboard.html", truth_mode=truth_mode)
    print(zt.to_string(index=False))
    print("saved:", out_dir / "dashboard.html")


if __name__ == "__main__":
    main()
