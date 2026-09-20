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

Usage (prediction mode consumes a completed, manifest-verified run):
  .venv/Scripts/python.exe -m scripts.pollution_dashboard
  .venv/Scripts/python.exe -m scripts.pollution_dashboard --mode demo-ground-truth
"""

from __future__ import annotations

import argparse
import hashlib
import html as _html
import json
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

MANIFEST_NAME = "run_manifest.json"


class ProvenanceError(RuntimeError):
    """A run cannot be trusted: no/again-incomplete manifest, missing sample IDs,
    misaligned predictions, or an artifact that does not match its hash."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


CLEAN_CLASSES = {"bottle", "can", "carton", "cup", "lid"}
DIRTY_CLASSES = {"cigarette"}

# hazard weight per class for PSI (cigarette butts: toxic litter, leach
# nicotine/heavy metals; per research report they are the #1 stream item)
HAZARD_W = {"cigarette": 3.0, "bottle": 1.0, "can": 1.0, "carton": 0.8, "cup": 0.8, "lid": 0.5}

PSI_BANDS = [
    (50, "Good"),
    (100, "Moderate"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
]
BAND_COLORS = {
    "Good": "#52b948",
    "Moderate": "#f5eb3d",
    "Unhealthy": "#f77c02",
    "Very Unhealthy": "#df2020",
    "Hazardous": "#7d2181",
}


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
    """Explicit demo/ground-truth mode: one row per crop from the class folders.

    This is only ever entered through ``--mode demo-ground-truth``; it is never a
    silent fallback for a failed prediction run (Q05)."""
    crops_dir = ROOT / "data" / "processed" / "crops"
    if not crops_dir.is_dir():
        raise ProvenanceError(
            f"ground-truth crops directory not found: {crops_dir}; "
            "run data prep or pass --mode predictions"
        )
    rows = []
    for cls_dir in sorted(crops_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*.jpg"):
            rows.append({"crop_path": str(p), "class": cls_dir.name, "file": p.name})
    return pd.DataFrame(rows, columns=["crop_path", "class", "file"])


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        raise ProvenanceError(
            f"run {run_dir.name!r} has no {MANIFEST_NAME}; refusing to guess "
            "provenance (re-run the experiment to write one)"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"run {run_dir.name!r} has a corrupt {MANIFEST_NAME}") from exc


def find_complete_run(run_id: str | None = None) -> tuple[Path, dict]:
    """Locate a run whose manifest says ``complete``. There is deliberately no
    fallback: an unfinished or manifest-less run is an error, not a demo."""
    experiments = ROOT / "experiments"
    if run_id is not None:
        run_dir = experiments / run_id
        if not run_dir.is_dir():
            raise ProvenanceError(f"experiment run {run_id!r} not found under {experiments}")
        manifest = _read_manifest(run_dir)
        status = manifest.get("status")
        if status != "complete":
            raise ProvenanceError(
                f"run {run_id!r} status is {status!r}, not 'complete'; "
                "refusing incomplete artifacts"
            )
        return run_dir, manifest
    if not experiments.is_dir():
        raise ProvenanceError(
            f"no experiments directory at {experiments}; run the experiment first "
            "(ground-truth labels are only available via --mode demo-ground-truth)"
        )
    complete = []
    for d in sorted(experiments.iterdir()):
        if not d.is_dir() or not (d / MANIFEST_NAME).is_file():
            continue
        try:
            manifest = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            complete.append((d.name, d, manifest))
    if not complete:
        raise ProvenanceError(
            "no completed experiment run with a run manifest found; refusing to "
            "substitute ground-truth labels"
        )
    _, run_dir, manifest = complete[-1]
    return run_dir, manifest


def load_predictions(
    clf: str = "rbf_svm", arm: str = "D_qubo", run_id: str | None = None
) -> pd.DataFrame:
    """Prediction mode (S1/Q04): consume a COMPLETED run's persisted predictions
    together with the exact held-out crop IDs saved alongside them. No split is
    replayed, no config is re-read, and there is no ground-truth fallback."""
    run_dir, manifest = find_complete_run(run_id)
    arms = manifest.get("arms") or []
    if arm not in arms:
        raise ProvenanceError(f"run {run_dir.name!r} did not run arm {arm!r} (arms={arms})")
    reps_completed = manifest.get("reps_completed")
    if not isinstance(reps_completed, int) or reps_completed < 1:
        raise ProvenanceError(f"run {run_dir.name!r} manifest records no completed repeats")
    artifacts = manifest.get("artifacts") or {}

    votes = defaultdict(Counter)  # crop id -> predicted class -> vote count
    truth = {}
    for rep in range(reps_completed):
        name = f"rep{rep}_{arm}.json"
        path = run_dir / name
        if not path.is_file():
            raise ProvenanceError(
                f"run {run_dir.name!r} is missing artifact {name}; refusing incomplete run"
            )
        recorded = artifacts.get(name)
        if not recorded or _sha256_file(path) != recorded.get("sha256"):
            raise ProvenanceError(
                f"artifact {name} does not match the run manifest (corrupt or substituted)"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "test_ids" not in payload:
            raise ProvenanceError(
                f"{name} has no persisted test_ids (legacy artifact); predictions "
                "cannot be mapped to crops without replaying the split — re-run the experiment"
            )
        test_ids = payload.get("test_ids") or []
        if "y_test" not in payload:
            raise ProvenanceError(f"{name} has no y_test labels")
        y_test = payload.get("y_test") or []
        try:
            pred = payload["metrics"][clf]["predictions"]
        except (KeyError, TypeError) as exc:
            raise ProvenanceError(f"{name} has no predictions for classifier {clf!r}") from exc
        if not (len(test_ids) == len(y_test) == len(pred)):
            raise ProvenanceError(
                f"{name} is misaligned: {len(test_ids)} test_ids, "
                f"{len(y_test)} y_test, {len(pred)} predictions"
            )
        if len(set(test_ids)) != len(test_ids):
            raise ProvenanceError(f"{name} contains duplicate test_ids")
        for crop_id, true_label, predicted in zip(test_ids, y_test, pred):
            if crop_id in truth and truth[crop_id] != true_label:
                raise ProvenanceError(
                    f"crop {crop_id!r} has inconsistent true labels across repeats"
                )
            truth[crop_id] = true_label
            votes[crop_id][predicted] += 1

    if not votes:
        raise ValueError(f"run {run_dir.name!r} produced no held-out predictions (empty dataset)")

    rows = [
        {
            "crop_path": crop_id,
            "class": counter.most_common(1)[0][0],
            "file": Path(crop_id).name,
            "votes": dict(counter),
            "truth": truth.get(crop_id),
        }
        for crop_id, counter in votes.items()
    ]
    out = pd.DataFrame(rows)
    out.attrs.update(
        {
            "mode": "predictions",
            "run_id": run_dir.name,
            "arm": arm,
            "clf": clf,
            "reps": reps_completed,
        }
    )
    log.info(
        "prediction mode: %d crops voted from %d %s/%s reps in run %s",
        len(out),
        reps_completed,
        arm,
        clf,
        run_dir.name,
    )
    return out


def compute_zone_table(df: pd.DataFrame) -> pd.DataFrame:
    import math
    from statistics import NormalDist

    from src.quobo.uncertainty import vote_agreement, wilson_interval

    columns = [
        "zone",
        "total",
        "clean",
        "dirty",
        "cleanliness",
        "psi",
        "band",
        "psi_ci",
        "psi_lo",
        "psi_hi",
        "dirty_rate_lo",
        "dirty_rate_hi",
        "vote_agreement",
        *HAZARD_W,
    ]
    counts = defaultdict(Counter)
    zone_votes = defaultdict(list)
    for _, r in df.iterrows():
        # folder-mode rows carry their zone directly; otherwise fall back to
        # the filename heuristic (demo batch mapping)
        z = r.get("zone_dir") or zone_of(r["crop_path"])
        if r["class"] not in HAZARD_W:
            raise ValueError(f"unsupported class: {r['class']!r}")
        counts[z][r["class"]] += 1
        if isinstance(r.get("votes"), dict) and r["votes"]:
            zone_votes[z].append(r["votes"])

    # global max of the hazard-weighted class sub-index across all zones —
    # the reference point that maps the worst (zone, class) cell to 500
    global_max = (
        max(
            (c.get(cls, 0) * HAZARD_W.get(cls, 1.0) for c in counts.values() for cls in HAZARD_W),
            default=0.0,
        )
        or 1.0
    )

    rows = []
    for z in sorted(counts):
        c = counts[z]
        total = sum(c.values())
        clean = sum(v for k2, v in c.items() if k2 in CLEAN_CLASSES)
        dirty = total - clean
        # dominant-pollutant rule (US EPA AQI / Singapore PSI / India CPCB
        # NAQI style): headline = max hazard-weighted class sub-index
        sub = {k2: c.get(k2, 0) * HAZARD_W.get(k2, 1.0) / global_max * 500.0 for k2 in HAZARD_W}
        psi = max(min(500, round(max(sub.values()))), 0)
        band = next(b for t, b in PSI_BANDS if psi <= t)

        dirty_lo, dirty_hi = wilson_interval(dirty, total)
        # Conditional on the observed normalization reference. Bonferroni adjusts
        # the six marginal Wilson bounds; it does not calibrate classifier errors
        # or correct within-photo clustering of crops.
        z_score = NormalDist().inv_cdf(1 - 0.05 / (2 * len(HAZARD_W)))
        bounds = {cls: wilson_interval(c.get(cls, 0), total, z_score) for cls in HAZARD_W}
        lo = max(
            total * bounds[cls][0] * weight / global_max * 500 for cls, weight in HAZARD_W.items()
        )
        hi = max(
            total * bounds[cls][1] * weight / global_max * 500 for cls, weight in HAZARD_W.items()
        )
        psi_lo = max(0, min(500, math.floor(lo)))
        psi_hi = max(0, min(500, math.ceil(hi)))
        agreement = vote_agreement(zone_votes.get(z, [])) if zone_votes.get(z) else None

        rows.append(
            {
                "zone": z,
                "total": total,
                "clean": clean,
                "dirty": dirty,
                "cleanliness": round(clean / total * 100, 1),
                "psi": psi,
                "band": band,
                "psi_ci": f"[{psi_lo}-{psi_hi}]",
                "psi_lo": psi_lo,
                "psi_hi": psi_hi,
                "dirty_rate_lo": round(dirty_lo, 3),
                "dirty_rate_hi": round(dirty_hi, 3),
                "vote_agreement": round(agreement, 3) if agreement is not None else None,
                **{k2: c.get(k2, 0) for k2 in HAZARD_W},
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
        rows=1,
        cols=2,
        subplot_titles=(
            "Relative PSI by zone (project index, unvalidated)",
            "Class composition (% of zone total)",
        ),
        horizontal_spacing=0.12,
    )
    # PSI bar, colored by band
    fig.add_trace(
        go.Bar(
            x=zt["zone"],
            y=zt["psi"],
            marker_color=[BAND_COLORS[b] for b in zt["band"]],
            hovertemplate="%{x}<br>Relative PSI: %{y}<br>%{customdata}<extra></extra>",
            customdata=zt["band"],
        ),
        row=1,
        col=1,
    )
    # 100% stacked class composition
    for cls in HAZARD_W:
        frac = (zt[cls] / zt["total"].replace(0, np.nan) * 100).fillna(0)
        fig.add_trace(
            go.Bar(
                x=zt["zone"],
                y=frac,
                name=cls,
                hovertemplate="%{x} · %{fullData.name}: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Relative PSI (unvalidated, non-health)", row=1, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=2)
    fig.update_layout(
        height=440,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _source_label(truth_mode: bool, provenance: dict | None) -> str:
    if truth_mode:
        return "ground-truth TACO labels (demo only — not classifier output)"
    provenance = provenance or {}
    if provenance.get("mode") == "predictions":
        return (
            f"classifier predictions — run {provenance.get('run_id')}, "
            f"arm {provenance.get('arm')}, {provenance.get('clf')}, "
            f"majority vote over {provenance.get('reps')} repeats"
        )
    return "classifier predictions (completed experiment run)"


def _render_empty_dashboard(out_html: Path, truth_mode: bool, provenance: dict | None) -> None:
    """Explicit empty state — no observations is a valid outcome, not a crash
    (Q05: empty zone rendering previously called idxmax() on an empty frame)."""
    label = _source_label(truth_mode, provenance)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Quobo — Pollution Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #1c2733; }}
  header {{ background: #10314f; color: #fff; padding: 22px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  header p {{ margin: 0; opacity: .75; font-size: 13px; }}
  .empty {{ margin: 32px; padding: 32px; background: #fff; border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); max-width: 720px; }}
  .empty h2 {{ margin: 0 0 8px; font-size: 20px; }}
  .empty p {{ margin: 4px 0; color: #5a6b7b; font-size: 14px; }}
</style></head><body>
<header>
  <h1>Quobo Pollution Dashboard — Gwalior Zones (demo)</h1>
  <p>Source: {_html.escape(label)} · TACO crops · {datetime.now().astimezone():%d %b %Y %H:%M %Z}</p>
</header>
<div class="empty">
  <h2>No observations</h2>
  <p>No classified items were available for this mode, so no zone metrics were computed.</p>
  <p>Source: {_html.escape(label)}</p>
</div>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")


def render_dashboard(
    zt: pd.DataFrame, out_html: Path, truth_mode: bool = False, provenance: dict | None = None
) -> None:
    if zt is None or zt.empty or ("total" in zt.columns and int(zt["total"].sum()) == 0):
        _render_empty_dashboard(out_html, truth_mode, provenance)
        return
    total_all = zt["total"].sum()
    clean_all = zt["clean"].sum()
    city_clean = clean_all / total_all * 100
    city_psi = zt["psi"].max()
    city_band = zt.loc[zt["psi"].idxmax(), "band"]
    source_label = _source_label(truth_mode, provenance)

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
        zone_cards += f"""
      <div class="card" style="border-top:6px solid {color}">
        <div class="zone-head"><span class="zone-name">{_html.escape(str(r["zone"]))}</span>
          <span class="badge" style="background:{color}">{_html.escape(str(r["band"]))} · relative PSI {r["psi"]} {_html.escape(str(r["psi_ci"]))}</span></div>
        <div class="big-stat">Cleanliness <b>{r["cleanliness"]}%</b></div>
        <div class="sub-stat">{r["clean"]} clean / {r["dirty"]} litter · {r["total"]} items{"" if r.get("vote_agreement") is None else f" · vote agreement {r['vote_agreement']:.0%}"}</div>
        {bars}
      </div>"""

    # simple zone map: 2x3 grid colored by PSI
    cells = ""
    for _, r in zt.iterrows():
        cells += (
            f'<div class="map-cell" style="background:{BAND_COLORS[r["band"]]}">'
            f'<b>{_html.escape(str(r["zone"]))}</b><span>PSI {r["psi"]}</span></div>'
        )

    # Enhancement 5: interactive Plotly charts (hover + zoom)
    interactive = _build_interactive_charts(zt)

    html = f"""<!DOCTYPE html>
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
  <p><b>PSI is a project-defined relative index — unvalidated and not a health or air-quality standard.</b></p>
</header>
<div class="kpis">
  <div class="kpi"><div class="v">{city_clean:.1f}%</div><div class="l">City cleanliness score</div></div>
  <div class="kpi"><div class="v" style="color:{BAND_COLORS[city_band]}">{city_psi}</div><div class="l">Worst-zone PSI ({city_band})</div></div>
  <div class="kpi"><div class="v">{total_all:,}</div><div class="l">Classified items</div></div>
  <div class="kpi"><div class="v">{len(zt)}</div><div class="l">Zones monitored</div></div>
</div>
<div class="grid">{zone_cards}</div>
<h3 style="padding:0 32px">Zone map (relative PSI index)</h3>
<div class="map">{cells}</div>
{interactive}
<div class="note"><b>Method.</b> Cleanliness score = clean count / total waste count × 100 (project definition).
<b>PSI is a project-defined relative index, not a health index.</b> It is a hazard-weighted dominant-class
score rescaled to 0–500 against the single worst (zone, class) cell observed in this dataset — a
dataset-relative normalization, not a measured pollutant concentration:
sub-index<sub>c</sub> = count<sub>c</sub> × weight<sub>c</sub> / max over all zones and classes of
(count × weight) × 500; weights: cigarette ×3, bottle/can ×1, carton/cup ×0.8, lid ×0.5.
The 0–500 scale and band names are borrowed for readability from air-quality indices (US EPA AQI /
Singapore PSI / India CPCB NAQI), but this index is <b>not validated</b> against any health outcome and is
<b>not comparable</b> to those indices or to any environmental or health threshold; treat the values as an
internal, dataset-relative ranking only.
PSI ranges are <b>conditional sampling bounds</b>: six Bonferroni-adjusted marginal Wilson
intervals (nominal 95% joint level) are mapped through the maximum weighted class index,
keeping the observed normalization reference fixed. These approximate bounds assume independent
crops and do not account for same-photo clustering or reference-estimation uncertainty.
They are not calibrated classifier-error intervals or validated environmental-health thresholds.
Bands use the point estimate. "Vote agreement" is repeat-to-repeat prediction stability,
not a calibrated probability of correctness. The displayed bounds do not use the vote distribution.
Zone assignment is a <b>demo mapping</b> of TACO image batches — TACO carries no GPS metadata. For a real Gwalior
deployment, place zone-labeled crops under data/geo/&lt;Zone&gt;/ and rerun; the analytics are unchanged.</div>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Render the pollution dashboard")
    parser.add_argument(
        "--mode",
        choices=("predictions", "demo-ground-truth"),
        default="predictions",
        help="predictions: consume a completed experiment run; "
        "demo-ground-truth: explicit demo labels (no model)",
    )
    parser.add_argument("--truth", action="store_true", help="alias for --mode demo-ground-truth")
    parser.add_argument(
        "--run-id", default=None, help="explicit completed experiment run directory"
    )
    parser.add_argument("--arm", default="D_qubo", help="selection arm to read")
    parser.add_argument("--clf", default="rbf_svm", help="classifier to read")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    truth_mode = args.truth or args.mode == "demo-ground-truth"

    provenance = {"mode": "demo-ground-truth" if truth_mode else "predictions"}
    if truth_mode:
        df = load_crops()
    else:
        df = load_predictions(clf=args.clf, arm=args.arm, run_id=args.run_id)
        provenance = {
            "mode": "predictions",
            "run_id": df.attrs.get("run_id"),
            "arm": df.attrs.get("arm"),
            "clf": df.attrs.get("clf"),
            "reps": df.attrs.get("reps"),
        }
    zt = compute_zone_table(df)
    out_dir = ROOT / "results" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_truth" if truth_mode else ""
    zt.to_csv(out_dir / f"zone_table{suffix}.csv", index=False)
    render_dashboard(zt, out_dir / "dashboard.html", truth_mode=truth_mode, provenance=provenance)
    if zt.empty:
        print("No observations.")
    else:
        print(zt.to_string(index=False))
    print("saved:", out_dir / "dashboard.html")


if __name__ == "__main__":
    main()
