"""Live web API for the Stage-5 pollution dashboard.

Wraps ``scripts.pollution_dashboard`` (the single source of truth for the
analytics: zones, cleanliness score, hazard-weighted PSI, Wilson bounds,
majority-vote predictions, manifest + SHA-256 provenance checks) behind a
JSON HTTP API for the React UI in ``dashboard-ui/``.

Provenance rules are NOT weakened here: prediction mode only ever serves a
manifest-verified complete run (or raises 422 with the reason), and
ground-truth mode must be requested explicitly — never as a fallback.

Usage (dev):
    .venv/Scripts/python.exe -m uvicorn scripts.dashboard_api:app --port 8020
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pollution_dashboard as PD  # noqa: E402
from src.quobo.config import ROOT as QROOT  # noqa: E402

app = FastAPI(title="Quobo Pollution Dashboard API", version="1.0")

# Vite dev server origins (the built dist is served same-origin by the mount
# at the bottom, so no CORS needed there).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "quobo-dashboard-api"}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    """All experiment runs with their manifest status, arms and classifiers.

    Incomplete/failed/legacy runs are listed too (the UI shows them disabled
    with the reason), but only ``complete`` ones can be served in prediction
    mode — enforced again server-side by load_predictions()."""
    experiments = QROOT / "experiments"
    runs: list[dict] = []
    if not experiments.is_dir():
        return runs
    for d in sorted(experiments.iterdir()):
        manifest_path = d / PD.MANIFEST_NAME
        if not d.is_dir():
            continue
        entry: dict = {
            "run_id": d.name,
            "status": "no_manifest",
            "reps_completed": 0,
            "arms": [],
            "clfs": [],
            "selectable": False,
        }
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
                entry["status"] = "corrupt_manifest"
            entry["status"] = manifest.get("status", "unknown")
            entry["reps_completed"] = manifest.get("reps_completed") or 0
            entry["arms"] = manifest.get("arms") or []
            if entry["status"] == "complete" and entry["arms"]:
                rep0 = d / f"rep0_{entry['arms'][0]}.json"
                if rep0.is_file():
                    try:
                        entry["clfs"] = sorted(
                            json.loads(rep0.read_text(encoding="utf-8")).get("metrics", {})
                        )
                    except json.JSONDecodeError:
                        entry["clfs"] = []
                entry["selectable"] = True
        runs.append(entry)
    return runs


@app.get("/api/dashboard")
def dashboard(
    mode: str = Query("predictions", pattern="^(predictions|demo-ground-truth)$"),
    run_id: str | None = Query(None),
    arm: str = Query("D_qubo"),
    clf: str = Query("rbf_svm"),
    zone_mode: str = Query("batch", pattern="^(batch|folder)$"),
) -> dict:
    """Compute the zone table live from a manifest-verified run (or the
    explicit demo ground-truth mode), and return it as JSON."""
    truth_mode = mode == "demo-ground-truth"
    try:
        if truth_mode:
            df = PD.load_crops()
            provenance: dict = {"mode": "demo-ground-truth"}
        else:
            df = PD.load_predictions(clf=clf, arm=arm, run_id=run_id)
            provenance = {
                "mode": "predictions",
                "run_id": df.attrs.get("run_id"),
                "arm": df.attrs.get("arm"),
                "clf": df.attrs.get("clf"),
                "reps": df.attrs.get("reps"),
            }
        if zone_mode == "folder":
            df = df.copy()
            df["zone_dir"] = df["crop_path"].map(lambda p: PD.zone_of(str(p), "folder"))
        zt = PD.compute_zone_table(df)
    except PD.ProvenanceError as exc:
        # 422: the request itself is well-formed but provenance refuses it —
        # surfaced in the UI as an explicit banner, never a silent fallback.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classes = list(PD.HAZARD_W)
    zones = []
    for _, r in zt.iterrows():
        zones.append(
            {
                "zone": str(r["zone"]),
                "total": int(r["total"]),
                "clean": int(r["clean"]),
                "dirty": int(r["dirty"]),
                "cleanliness": float(r["cleanliness"]),
                "psi": int(r["psi"]),
                "band": str(r["band"]),
                "psi_ci": str(r["psi_ci"]),
                "psi_lo": int(r["psi_lo"]),
                "psi_hi": int(r["psi_hi"]),
                "dirty_rate_lo": float(r["dirty_rate_lo"]),
                "dirty_rate_hi": float(r["dirty_rate_hi"]),
                "vote_agreement": None
                if r["vote_agreement"] is None
                else float(r["vote_agreement"]),
                "counts": {c: int(r[c]) for c in classes},
            }
        )

    kpis: dict = {
        "cleanliness": None,
        "worst_zone": None,
        "total_items": 0,
        "zone_count": len(zones),
    }
    if not zt.empty and int(zt["total"].sum()) > 0:
        kpis = {
            "cleanliness": round(float(zt["clean"].sum() / zt["total"].sum() * 100), 1),
            "worst_zone": {
                "zone": str(zt.loc[zt["psi"].idxmax(), "zone"]),
                "psi": int(zt["psi"].max()),
                "band": str(zt.loc[zt["psi"].idxmax(), "band"]),
            },
            "total_items": int(zt["total"].sum()),
            "zone_count": len(zones),
        }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "zone_mode": zone_mode,
        "provenance": provenance,
        "source_label": PD._source_label(truth_mode, provenance),
        "classes": classes,
        "hazard_weights": PD.HAZARD_W,
        "bands": [{"max": t, "name": b, "color": PD.BAND_COLORS[b]} for t, b in PD.PSI_BANDS],
        "band_colors": PD.BAND_COLORS,
        "kpis": kpis,
        "zones": zones,
    }


# Production-style single-port serving: if the React app has been built,
# serve dashboard-ui/dist at "/" (API routes above take precedence).
_dist = ROOT / "dashboard-ui" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")
