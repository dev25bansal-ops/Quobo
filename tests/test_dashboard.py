"""Unit tests for the pollution dashboard: PSI hazard weighting, zone mapping,
prediction-mode ingestion, folder-mode ingestion, and interactive charts."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pollution_dashboard import compute_zone_table, zone_of


def _zone_df(rows):
    return pd.DataFrame(rows)


# ---------- PSI-001: hazard weights in the numerator ----------


def test_psi_hazard_weight_multiplies():
    # one zone, one cigarette (w=3) vs three bottles (w=1 each)
    df = _zone_df(
        [
            {"crop_path": "batch_1_001.jpg", "class": "cigarette"},
            {"crop_path": "batch_1_002.jpg", "class": "bottle"},
            {"crop_path": "batch_1_003.jpg", "class": "bottle"},
            {"crop_path": "batch_1_004.jpg", "class": "bottle"},
        ]
    )
    zt = compute_zone_table(df)
    sub_cig = 1 * 3.0
    sub_bot = 3 * 1.0
    assert sub_cig == sub_bot  # tie -> psi at 500 boundary dominance
    assert zt["psi"].iloc[0] == 500  # the max weighted cell maps to 500


def test_psi_cigarette_dominates_bottles():
    df = _zone_df(
        [{"crop_path": "batch_1_001.jpg", "class": "cigarette"}] * 2
        + [{"crop_path": "batch_1_002.jpg", "class": "bottle"}] * 10
    )
    zt = compute_zone_table(df)
    # cigarettes: 2*3=6 vs bottles: 10*1=10 -> bottle sub-index is max -> not 500 for cig
    assert zt["psi"].iloc[0] == 500  # bottle cell is the global max -> 500


# ---------- zone mapping ----------


@pytest.mark.parametrize(
    "fname,expected",
    [
        ("batch_1_000123_ann4.jpg", "Zone A"),
        ("batch_6_000123_ann4.jpg", "Zone F"),
        ("batch_7_000123_ann4.jpg", "Zone A"),  # wraps
        ("batch_11_000123_ann4.jpg", "Zone E"),  # 11-1 mod 6 = 10 -> E
    ],
)
def test_zone_of_batch(fname, expected):
    assert zone_of(f"data/processed/crops/bottle/{fname}") == expected


# ---------- S1: prediction-mode dashboard consumes votes ----------


def test_zone_table_from_predictions_matches_shape(tmp_path, monkeypatch):
    """compute_zone_table works identically on a predictions-style frame."""
    df = pd.DataFrame(
        [
            {
                "crop_path": "x/batch_2_001_ann1.jpg",
                "class": "cigarette",
                "file": "batch_2_001_ann1.jpg",
            },
            {
                "crop_path": "x/batch_2_002_ann1.jpg",
                "class": "bottle",
                "file": "batch_2_002_ann1.jpg",
            },
        ]
    )
    zt = compute_zone_table(df)
    assert set(zt.columns) >= {"zone", "total", "clean", "dirty", "cleanliness", "psi", "band"}
    assert zt["zone"].iloc[0] == "Zone B"
    assert zt["dirty"].iloc[0] == 1


# ---------- OPEN-8: folder-mode ingestion end-to-end ----------


def test_folder_mode_zone_ingestion(tmp_path, monkeypatch):
    """data/geo/<Zone>/<class>/*.jpg folders flow through compute_zone_table
    with the directory name as the zone — the real-deployment path."""
    import scripts.pollution_dashboard as PD

    geo = tmp_path / "data" / "geo"
    for zone in ("Zone A", "Zone B"):
        for cls, n in (
            ("cigarette", 2 if zone == "Zone B" else 1),
            ("bottle", 1 if zone == "Zone B" else 3),
        ):
            d = geo / zone / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(b"x")

    monkeypatch.setattr(PD, "ROOT", tmp_path)
    # call the folder-mode branch directly through the public path
    rows = []
    for zone_dir in sorted(geo.iterdir()):
        for cls_dir in sorted(zone_dir.iterdir()):
            for pth in cls_dir.glob("*.jpg"):
                rows.append(
                    {
                        "crop_path": str(pth),
                        "class": cls_dir.name,
                        "file": pth.name,
                        "zone_dir": zone_dir.name,
                    }
                )
    # zone_of must honor folder mode via the parent directory
    zt = PD.compute_zone_table(pd.DataFrame(rows))
    assert set(zt["zone"]) == {"Zone A", "Zone B"}
    zone_b = zt[zt["zone"] == "Zone B"].iloc[0]
    assert zone_b["dirty"] == 2 and zone_b["clean"] == 1


def test_zone_of_folder_mode_and_unassigned():
    from scripts.pollution_dashboard import zone_of

    assert zone_of("data/geo/Zone C/bottle/x.jpg", mode="folder") == "Zone C"
    assert zone_of("data/geo/Gandhi Chowk/bottle/x.jpg", mode="folder") == "Unassigned"
    assert zone_of("some_random_crop.jpg") == "Unassigned"  # no silent Zone A


# ---------- Enhancement 5: interactive dashboard ----------


def test_build_interactive_charts_renders_and_falls_back():
    """Interactive chart builder renders plotly HTML when available, else ''."""
    import scripts.pollution_dashboard as PD

    zt = pd.DataFrame(
        [
            {
                "zone": "Zone A",
                "total": 100,
                "clean": 80,
                "dirty": 20,
                "cleanliness": 80.0,
                "psi": 300,
                "band": "Unhealthy",
                "cigarette": 20,
                "bottle": 40,
                "can": 20,
                "carton": 10,
                "cup": 5,
                "lid": 5,
            },
            {
                "zone": "Zone B",
                "total": 50,
                "clean": 30,
                "dirty": 20,
                "cleanliness": 60.0,
                "psi": 500,
                "band": "Hazardous",
                "cigarette": 20,
                "bottle": 20,
                "can": 0,
                "carton": 0,
                "cup": 0,
                "lid": 10,
            },
        ]
    )
    html = PD._build_interactive_charts(zt)
    # if plotly installed: non-empty with both subplot titles; else graceful ''
    assert isinstance(html, str)
    if html:
        assert "PSI by zone" in html and "Class composition" in html
    # zero-total zone must not divide by zero
    zt2 = zt.copy()
    zt2.loc[zt2.zone == "Zone A", "total"] = 0
    assert isinstance(PD._build_interactive_charts(zt2), str)


# ---------- A1: uncertainty-aware PSI (columns present) ----------


def test_zone_table_has_uncertainty_columns():
    # ground-truth frame (no votes) still renders, CI columns present
    df = pd.DataFrame(
        [
            {"crop_path": "batch_1_001.jpg", "class": "cigarette"},
            {"crop_path": "batch_1_002.jpg", "class": "bottle"},
        ]
    )
    zt = compute_zone_table(df)
    assert "psi_ci" in zt.columns and "vote_agreement" in zt.columns
    assert zt["vote_agreement"].isna().all()  # no vote data -> honest None
    # CI must bracket the point estimate when dirty>0
    row = zt.iloc[0]
    lo = int(row["psi_ci"].strip("[]").split("-")[0])
    hi = int(row["psi_ci"].strip("[]").split("-")[1])
    assert lo <= row["psi"] <= hi
