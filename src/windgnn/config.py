"""Paths, dataset constants and figure styling for the wind-following GNN project.

The SDWPF release is not redistributed with the repository: point
``WINDGNN_DATA_ROOT`` at the folder that contains ``sdwpf_kddcup/`` and
``sdwpf_full/``. By default we look for ``SDWPF_dataset/`` next to the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("WINDGNN_DATA_ROOT", REPO_ROOT / "SDWPF_dataset"))

KDDCUP_DIR = DATA_ROOT / "sdwpf_kddcup"
FULL_DIR = DATA_ROOT / "sdwpf_full"

#: 245-day / 134-turbine variant used throughout this project (matches Zhang et al. 2026).
SCADA_245 = KDDCUP_DIR / "sdwpf_245days_v1.csv"
LOCATION_245 = KDDCUP_DIR / "sdwpf_baidukddcup2022_turb_location.csv"
#: Two-year variant with ERA5 weather: kept for future work / cross-checks only.
SCADA_FULL_PARQUET = FULL_DIR / "sdwpf_2001_2112_full.parquet"
LOCATION_FULL = FULL_DIR / "sdwpf_turb_location_elevation.csv"

INTERIM_DIR = REPO_ROOT / "data" / "interim"
RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = REPO_ROOT / ".cache"

for _d in (INTERIM_DIR, RAW_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#: matplotlib needs a writable config dir; ~/.config may be read-only here.
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mpl"))

# --------------------------------------------------------------------------- #
# Dataset constants (SDWPF, KDD Cup 2022 release)
# --------------------------------------------------------------------------- #
N_TURBINES = 134
STEPS_PER_DAY = 144          # 10-minute SCADA records
N_DAYS = 245
N_STEPS = N_DAYS * STEPS_PER_DAY          # 35 280 timestamps per turbine
N_RECORDS = N_TURBINES * N_STEPS          # 4 727 520 rows
SAMPLE_MINUTES = 10

RATED_POWER_KW = 1500.0      # Sinovel SL1500/82, 1.5 MW
HUB_HEIGHT_M = 70.0
ROTOR_DIAMETER_M = 82.0

#: The KDD Cup release stores ``Day`` (1..245) and ``Tmstamp`` (00:00..23:50) but no
#: calendar date. ``sdwpf_full`` cannot be used to recover it: its timestamp column is
#: internally inconsistent (some days hold 144 rows, others 96) and the KDD Cup values
#: do not occur in it. Any datetime is therefore RELATIVE to this anchor and is only
#: used for plotting/seasonal framing -- never as a scientific claim.
RELATIVE_ANCHOR = "2020-01-01"

SCADA_COLUMNS = [
    "TurbID", "Day", "Tmstamp",
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
    "Pab1", "Pab2", "Pab3", "Prtv", "Patv",
]

#: Columns carrying SCADA measurements (i.e. everything but the index columns).
MEASUREMENT_COLS = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]

TARGET_COL = "Patv"

#: Dataset-supplied validity thresholds (Zhou et al., Sci. Data 11:649, Table 2 / Usage Notes).
UNKNOWN_WSPD_THRESHOLD = 2.5     # m/s: Patv<=0 with Wspd>2.5 means the turbine was stopped
PITCH_UNKNOWN_THRESHOLD = 89.0   # deg: blades feathered => turbine at rest
NDIR_ABNORMAL_RANGE = (-720.0, 720.0)
WDIR_ABNORMAL_RANGE = (-180.0, 180.0)

#: Chronological split used by Zhang et al. 2026 (70 / 10 / 20 %), in days.
SPLIT_DAYS = {"train": (1, 171), "val": (172, 196), "test": (197, 245)}

#: How the absolute (true-north) wind direction is reconstructed from the SCADA
#: columns. ``Wdir`` is documented as the angle *between* the wind direction and the
#: nacelle direction, so the absolute direction must combine ``Ndir`` and ``Wdir``.
#: Notebook 01 tests the candidates empirically; this is the one it selects.
WIND_DIRECTION_CONVENTION = "rel_sum"     # abs_from = (Ndir + Wdir) mod 360
WIND_DIRECTION_CANDIDATES = {
    "rel_sum": "absolute direction = Ndir + Wdir (wind side of the nacelle)",
    "rel_diff": "absolute direction = Ndir - Wdir (opposite sign convention)",
    "absolute": "Wdir is already the absolute wind direction (Ndir unused)",
}

# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #

def set_plot_style(force_agg: bool | None = None) -> None:
    """Apply a consistent, report-friendly matplotlib style.

    Inside a Jupyter kernel the active (inline) backend is kept so figures display;
    in plain scripts we switch to Agg so everything runs headless.
    """
    import sys

    import matplotlib

    if force_agg is None:
        force_agg = "ipykernel" not in sys.modules
    if force_agg:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def save_figure(fig, name: str):
    """Save a figure into ``results/figures`` and return the path."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path
