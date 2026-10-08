# Wind-following GNN for wind-turbine power forecasting (SDWPF)

Forecast the power of **every turbine** in a wind farm up to 6 hours ahead — including
turbines whose own SCADA measurements are missing — using a graph whose **directed edges
follow the wind**: an edge `A → B` exists at time `t` when `B` lies downwind of `A`, so the
graph changes as the wind turns.

This repository contains the data-analysis and graph-construction stages of the project, plus
the reusable `windgnn` package they are built on.

| Stage | Content | Status |
|---|---|---|
| A | Data analysis: integrity, wind-direction reconstruction, record validity, geometry, correlation structure, splits | done — `notebooks/01_data_analysis.ipynb` |
| B | Graph construction: static baselines, the wind-following graph, farm-orientation estimation, upwind census | done — `notebooks/02_graph_construction.ipynb` |
| C | Windowing, outage simulation and baselines (persistence, LSTM, GCN, STGCN, Graph WaveNet, impute-then-forecast) | planned |
| D | GATv2 model on the wind-following graph, trained with randomly hidden inputs | planned |
| E | H1/H2/H3 experiments (hidden-input share, upwind-neighbour groups, up/down-wind asymmetry) + explainability | planned |

---

## Repository layout

```
notebooks/
  01_data_analysis.ipynb        Stage A — data analysis
  02_graph_construction.ipynb   Stage B — graph construction
src/windgnn/
  config.py                     paths, dataset constants, plot styling
  data.py                       loading, cleaning, validity flags, wind direction
  graphs.py                     geometry, static graphs, wind-following graph
  viz.py                        plotting helpers
results/
  figures/                      report figures (stage_a_*.png, stage_b_*.png)
  tables/                       result tables (stage_a_*.csv, stage_b_*.csv)
scripts/
  make_slide_assets.py          exports slide-ready figures/tables into slides/
slides/                         presentation assets + PLACEMENT.md (where each one goes)
data/interim/                   generated caches (gitignored)
SDWPF_dataset/                  raw dataset — download separately (gitignored)
pyproject.toml / uv.lock        environment definition
```

---

## Quick start

### 1. Install `uv`

`uv` is a single fast tool that replaces `pip`, `venv`, `pip-tools` and `pyenv`. Install it
once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# or, if you already have a package manager
brew install uv          # macOS / Homebrew
pipx install uv          # any platform with pipx
pip install uv           # as a plain Python package
```

Verify: `uv --version`. You normally do **not** need to install Python yourself — `uv` can
download and manage interpreters.

### 2. `uv` in sixty seconds

The model to keep in mind:

* `pyproject.toml` — what the project *asks for* (dependency names and version ranges).
* `uv.lock` — the exact resolved versions, committed so everyone gets an identical environment.
* `.venv/` — the actual environment; created and updated automatically, never edited by hand.
* `.python-version` — pins the interpreter for this repo (here: **3.13**).

| Command | What it does |
|---|---|
| `uv sync` | Create/refresh `.venv` from `pyproject.toml` + `uv.lock` |
| `uv run <cmd>` | Run a command inside the project environment (syncs first if needed) |
| `uv add <pkg>` | Add a dependency, update `pyproject.toml` and `uv.lock` |
| `uv add --dev <pkg>` | Same, but as a development-only dependency |
| `uv remove <pkg>` | Drop a dependency |
| `uv lock` / `uv lock --upgrade` | Re-resolve the lock file (optionally to newer versions) |
| `uv python install 3.13` | Download a managed Python interpreter |
| `uv python pin 3.13` | Write `.python-version` |
| `uv tree` | Show the dependency tree |

A typical session:

```bash
git clone git@github.com:andrei00001111/GraphMLAssignment.git
cd GraphMLAssignment

uv add numpy                  # If you want to add a dependency (instead of doing "pip install numpy")
uv sync                       # build .venv, install pandas/numpy/networkx/... (pinned)
```
Now you can select the kernel and run your notebook as usual.

Adding something later (example: PyTorch for Stage C/D):

```bash
uv add torch torch_geometric  # updates pyproject.toml + uv.lock and installs it
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Notes that save time:

* **No activation needed.** `uv run …` uses the project environment by default. If you prefer
  a classic shell, `source .venv/bin/activate` also works.
* **The package lives in `src/`**, which is not installed into the environment. Run code with
  `PYTHONPATH=src`, or just use the notebooks — each one bootstraps `sys.path` itself in its
  first cell.
* **Every command is reproducible** from `uv.lock`: `uv sync --frozen` installs exactly the
  locked versions without re-resolving.

### 3. Get the dataset

The SDWPF dataset is **not** redistributed here (size and licence), and `SDWPF_dataset/` is
gitignored. Download it from Figshare (DOI [`10.6084/m9.figshare.24798654`](https://doi.org/10.6084/m9.figshare.24798654))
and arrange it like this:

```
SDWPF_dataset/
  sdwpf_kddcup/
    sdwpf_245days_v1.csv                      # the SCADA table used throughout (334 MB)
    sdwpf_baidukddcup2022_turb_location.csv   # turbine x, y coordinates
  sdwpf_full/                                 # optional: two-year variant
    sdwpf_2001_2112_full.parquet
    sdwpf_turb_location_elevation.csv         # coordinates + elevation
```

Only the `sdwpf_kddcup` files are required. Point elsewhere with an environment variable if
you keep the data on another disk:

```bash
export WINDGNN_DATA_ROOT=/path/to/SDWPF_dataset
```

---

## The dataset

### What it is

**SDWPF** (*Spatial Dynamic Wind Power Forecasting*) was released with the Baidu KDD Cup 2022
and described by Zhou et al., *Scientific Data* **11**:649 (2024). It is real SCADA data from a
Chinese wind farm of **134 turbines** (China Longyuan Power Group). Each turbine is a
**Sinovel SL1500/82**: 1.5 MW rated power, 82 m rotor diameter, 70 m hub height, three blades.
Records are **10-minute averages** of 1 Hz samples, and turbine positions are given in metres
in a relative coordinate system.

Two variants ship in the same release:

| variant | period | records | columns |
|---|---|---|---|
| `sdwpf_kddcup` — **used here** | 245 days | 134 × 245 × 144 = **4,727,520** | 13 SCADA |
| `sdwpf_full` | Jan 2020 – Dec 2021 | 11,361,191 | 19 (SCADA + ERA5 weather) |

We use the 245-day variant because it is the one used in the published GNN baselines, which
makes the comparison fair. The two-year file is kept for future work: be aware that its
timestamp column is internally inconsistent (some days hold 144 rows, others 96) and that it
cannot be used to recover calendar dates for the KDD Cup file.

### Files

| File | Contents |
|---|---|
| `sdwpf_245days_v1.csv` | The SCADA table. Rows are **turbine-major**: all 35,280 timestamps of turbine 1, then turbine 2, … |
| `sdwpf_baidukddcup2022_turb_location.csv` | `TurbID, x, y` — relative position in metres |
| `sdwpf_turb_location_elevation.csv` | Same, plus `Ele` — ground elevation in metres |
| `sdwpf_2001_2112_full.parquet` | Two-year variant with weather columns (not used yet) |

### Column reference

| # | Column | Unit | Meaning | Notes |
|---|---|---|---|---|
| 1 | `TurbID` | – | Turbine identifier | 1 … 134 |
| 2 | `Day` | – | Day of the record | 1 … 245; **no calendar date is provided** |
| 3 | `Tmstamp` | `HH:MM` | Time of day | 144 values, 10-minute steps (UTC+08:00) |
| 4 | `Wspd` | m/s | Wind speed at the top of the turbine | mechanical anemometer |
| 5 | `Wdir` | ° | **Relative** wind direction — the angle between the wind direction and the nacelle direction | see the warning below |
| 6 | `Etmp` | °C | Ambient temperature at the nacelle | |
| 7 | `Itmp` | °C | Temperature inside the nacelle | |
| 8 | `Ndir` | ° | Nacelle (yaw) bearing, degrees from true north | can wind up past ±360° (dataset allows ±720°) |
| 9–11 | `Pab1`, `Pab2`, `Pab3` | ° | Blade pitch angles | in practice identical for the three blades |
| 12 | `Prtv` | kW | Reactive power | often slightly negative |
| 13 | `Patv` | kW | **Active power — the forecasting target** | rated 1,500 kW |

The two-year variant adds ERA5 weather columns: `T2m` (°C), `Sp` (Pa), `RelH` (relative
humidity), `Wspd_w` (m/s at 10 m), `Wdir_w` (° at 10 m), `Tp` (total precipitation, m).

### Watch out: `Wdir` is *relative*, not absolute

This is the single most important detail in the dataset, and getting it wrong inverts every
graph edge. `Wdir` is the angle between the wind direction and the **turbine's own nacelle
direction**, so the absolute direction (the bearing the wind blows *from*) is

```python
wind_dir_abs  = (Ndir + Wdir) % 360      # direction the wind comes FROM
wind_dir_flow = (wind_dir_abs + 180) % 360   # direction the wind blows TOWARDS
```

Two independent checks confirm it (both in `notebooks/01_data_analysis.ipynb`): taken as an
absolute direction, raw `Wdir` would be constant to within a few degrees for all 245 days
(circular concentration 0.998 — physically impossible), and its magnitude collapses from ~31°
in calm air to ~2.5° above 8 m/s, exactly the signature of a yaw-aligned relative angle.

Two further consequences:

* A minority of turbines carry a **stable per-turbine bias** in `Ndir`. Removing each
  turbine's own median deviation raises cross-turbine coherence from 0.47 to 0.96, so
  `windgnn.data` estimates a farm-wide direction from a robust consensus plus a per-turbine
  offset table (fitted on training days only).
* The turbine `(x, y)` coordinates have **no documented azimuth**. That unknown, plus the
  global rotation above, collapse into one parameter (`alpha`, the compass azimuth of the
  farm's `+y` axis) which `graphs.py` estimates from wake physics — see notebook 02.

### Record validity rules

Following the dataset paper, `windgnn.data.flag_records` labels every record:

| Category | Rule | Treatment |
|---|---|---|
| zero | `Patv < 0` | auxiliary loads; clipped to 0 |
| missing | sensor value absent (NaN) | not scored |
| unknown | `Patv ≤ 0 and Wspd > 2.5` (turbine stopped) **or** any `Pabᵢ > 89°` (blades feathered) | not scored |
| abnormal | `Ndir` outside [−720, 720] **or** `Wdir` outside [−180, 180] | not scored |
| `target_valid` | none of missing / unknown / abnormal | the only records metrics use |

Measured on the 245-day file: **26.7%** zero, **1.05%** missing, **22.9%** unknown, 127
abnormal records → **76.06% `target_valid`**. The categories are not mutually exclusive: a
turbine standing still with wind above the cut-in speed is counted as both `zero` and
`unknown`.

### Farm facts worth knowing

Measured in notebook 01 rather than assumed:

* **Layout:** 6 longitudinal columns of ~22 turbines, **474 m** spacing within a column,
  **991 m** between columns; extent 5.5 km × 12.1 km; elevation 1,391–1,470 m.
  (The published baseline describes five columns; our gap-based detection finds six.)
* **Correlation:** mean inter-turbine power correlation **0.896**, and it stays above **0.88 at
  every distance** — including pairs 8–13 km apart in the same column. Correlation-based
  similarity is therefore weak; this is why a *fixed* graph is a strong baseline.
* **Gaps:** 121,012 contiguous invalid gaps, median 3 steps, longest **2,861 steps = 19.9 days**
  (farm-wide outages happen, which matters for the "turbines with no neighbours" scenario).
* **Persistence floor** (what any model must beat), MAE in kW: 77.7 at 10 min, 173.3 at 1 h,
  334.9 at 6 h.
* **Power statistics** depend on the definition you use: farm mean **350.5 kW** over all
  records (matching the published baseline's 349.3 kW) but **456.0 kW** over `target_valid`
  records only. Always state which one a metric uses.
* **Splits:** days 1–171 / 172–196 / 197–245 = 70 / 10 / 20%, applied chronologically, with a
  purge gap of `input window + horizon` steps at each boundary.

---

## Running the notebooks

Run them **in order** — notebook 02 reads artifacts written by notebook 01.

```bash
uv run jupyter lab                       # interactive
```

Headless, to reproduce every figure and table without opening a browser:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/01_data_analysis.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/02_graph_construction.ipynb
```

In sandboxed or containerised environments where `~/.cache` and `~/.jupyter` are read-only,
redirect the tool caches into the repo first:

```bash
export UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.cache/mpl \
       JUPYTER_CONFIG_DIR=.cache/jupyter JUPYTER_DATA_DIR=.cache/jupyter/data \
       JUPYTER_RUNTIME_DIR=.cache/jupyter/runtime IPYTHONDIR=.cache/ipython
```

### `01_data_analysis.ipynb` — Stage A

Integrity checks (row/turbine/timestamp counts, duplicates, coordinate files), the
wind-direction reconstruction and its convention test, record-validity flags and gap
statistics, farm geometry and spacing, power statistics, the correlation structure, the
persistence baseline, and the chronological split definition.

### `02_graph_construction.ipynb` — Stage B

Builds the static baselines (k-NN, radius, fully connected) and the wind-following graph:

```
edge i -> j  at time t   <=>   (p_j - p_i) · u_t / |p_j - p_i| >= cos(phi)
                                and |p_j - p_i| <= R
```

where `u_t` is the unit wind-flow vector at `t`. Each edge carries four features:
`dist / R`, the along-flow cosine, the cross-flow sine, and the angle in degrees.

The notebook also estimates `alpha` (record-level downstream power gradient plus a
pair-fixed-effects wake check), reports graph size/density/degree across the hyperparameter
grid, and builds the upwind census that defines the exposure groups used by H2.

**Primary configuration:** `R = 1500 m`, cone half-angle `phi = 45°`, `alpha = 275°`
(18 rotor diameters along a column; mean in-degree 2.7, no isolated turbines; ~365 edges per
snapshot). The size matters — at `R = 1000 m` the sector-averaged mean in-degree is 1.08 and
**22.6% of turbines are isolated on average** (45% in the worst directions), which would
silently reduce the "graph" model to a per-turbine MLP.

Because the wake signal is weak at this farm's spacing (deficits of a few kW against a 456 kW
mean), `alpha` versus `alpha + 180 = 95°` is carried forward as the **reversed-direction
control** rather than assumed.

---

## The `windgnn` package

Import it after putting `src/` on the path (`PYTHONPATH=src`, or the notebooks' first cell).

**`windgnn.config`** — resolved paths (`DATA_ROOT`, `SCADA_245`, `INTERIM_DIR`, `FIGURES_DIR`, …),
dataset constants (`N_TURBINES`, `STEPS_PER_DAY`, `RATED_POWER_KW`, `SPLIT_DAYS`, the validity
thresholds), plotting style and `save_figure`.

**`windgnn.data`** — loading and preparation:

| Function | Purpose |
|---|---|
| `load_location()`, `load_scada()` | read coordinates / SCADA slices |
| `add_time_index()` | global 10-minute step `t` + a *relative* timestamp |
| `flag_records()`, `clean()` | the validity rules above; `clean` = time index + flags + derived features |
| `load_analysis_table()` | the cleaned 4.7M-row table, cached as parquet in `data/interim/` |
| `wind_direction()`, `wind_direction_diagnostics()` | absolute direction and the convention test |
| `estimate_farm_wind_direction()`, `turbine_yaw_offsets()` | farm direction + per-turbine bias table |
| `power_wide()`, `split_bounds()`, `split_labels()` | time × turbine matrices and split helpers |
| `circular_mean/median/std`, `mean_resultant_length`, `wrap180` | circular statistics |

**`windgnn.graphs`** — geometry and graphs:

| Function | Purpose |
|---|---|
| `pairwise_distances()`, `spacing_summary()`, `detect_columns()`, `extent()` | farm geometry |
| `static_distance_knn()`, `static_radius_graph()`, `static_full_graph()` | direction-blind baselines |
| `candidate_pairs()`, `pair_alignment()`, `edges_for_direction()` | edge features and per-snapshot edges |
| `flow_unit_vectors()`, `sector_index()`, `sector_centers()` | wind direction in the farm frame |
| `orientation_scan()`, `downstream_gradient_scan()`, `alignment_deficit_profile()` | estimating/validating `alpha` |
| `upwind_census()`, `graph_stats()` | exposure groups and graph descriptors |

**`windgnn.viz`** — `plot_layout`, `plot_power_curve`, `plot_correlation_matrix`,
`plot_correlation_vs_distance`, `plot_wind_rose`, `plot_direction_hist`,
`plot_missingness_by_turbine`, `plot_gap_histogram`, `plot_persistence_error`.

---

## Generated artifacts

Running the notebooks writes:

* `results/figures/stage_a_*.png`, `results/figures/stage_b_*.png` — report figures
* `results/tables/stage_a_*.csv`, `results/tables/stage_b_*.csv` — the numbers behind them
* `data/interim/` (gitignored):
  * `scada_245d_rel_sum.parquet` — cleaned 4.7M-row table (~250 MB, built once, then reused)
  * `farm_wind_direction.csv` — one absolute wind direction per snapshot
  * `turbine_yaw_offsets.csv` — the per-turbine nacelle bias table
  * `candidate_pairs_3000m.npz` — the precomputed pair geometry superset
  * `graph_meta.json` — the chosen graph parameters: `alpha_deg`, `radius_primary_m`,
    `cone_primary_deg` and the sweep grids

### Presentation assets

`slides/` holds figures and tables rendered at the **exact physical size of the slot they fill**
in the 16:9 deck (25.4 × 14.29 cm page; 23.7 × 9.5 cm body). Drop them in at 100 % scale at the
coordinates listed in `slides/PLACEMENT.md`; `slides/slide_text_drafts.md` has copy-paste text for
the hypothesis and risk slides. Regenerate with:

```bash
PYTHONPATH=src MPLCONFIGDIR=.cache/mpl uv run python scripts/make_slide_assets.py
```

Slot sizes are the `S` dict at the top of that script, so a layout change means editing the dict
rather than resizing images by hand.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Read-only file system: '.../.cache/uv'` | `export UV_CACHE_DIR=.uv-cache` before any `uv` command |
| matplotlib warns about a non-writable config dir | `export MPLCONFIGDIR=.cache/mpl` |
| `OSError: Read-only file system: '.../.jupyter'` | export `JUPYTER_CONFIG_DIR` / `JUPYTER_DATA_DIR` / `JUPYTER_RUNTIME_DIR` (block above) |
| `ModuleNotFoundError: No module named 'windgnn'` | run with `PYTHONPATH=src`, or use the notebooks |
| `Bad owner or permissions on /etc/ssh/ssh_config.d/...` when using git over SSH | fix with `sudo chown root:root /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf && sudo chmod 644 …`, or bypass with `GIT_SSH_COMMAND="ssh -F /dev/null" git …` |
| Want a clean rebuild | delete `data/interim/` (caches regenerate in seconds) |

No GPU is required to run Stages A and B. Stages C/D need PyTorch and PyTorch Geometric,
which are **not** installed yet: `uv add torch torch_geometric` (plus the CUDA/ROCm index URL
if you want acceleration).

---

## Data citation

> Zhou, J., Lu, X., Xiao, Y., Tang, J., Su, J., Li, Y., Liu, J., Lyu, J., Ma, Y. & Dou, D.
> *SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array.*
> Scientific Data **11**, 649 (2024). Dataset: Figshare, DOI
> [`10.6084/m9.figshare.24798654`](https://doi.org/10.6084/m9.figshare.24798654).

The official KDD Cup 2022 baseline (PaddlePaddle) is available at
<https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/wpf_baseline_gru>.
