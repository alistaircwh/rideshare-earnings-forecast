# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAST30034 (Applied Data Science) project at the University of Melbourne. Goal: predict earnings for rideshare services (Uber/Lyft via NYC TLC FHVHV data) for May–November 2023. Training data: May–October 2023; test set: November 2023.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12. PySpark 3.5.2 is the primary processing engine — a local JVM must be available.

## Execution Pipeline (run in order)

1. **Download raw data**
   ```bash
   python scripts/download.py
   ```
   Fetches NYC TLC FHVHV Parquet files into `data/raw/`.

2. **Preprocess** — run `notebook/preprocess.ipynb`
   Cleans data, one-hot encodes categorical features (license/day/hour/location/precipitation), standardizes numerics, applies IQR outlier removal (domain-aware: only for groups with N > 100). Outputs curated Parquet files to `data/curated/`.

3. **Analysis** — run `notebook/analysis.ipynb`
   EDA and geospatial visualizations (geopandas + folium). Outputs HTML maps and PNG plots to `plots/`.

4. **Modelling** — run `notebook/model.ipynb`
   PySpark ML `LinearRegression` trained on months 5–10, evaluated on month 11. Metrics: RMSE and R².

## Architecture

The project follows a **modules + thin notebooks** pattern. Core logic lives in `scripts/`; notebooks are a narrative layer that imports + orchestrates.

```
scripts/
  download.py       # CLI: downloads raw FHVHV Parquet from NYC TLC
  spark.py          # Spark session factory + curated-data StructType schema
  io_utils.py       # load_raw_fhvhv, load_weather, load_holidays, load_curated, write_curated
  preprocessing.py  # Cleaning, feature engineering, weather join, holiday flag
  modelling.py      # Feature assembly, temporal split, train_linear_regression, evaluate
  visualisation.py  # Folium choropleths + matplotlib/seaborn plots
  functions.py      # IQR outlier rule, z-score standardisation, shape/null helpers
notebook/
  preprocess.ipynb  # Step 2: orchestrates preprocessing.py
  analysis.ipynb    # Step 3: builds maps + plots via visualisation.py
  model.ipynb       # Step 4: trains + evaluates models via modelling.py
data/
  raw/              # Raw FHVHV Parquet files (gitignored)
  curated/          # Month-partitioned Parquet output from preprocessing
  raw_csv/          # External reference data (taxi zone shapefiles, weather, holidays)
  results/          # Model prediction outputs
plots/              # Generated visualisations (PNG + interactive folium HTML)
report/             # Final PDF report
```

**Data flow:** `download.py` → `data/raw/` → `preprocess.ipynb` → `data/curated/chunk_*.parquet` → `analysis.ipynb` + `model.ipynb` → `plots/` + `data/results/`

## Key Technical Notes

- **Uber = HV0003, Lyft = HV0005** — these are the license-type filters (see `FHVHV_LICENSES` in `preprocessing.py`).
- Each notebook prepends `sys.path.append('../scripts')` and imports directly (e.g. `from preprocessing import drop_unused_columns`). The modules import each other the same way (`from functions import apply_iqr_rule`), so the sys.path manipulation in the notebook is sufficient for all transitive imports.
- The curated Parquet schema is canonicalised in `spark.CURATED_SCHEMA` and loaded via `io_utils.load_curated()` — required because the VectorUDT columns (`license_vec`, `day_vec`, etc.) don't survive a schema-inferring re-read.
- PySpark sessions are created locally inside each notebook via `get_spark()`; no external cluster required.
- `.gitattributes` marks `plots/*.html` as `linguist-generated=true` so the folium-generated maps don't dominate GitHub's language stats.
