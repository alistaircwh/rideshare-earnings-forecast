# NYC Rideshare Earnings Prediction

> Forecasting hourly driver earnings for Uber and Lyft across **~105 million NYC trips** (May–November 2023), with PySpark, geospatial EDA, and a temporal holdout to simulate real-world generalisation.

![Earnings by hour of day](plots/average_earnings_per_hour_by_hour_of_day.png)

---

## Results

| Model | RMSE | R² |
|---|---|---|
| **Linear Regression** | **$4.88** | **0.850** |
| Lasso Regression (L1, λ=1) | $5.42 | 0.815 |

**Key findings**

- ✈️ **Airport pickups pay best.** JFK, LGA and EWR consistently yield the highest earnings per hour across every day of the week.
- 🌙 **Earnings peak before sunrise.** Hourly earnings spike between **5–8 AM** (commuter rush) and again near **10 PM–midnight** (late-night demand), then bottom out mid-afternoon.
- 🚖 **Uber narrowly out-earns Lyft.** Uber (HV0003) drivers earn marginally more per hour on median than Lyft (HV0005), with a noticeably longer low-earnings tail.
- 🌧️ **Weather matters — modestly.** Feels-like temperature and precipitation have a small but measurable effect on per-hour earnings.

---

## Visualisations

**Earnings per hour — Uber vs Lyft**

![Earnings per hour by service](plots/earnings_per_hour_by_service.png)

**Interactive geospatial maps** (rendered via htmlpreview — click to open)

| Map | Description |
|---|---|
| [Average earnings per hour](https://htmlpreview.github.io/?https://github.com/alistaircwh/Rideshare-Earnings-Forecasting/blob/main/plots/average_earnings_per_hour_map.html) | Choropleth of mean hourly earnings by NYC taxi zone |
| [Log average earnings per hour](https://htmlpreview.github.io/?https://github.com/alistaircwh/Rideshare-Earnings-Forecasting/blob/main/plots/log_average_earnings_per_hour_map.html) | Log-scaled version — surfaces variation across mid-range zones |
| [Trip-count demand](https://htmlpreview.github.io/?https://github.com/alistaircwh/Rideshare-Earnings-Forecasting/blob/main/plots/trip_count_demand_map.html) | Pickup-volume heatmap across the five boroughs |

---

## Methodology

```
NYC TLC (FHVHV)          External Data
May–Oct 2023       ──┐   (weather, holidays,
                      ├──► Preprocessing ──► Feature Engineering ──► Model Training
Nov 2023 (holdout) ──┘   taxi zone shapefiles)                            │
                                                                           ▼
                                                                     Evaluation +
                                                                     EDA / Maps
```

**Train/test split.** Months 5–10 form the training set (~90M trips); **November 2023** is held out (~15M trips) as a temporal test set to simulate real-world generalisation rather than random shuffling.

**Features.** One-hot encoded license type (Uber/Lyft), standardised trip distance, day-of-week and hour-of-day encodings, pickup-zone encoding (NYC taxi zones), weather conditions (feels-like temperature, precipitation type/amount), and a public-holiday indicator.

**Outlier removal.** Domain-aware IQR rule with threshold `(√log(N) − 0.5) × IQR` — more permissive than the standard 1.5×IQR multiplier so high-value airport and long-haul trips are not over-filtered. Applied only to groups with N > 100.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **PySpark 3.5** | Distributed data processing and ML |
| pandas / NumPy | Local data manipulation and visualisation prep |
| geopandas + folium | Geospatial analysis and interactive choropleth maps |
| scikit-learn / XGBoost | Supporting modelling utilities |
| matplotlib / seaborn | Static visualisations |

---

## Data Source

[NYC TLC For-Hire Vehicle High-Volume trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — publicly available monthly Parquet files. Only FHVHV records (Uber: `HV0003`, Lyft: `HV0005`) are used.

External datasets: hourly NYC weather data (May 2023–May 2024) and NYC public holiday calendar.

---

## Repo Layout

The project follows a **modules + thin notebooks** pattern. All non-trivial logic lives in importable Python modules under [`scripts/`](scripts/); the notebooks are a narrative orchestration layer.

```
scripts/
├── download.py         CLI script — fetches raw FHVHV Parquet from NYC TLC
├── spark.py            Spark session factory + curated-data schema
├── io_utils.py         Load raw FHVHV / weather / holidays / curated Parquet
├── preprocessing.py    Cleaning, feature engineering, weather join, holiday flag
├── modelling.py        Feature assembly, temporal split, train, evaluate
├── visualisation.py    Geospatial choropleths, boxplot, hourly line plot
└── functions.py        IQR outlier rule, z-score standardisation, shape/null helpers

notebook/
├── preprocess.ipynb    Orchestrates the preprocessing pipeline
├── analysis.ipynb      Builds the maps and EDA plots
└── model.ipynb         Trains and evaluates Linear + Lasso models
```

## Setup

**Requirements:** Python 3.12, Java (for PySpark)

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run these steps in order:

1. **Download raw data**
   ```bash
   python scripts/download.py
   ```
   Downloads FHVHV Parquet files for May–November 2023 into `data/raw/`.

2. **Preprocess** — [`notebook/preprocess.ipynb`](notebook/preprocess.ipynb)
   Orchestrates `preprocessing.py` to clean and feature-engineer the trip data, join external weather + holiday data, and write curated month-partitioned Parquet to `data/curated/`.

3. **Analysis** — [`notebook/analysis.ipynb`](notebook/analysis.ipynb)
   Calls `visualisation.py` to build folium choropleths and matplotlib charts. Outputs to `plots/`.

4. **Model** — [`notebook/model.ipynb`](notebook/model.ipynb)
   Calls `modelling.py` to assemble features, temporally split, and train Linear + Lasso regressions evaluated on the November 2023 holdout.

---

## Future Improvements

- **Richer features** — traffic density, surge multiplier data, driver ratings.
- **Non-linear models** — benchmark against gradient boosting (XGBoost, LightGBM) or a small neural network.
- **Per-zone models** — fit separately per borough or zone cluster to capture spatial heterogeneity.
- **Real-time scoring** — deploy as a lightweight API for live earnings estimation.

---

## Report

Full methodology, analysis and results write-up: [report/report.pdf](report/report.pdf).
