"""Preprocessing pipeline for FHVHV trips and external datasets.

Each function takes a Spark DataFrame and returns a transformed one,
so they compose cleanly via DataFrame.transform(...) or sequential calls.
"""

from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from functions import apply_iqr_rule, standardise_column


FHVHV_LICENSES = ("HV0003", "HV0005")  # Uber, Lyft

# Columns not aligned with the earnings-prediction objective.
DROP_COLUMNS_INITIAL = [
    "dispatching_base_num",
    "DropOff_datetime",
    "DOLocationID",
    "originating_base_num",
    "request_datetime",
    "on_scene_datetime",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]


# ─── FHVHV trip cleaning ───────────────────────────────────────────────────


def drop_unused_columns(df: DataFrame) -> DataFrame:
    """Drop columns not relevant to the earnings-prediction task."""
    return df.drop(*DROP_COLUMNS_INITIAL)


def filter_to_uber_lyft(df: DataFrame) -> DataFrame:
    """Keep only Uber (HV0003) and Lyft (HV0005) trips."""
    return df.filter(F.col("hvfhs_license_num").isin(FHVHV_LICENSES))


def filter_date_range(df: DataFrame, column: str, start: str, end: str) -> DataFrame:
    """Filter rows where `column` is in the half-open interval [start, end)."""
    return df.filter((F.col(column) >= start) & (F.col(column) < end))


def filter_valid_pulocation(df: DataFrame) -> DataFrame:
    """Keep only trips whose pickup location ID is within TLC's defined range."""
    return df.filter((F.col("PULocationID") >= 1) & (F.col("PULocationID") <= 263))


def apply_baseline_filters(df: DataFrame) -> DataFrame:
    """Filter out implausible trip distance, time, tip, and pay values."""
    return (
        df.filter(F.col("trip_miles") > 0.15)
        .filter(F.col("trip_time") > 60)
        .filter(F.col("tips") >= 0)
        .filter(F.col("driver_pay") > 3)
    )


# ─── Feature engineering ───────────────────────────────────────────────────


def encode_categorical(
    df: DataFrame,
    input_col: str,
    output_vec_col: str,
    index_col: str | None = None,
) -> DataFrame:
    """One-hot encode `input_col` into `output_vec_col` via a StringIndexer."""
    index_col = index_col or f"{input_col}_index"
    df = StringIndexer(inputCol=input_col, outputCol=index_col).fit(df).transform(df)
    df = OneHotEncoder(inputCol=index_col, outputCol=output_vec_col).fit(df).transform(df)
    return df


def add_time_features(df: DataFrame) -> DataFrame:
    """Derive day_of_week, hour_of_day, month, pickup_hour, pickup_date."""
    return (
        df.withColumn("day_of_week", F.dayofweek("pickup_datetime"))
        .withColumn("hour_of_day", F.hour("pickup_datetime"))
        .withColumn("month", F.month("pickup_datetime"))
        .withColumn("pickup_hour", F.date_trunc("hour", F.col("pickup_datetime")))
        .withColumn("pickup_date", F.to_date(F.col("pickup_datetime")))
    )


def remove_outliers(df: DataFrame, columns: list[str]) -> DataFrame:
    """Apply the project's domain-aware IQR rule to each listed column."""
    for column in columns:
        df = apply_iqr_rule(df, column)
    return df


def standardise(df: DataFrame, columns: list[str]) -> DataFrame:
    """Z-score standardise each listed column."""
    for column in columns:
        df = standardise_column(df, column)
    return df


def add_earnings_columns(df: DataFrame) -> DataFrame:
    """Compute earnings (driver_pay + tips) and earnings_per_hour."""
    return df.withColumn(
        "earnings_per_hour",
        F.round((F.col("driver_pay") + F.col("tips")) / (F.col("trip_time") / 3600), 2),
    ).withColumn(
        "earnings",
        F.round(F.col("driver_pay") + F.col("tips"), 2),
    )


# ─── External datasets ─────────────────────────────────────────────────────


def clean_weather(df: DataFrame) -> DataFrame:
    """Impute null preciptype as 'None' and select the columns we need.

    IQR outlier removal is intentionally NOT applied — the observed NYC
    feels-like temperature and precipitation ranges are plausible per
    domain knowledge, and dropping rows would punch holes in the join.
    """
    return df.withColumn(
        "preciptype",
        F.when(F.col("preciptype").isNull(), "None").otherwise(F.col("preciptype")),
    ).select("datetime", "feelslike", "precip", "preciptype")


def join_weather(trips: DataFrame, weather: DataFrame) -> DataFrame:
    """Left-join weather onto trips on pickup_hour == datetime."""
    return trips.join(weather, trips["pickup_hour"] == weather["datetime"], "left")


def add_holiday_flag(df: DataFrame, holiday_dates: list) -> DataFrame:
    """Add a binary `is_public_holiday` column from a list of holiday dates."""
    return df.withColumn(
        "is_public_holiday",
        F.col("pickup_date").isin(holiday_dates).cast("int"),
    )
