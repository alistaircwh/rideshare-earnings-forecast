"""Data I/O helpers for raw FHVHV, external CSVs, and curated Parquet."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark import CURATED_SCHEMA


def load_raw_fhvhv(spark: SparkSession, path: str = "../data/raw") -> DataFrame:
    """Load the raw FHVHV Parquet files (one per month) from `path`."""
    return spark.read.parquet(path)


def load_weather(spark: SparkSession, path: str) -> DataFrame:
    """Load the NYC hourly weather CSV with header + inferred schema."""
    return spark.read.csv(path, header=True, inferSchema=True)


def load_holidays(spark: SparkSession, *paths: str) -> DataFrame:
    """Load and union one or more US public-holiday CSVs."""
    dfs = [spark.read.csv(p, header=True, inferSchema=True) for p in paths]
    out = dfs[0]
    for df in dfs[1:]:
        out = out.union(df)
    return out


def load_curated(
    spark: SparkSession,
    glob_path: str = "../data/curated/chunk_*.parquet",
) -> DataFrame:
    """Load curated month-partitioned Parquet files using the canonical schema."""
    return spark.read.schema(CURATED_SCHEMA).parquet(glob_path)


def write_curated(df: DataFrame, output_dir: str, months: list[int]) -> None:
    """Write the curated DataFrame as one Parquet chunk per month."""
    for month in months:
        chunk = df.filter(F.col("month") == month)
        chunk.write.mode("append").parquet(f"{output_dir}/chunk_{month}.parquet")
