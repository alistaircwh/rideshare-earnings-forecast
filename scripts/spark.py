"""Spark session factory and curated-data schema.

Centralises the Spark configuration used across the preprocessing,
analysis, and modelling notebooks so the same tuning lives in one place.
"""

from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def get_spark(
    app_name: str = "Rideshare_Analysis",
    executor_memory: str = "10g",
    shuffle_partitions: int = 200,
) -> SparkSession:
    """Build a local Spark session tuned for the FHVHV workload."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.repl.eagerEval.enabled", True)
        .config("spark.sql.parquet.cacheMetadata", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "Etc/UTC")
        .getOrCreate()
    )


# Schema of the curated month-partitioned parquet files written by the
# preprocessing notebook. Defining it explicitly lets the modelling and
# analysis notebooks load via spark.read.schema(...) without re-inferring
# (faster, and guarantees vector columns survive the round trip).
CURATED_SCHEMA = StructType(
    [
        StructField("hvfhs_license_num", StringType(), True),
        StructField("PULocationID", IntegerType(), True),
        StructField("trip_miles", DoubleType(), True),
        StructField("license_vec", VectorUDT(), True),
        StructField("day_of_week", IntegerType(), True),
        StructField("hour_of_day", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("day_vec", VectorUDT(), True),
        StructField("hour_vec", VectorUDT(), True),
        StructField("PULocation_vec", VectorUDT(), True),
        StructField("trip_miles_standardised", DoubleType(), True),
        StructField("trip_time_standardised", DoubleType(), True),
        StructField("earnings_per_hour", DoubleType(), True),
        StructField("earnings", DoubleType(), True),
        StructField("feelslike", DoubleType(), True),
        StructField("feelslike_standardised", DoubleType(), True),
        StructField("precip_standardised", DoubleType(), True),
        StructField("preciptype_vec", VectorUDT(), True),
        StructField("is_public_holiday", IntegerType(), True),
    ]
)
