"""Feature assembly, temporal train/test split, and model training/evaluation."""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


FEATURE_COLUMNS = [
    "license_vec",
    "trip_miles_standardised",
    "day_vec",
    "hour_vec",
    "PULocation_vec",
    "feelslike_standardised",
    "precip_standardised",
    "preciptype_vec",
    "is_public_holiday",
]

LABEL_COLUMN = "earnings"


def assemble_features(
    df: DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
) -> DataFrame:
    """Combine feature columns into a single 'features' vector for ML."""
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    return assembler.transform(df).select("features", label_column, "month")


def temporal_split(
    df: DataFrame,
    train_months: tuple[int, int] = (5, 10),
    test_month: int = 11,
) -> tuple[DataFrame, DataFrame]:
    """Temporal holdout split — train on May–Oct 2023, test on November 2023.

    Avoids leakage from random shuffling and mirrors how the model would
    be evaluated against unseen future months in production.
    """
    train = df.filter(col("month").between(*train_months))
    test = df.filter(col("month") == test_month)
    return train, test


def train_linear_regression(
    train: DataFrame,
    reg_param: float = 0.0,
    elastic_net_param: float = 0.0,
    label_column: str = LABEL_COLUMN,
) -> LinearRegressionModel:
    """Fit a (possibly regularised) linear regression on the training data.

    Set `elastic_net_param=1.0` with `reg_param>0` for Lasso (L1).
    """
    lr = LinearRegression(
        featuresCol="features",
        labelCol=label_column,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
    )
    return lr.fit(train)


def evaluate(predictions: DataFrame, label_column: str = LABEL_COLUMN) -> dict[str, float]:
    """Return {'rmse': ..., 'r2': ...} for the given predictions DataFrame."""
    metrics: dict[str, float] = {}
    for metric in ("rmse", "r2"):
        evaluator = RegressionEvaluator(
            labelCol=label_column,
            predictionCol="prediction",
            metricName=metric,
        )
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics
