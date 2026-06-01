"""Visualisation helpers — geospatial choropleths and earnings plots."""

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame


NYC_CENTER = [40.73, -73.74]


# ─── Geospatial ────────────────────────────────────────────────────────────


def load_taxi_zone_geodata(shapefile_path: str, lookup_path: str) -> gpd.GeoDataFrame:
    """Load the taxi-zone shapefile + lookup, reprojected to lat/long."""
    shapes = gpd.read_file(shapefile_path)
    shapes["geometry"] = shapes["geometry"].to_crs(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    )
    zones = pd.read_csv(lookup_path)
    return gpd.GeoDataFrame(pd.merge(zones, shapes, on="LocationID", how="inner"))


def build_zone_geojson(gdf: gpd.GeoDataFrame) -> str:
    """GeoJSON string of one polygon per taxi-zone LocationID."""
    return gdf[["LocationID", "geometry"]].drop_duplicates("LocationID").to_json()


def sample_to_pandas(
    df: DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Spark → pandas via column selection + random sampling."""
    return df.select(columns).sample(fraction, seed=seed).toPandas()


def build_earnings_map(
    zone_stats: pd.DataFrame,
    geojson: str,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    legend: str,
    output_path: str,
) -> folium.Map:
    """Choropleth of `value_col` per pickup zone, with top-3 zones pinned."""
    m = folium.Map(location=NYC_CENTER, tiles="OpenStreetMap", zoom_start=10)

    folium.Choropleth(
        geo_data=geojson,
        name="choropleth",
        data=zone_stats.reset_index(),
        columns=["PULocationID", value_col],
        key_on="properties.LocationID",
        fill_color="YlOrRd",
        nan_fill_color="black",
        legend_name=legend,
    ).add_to(m)

    merged = gdf.merge(
        zone_stats, left_on="LocationID", right_on="PULocationID", how="inner"
    )
    top_3 = merged.sort_values(value_col, ascending=False).head(3)
    top_3["centroid"] = top_3["geometry"].apply(lambda g: (g.centroid.y, g.centroid.x))
    for zone, coord, value in top_3[["Zone", "centroid", value_col]].values:
        m.add_child(folium.Marker(location=coord, popup=f"{zone} -> {value:.2f}"))

    m.save(output_path)
    return m


def build_demand_map(demand: pd.DataFrame, geojson: str, output_path: str) -> folium.Map:
    """Choropleth of trip-count demand per pickup zone."""
    m = folium.Map(location=NYC_CENTER, tiles="OpenStreetMap", zoom_start=10)

    folium.Choropleth(
        geo_data=geojson,
        name="choropleth",
        data=demand.reset_index(),
        columns=["PULocationID", "trip_count"],
        key_on="properties.LocationID",
        fill_color="YlOrRd",
        nan_fill_color="black",
        legend_name="Trip Count (Demand)",
    ).add_to(m)

    m.save(output_path)
    return m


# ─── Static plots ──────────────────────────────────────────────────────────


def plot_earnings_by_service(sample_pd: pd.DataFrame, output_path: str) -> None:
    """Boxplot of log earnings/hour for Uber (HV0003) vs Lyft (HV0005)."""
    uber = sample_pd[sample_pd["hvfhs_license_num"] == "HV0003"]
    lyft = sample_pd[sample_pd["hvfhs_license_num"] == "HV0005"]

    combined = pd.concat([uber.assign(service="Uber"), lyft.assign(service="Lyft")])
    combined["log_earnings_per_hour"] = np.log1p(combined["earnings_per_hour"])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="service", y="log_earnings_per_hour", data=combined, palette="Set3")
    plt.title("Boxplot: Earnings per Hour by Service (Uber vs Lyft)")
    plt.xlabel("Service")
    plt.ylabel("Log Earnings per Hour (USD)")
    plt.savefig(output_path)
    plt.show()


def plot_earnings_by_hour(sample_pd: pd.DataFrame, output_path: str) -> None:
    """Line plot of average earnings/hour by hour-of-day, one line per day-of-week."""
    hourly = (
        sample_pd.groupby(["hour_of_day", "day_of_week"])["earnings_per_hour"]
        .mean()
        .reset_index()
    )
    pivot = hourly.pivot(
        index="hour_of_day", columns="day_of_week", values="earnings_per_hour"
    )
    day_order = [1, 2, 3, 4, 5, 6, 7]
    pivot = pivot[day_order]

    plt.figure(figsize=(14, 8))
    for day in day_order:
        plt.plot(pivot.index, pivot[day], label=day)

    plt.title("Average Earnings per Hour by Hour of Day (for each Day of the Week)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Earnings per Hour (USD)")
    plt.legend(
        title="Day of the Week",
        labels=[
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ],
    )
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()
