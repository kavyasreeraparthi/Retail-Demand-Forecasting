"""
data_analysis.py
------------------
Supporting Python analysis for the RapidMiner-based Retail Demand
Forecasting project.

IMPORTANT: This script does NOT perform the core forecasting work.
Its jobs are:
  1. Load and inspect the raw dataset (mirrors what "Read CSV" +
     inspection would show inside RapidMiner)
  2. Produce the demand analysis charts required by the case study
  3. Aggregate demand by date (same aggregation the RapidMiner
     "Aggregate" operator performs) and save it as a CSV that the
     RapidMiner process can also read, and that forecasting_validation.py
     uses for the Python-side verification forecast.

Run with:
    python python/data_analysis.py
(run from the project root, so the relative paths below resolve)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/retail_demand.csv"
OUT_DIR = "outputs"


def load_and_inspect(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Unique Store IDs: {df['Store ID'].nunique()}")
    print(f"Unique Product IDs: {df['Product ID'].nunique()}")
    print(f"Categories: {sorted(df['Category'].unique())}")
    print(f"Regions: {sorted(df['Region'].unique())}")
    print(f"Missing values total: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print()

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, explainable feature engineering (mirrors the RapidMiner
    'Generate Attributes' operator).
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.day_name()

    df["Discounted_Price"] = df["Price"] * (1 - df["Discount"] / 100)

    # Sell-through rate = Units Sold / Inventory Level, with safe
    # handling for Inventory Level == 0 (division by zero).
    df["Sell_Through_Rate"] = np.where(
        df["Inventory Level"] > 0,
        df["Units Sold"] / df["Inventory Level"],
        np.nan,
    )

    return df


def aggregate_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Demand operator equivalent: sum Demand across all
    Store x Product combinations for each calendar date, producing a
    single overall daily demand time series.
    """
    daily = df.groupby("Date", as_index=False)["Demand"].sum()
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily = daily.rename(columns={"Demand": "Total_Demand"})
    return daily


def plot_demand_distribution(df: pd.DataFrame):
    plt.figure(figsize=(7, 4.5))
    plt.hist(df["Demand"], bins=40, color="#4C72B0", edgecolor="white")
    plt.title("Distribution of Demand (row-level, Store x Product x Day)")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_01_demand_distribution.png", dpi=130)
    plt.close()


def plot_demand_by_category(df: pd.DataFrame):
    totals = df.groupby("Category")["Demand"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 4.5))
    plt.bar(totals.index, totals.values, color="#4C72B0")
    plt.title("Total Demand by Category")
    plt.ylabel("Total Demand")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_02_demand_by_category.png", dpi=130)
    plt.close()


def plot_demand_by_region(df: pd.DataFrame):
    totals = df.groupby("Region")["Demand"].sum().sort_values(ascending=False)
    plt.figure(figsize=(6, 4.5))
    plt.bar(totals.index, totals.values, color="#55A868")
    plt.title("Total Demand by Region")
    plt.ylabel("Total Demand")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_03_demand_by_region.png", dpi=130)
    plt.close()


def plot_promotion_vs_demand(df: pd.DataFrame):
    means = df.groupby("Promotion")["Demand"].mean()
    labels = ["No Promotion (0)", "Promotion (1)"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, [means.get(0, 0), means.get(1, 0)], color=["#C0392B", "#27AE60"])
    plt.title("Average Demand: Promotion vs No Promotion")
    plt.ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_04_promotion_vs_demand.png", dpi=130)
    plt.close()


def plot_inventory_vs_units_sold(df: pd.DataFrame):
    sample = df.sample(min(3000, len(df)), random_state=42)
    plt.figure(figsize=(7, 5))
    plt.scatter(sample["Inventory Level"], sample["Units Sold"], alpha=0.3, s=10, color="#4C72B0")
    plt.title("Inventory Level vs Units Sold")
    plt.xlabel("Inventory Level")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_05_inventory_vs_unitssold.png", dpi=130)
    plt.close()


def plot_weather_vs_demand(df: pd.DataFrame):
    means = df.groupby("Weather Condition")["Demand"].mean().sort_values(ascending=False)
    plt.figure(figsize=(7, 4.5))
    plt.bar(means.index, means.values, color="#8172B2")
    plt.title("Average Demand by Weather Condition")
    plt.ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_06_weather_vs_demand.png", dpi=130)
    plt.close()


def plot_seasonality_vs_demand(df: pd.DataFrame):
    means = df.groupby("Seasonality")["Demand"].mean()
    order = ["Winter", "Spring", "Summer", "Autumn"]
    means = means.reindex([s for s in order if s in means.index])
    plt.figure(figsize=(6, 4.5))
    plt.bar(means.index, means.values, color="#CCB974")
    plt.title("Average Demand by Season")
    plt.ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_07_seasonality_vs_demand.png", dpi=130)
    plt.close()


def plot_monthly_trend(daily: pd.DataFrame):
    m = daily.copy()
    m["Month"] = m["Date"].dt.to_period("M")
    monthly = m.groupby("Month")["Total_Demand"].sum()
    plt.figure(figsize=(11, 4.5))
    plt.plot(monthly.index.astype(str), monthly.values, marker="o", color="#2E5090")
    plt.title("Monthly Total Demand Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Demand")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chart_08_monthly_demand_trend.png", dpi=130)
    plt.close()


def plot_daily_trend(daily: pd.DataFrame):
    plt.figure(figsize=(11, 4.5))
    plt.plot(daily["Date"], daily["Total_Demand"], color="#2E5090", linewidth=0.9)
    plt.title("Daily Total Demand Trend (2022-01-01 to 2024-01-30)")
    plt.xlabel("Date")
    plt.ylabel("Total Demand")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/demand_trend.png", dpi=130)
    plt.close()


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_inspect()
    featured = add_features(df)
    daily = aggregate_daily_demand(df)

    plot_demand_distribution(df)
    plot_demand_by_category(df)
    plot_demand_by_region(df)
    plot_promotion_vs_demand(df)
    plot_inventory_vs_units_sold(df)
    plot_weather_vs_demand(df)
    plot_seasonality_vs_demand(df)
    plot_monthly_trend(daily)
    plot_daily_trend(daily)

    featured.to_csv(f"{OUT_DIR}/featured_dataset.csv", index=False)
    daily.to_csv(f"{OUT_DIR}/daily_demand_aggregated.csv", index=False)

    print("EDA complete. Charts saved to outputs/. Daily aggregated series saved to")
    print(f"  {OUT_DIR}/daily_demand_aggregated.csv")
    print("This is the file the RapidMiner forecasting process should also use")
    print("(or you can let RapidMiner do its own Read CSV + Aggregate on the raw file).")

    return df, featured, daily


if __name__ == "__main__":
    run_eda()
