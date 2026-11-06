"""
forecasting_validation.py
---------------------------
SUPPORTING / VERIFICATION SCRIPT.

RapidMiner is the primary forecasting environment for this project
(see rapidminer/forecasting_process.rmp and the Phase 3 instructions).
This script reproduces the same ARIMA vs Holt-Winters comparison in
Python, using the SAME chronological validation logic (60-day holdout,
30-day forecast horizon), so you can:

  1. Cross-check the numbers RapidMiner gives you
  2. Have a working forecast even before you've built/run the
     RapidMiner process
  3. Feed real numbers into the Tkinter GUI (gui.py) and
     inventory_analysis.py

Every number below is computed live from outputs/daily_demand_aggregated.csv
(created by data_analysis.py) - nothing is hard-coded.

Run with:
    python python/forecasting_validation.py
(after running data_analysis.py first)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DAILY_PATH = "outputs/daily_demand_aggregated.csv"
OUT_DIR = "outputs"
HOLDOUT_DAYS = 60   # chronological validation window
HORIZON_DAYS = 30   # final future forecast horizon


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape(y_true, y_pred)}


def load_series() -> pd.Series:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["Date"])
    ts = daily.set_index("Date")["Total_Demand"]
    ts.index.freq = "D"
    return ts


def run_arima(train: pd.Series, horizon: int):
    """ARIMA(1,1,1) - same simple starting configuration used in the
    RapidMiner ARIMA operator (p=1, d=1, q=1)."""
    model = ARIMA(train, order=(1, 1, 1))
    fit = model.fit()
    return np.array(fit.forecast(steps=horizon))


def run_holt_winters(train: pd.Series, horizon: int, seasonal_periods: int = 7):
    """
    Holt-Winters with weekly seasonality (period=7), since the data is
    daily. Falls back to non-seasonal if there isn't enough history for
    two full seasonal cycles.
    """
    use_seasonal = len(train) >= 2 * seasonal_periods
    if use_seasonal:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=seasonal_periods
        )
    else:
        model = ExponentialSmoothing(train, trend="add", seasonal=None)
    fit = model.fit()
    return np.array(fit.forecast(horizon))


def run_validation():
    ts = load_series()
    train, test = ts.iloc[:-HOLDOUT_DAYS], ts.iloc[-HOLDOUT_DAYS:]

    print(f"Training window: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Test (holdout) window: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
    print()

    results = {}

    pred_arima = run_arima(train, HOLDOUT_DAYS)
    results["ARIMA"] = evaluate(test.values, pred_arima)

    pred_hw = run_holt_winters(train, HOLDOUT_DAYS)
    results["Holt-Winters"] = evaluate(test.values, pred_hw)

    best_model = min(results, key=lambda m: results[m]["RMSE"])

    print("MODEL COMPARISON (Python verification, on real data):")
    comp_rows = []
    for name, metrics in results.items():
        row = {"Model": name, **{k: round(v, 3) for k, v in metrics.items()}}
        comp_rows.append(row)
        print(f"  {name}: MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%")
    print(f"\nBest model (lowest RMSE): {best_model}")

    comp_df = pd.DataFrame(comp_rows)
    comp_df["Selected"] = comp_df["Model"] == best_model
    comp_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

    # ---- Refit best model on FULL history, forecast HORIZON_DAYS ahead ----
    if best_model == "ARIMA":
        future_pred = run_arima(ts, HORIZON_DAYS)
    else:
        future_pred = run_holt_winters(ts, HORIZON_DAYS)

    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Demand": np.round(future_pred, 2)})
    forecast_df.to_csv(f"{OUT_DIR}/forecast_results.csv", index=False)

    # ---- Plots ----
    plt.figure(figsize=(10, 4.5))
    plt.plot(train.index[-30:], train.values[-30:], label="Train (recent)", color="gray")
    plt.plot(test.index, test.values, label="Actual", color="black", marker="o", markersize=3)
    plt.plot(test.index, pred_arima, "--", label="ARIMA forecast", linewidth=1.5)
    plt.plot(test.index, pred_hw, "--", label="Holt-Winters forecast", linewidth=1.5)
    plt.title(f"Actual vs Predicted - Holdout Window (Best: {best_model})")
    plt.xlabel("Date")
    plt.ylabel("Total Demand")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=130)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(ts.index[-60:], ts.values[-60:], label="Recent history", color="gray")
    plt.plot(future_dates, future_pred, label=f"Future forecast ({best_model})", color="#C0392B", marker="o", markersize=3)
    plt.title(f"{HORIZON_DAYS}-Day Future Demand Forecast")
    plt.xlabel("Date")
    plt.ylabel("Total Demand")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/forecast_plot.png", dpi=130)
    plt.close()

    # ---- Save a small summary JSON for the GUI to read ----
    summary = {
        "best_model": best_model,
        "comparison": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in results.items()},
        "avg_daily_demand_forecast": round(float(np.mean(future_pred)), 2),
        "next_day_forecast": round(float(future_pred[0]), 2),
        "horizon_days": HORIZON_DAYS,
        "forecast_period": f"{future_dates[0].date()} to {future_dates[-1].date()}",
    }
    with open(f"{OUT_DIR}/forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: outputs/model_comparison.csv, outputs/forecast_results.csv,")
    print(f"       outputs/model_comparison.png, outputs/forecast_plot.png,")
    print(f"       outputs/forecast_summary.json")

    return results, best_model, forecast_df


if __name__ == "__main__":
    run_validation()
