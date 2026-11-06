"""
inventory_analysis.py
-----------------------
Phase 4: Inventory Gap Analysis (simple decision-support layer).

Formula (as specified):
    Inventory Gap = Forecast Demand - Current Inventory

Interpretation:
    Gap > 0   -> potential shortage, additional inventory may be required
    Gap ~ 0   -> inventory approximately sufficient
    Gap < 0   -> inventory sufficient / possibly excessive

"Current Inventory" is taken as the total Inventory Level (summed across
all Store x Product combinations) on the LAST date present in the raw
dataset - i.e. the most recent real stock snapshot we have.

This is intentionally simple: a decision-support signal, not a
supply-chain optimization system.
"""

import json
import pandas as pd

RAW_PATH = "data/retail_demand.csv"
FORECAST_PATH = "outputs/forecast_results.csv"
OUT_DIR = "outputs"

# A gap within +/- this tolerance (as a fraction of current inventory)
# is treated as "approximately sufficient" rather than a hard shortage
# or surplus flag.
TOLERANCE_FRACTION = 0.05


def get_current_inventory(raw_path: str = RAW_PATH) -> tuple[float, str]:
    """Total inventory across all Store x Product rows on the most recent date."""
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    current_inventory = df.loc[df["Date"] == last_date, "Inventory Level"].sum()
    return float(current_inventory), str(last_date.date())


def compute_inventory_gap(forecast_path: str = FORECAST_PATH):
    forecast_df = pd.read_csv(forecast_path, parse_dates=["Date"])
    current_inventory, snapshot_date = get_current_inventory()

    rows = []
    for _, r in forecast_df.iterrows():
        forecast_demand = r["Forecasted_Demand"]
        gap = forecast_demand - current_inventory
        tolerance = TOLERANCE_FRACTION * current_inventory

        if gap > tolerance:
            recommendation = f"Potential shortage of approximately {gap:.0f} units - consider reordering."
        elif gap < -tolerance:
            recommendation = f"Inventory appears sufficient, with a surplus of approximately {abs(gap):.0f} units."
        else:
            recommendation = "Inventory is approximately sufficient for forecasted demand."

        rows.append({
            "Date": r["Date"].date(),
            "Forecast_Demand": round(forecast_demand, 2),
            "Current_Inventory_Snapshot": round(current_inventory, 2),
            "Inventory_Gap": round(gap, 2),
            "Recommendation": recommendation,
        })

    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(f"{OUT_DIR}/inventory_gap_analysis.csv", index=False)

    # Next-day summary (the single headline figure the GUI shows)
    next_day = gap_df.iloc[0]
    avg_forecast_demand = float(forecast_df["Forecasted_Demand"].mean())
    # Days of Supply: how many days current stock actually covers at the
    # forecasted average daily demand rate. This matters because a large
    # absolute "surplus" can still mean very few days of real cover -
    # comparing total stock to a single day's forecast alone can be
    # misleading, so this is reported alongside the raw gap.
    days_of_supply = round(current_inventory / avg_forecast_demand, 1) if avg_forecast_demand > 0 else None

    summary = {
        "inventory_snapshot_date": snapshot_date,
        "current_inventory": round(current_inventory, 2),
        "next_day_forecast_demand": float(next_day["Forecast_Demand"]),
        "next_day_inventory_gap": float(next_day["Inventory_Gap"]),
        "next_day_recommendation": next_day["Recommendation"],
        "avg_forecast_daily_demand": round(avg_forecast_demand, 2),
        "days_of_supply_remaining": days_of_supply,
    }
    with open(f"{OUT_DIR}/inventory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Current inventory snapshot (as of {snapshot_date}): {current_inventory:.0f} units (summed across all stores/products)")
    print(f"\nInventory gap for next {len(gap_df)} forecast days (head):")
    print(gap_df.head())
    print(f"\nNext-day recommendation: {next_day['Recommendation']}")
    print(f"Days of supply remaining (current stock / avg forecast daily demand): {days_of_supply}")
    print(f"\nSaved: outputs/inventory_gap_analysis.csv, outputs/inventory_summary.json")

    return gap_df, summary


if __name__ == "__main__":
    compute_inventory_gap()
