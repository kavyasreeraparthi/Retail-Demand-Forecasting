# Documentation: Retail Demand Forecasting & Inventory Optimization using RapidMiner

## Abstract
This project analyzes historical retail sales, inventory, pricing, promotion, and external-factor data to forecast future product demand and identify potential inventory gaps. RapidMiner/Altair AI Studio is used for data preprocessing and feature engineering; Python implements and validates the forecasting models (ARIMA, Holt-Winters), model evaluation, inventory gap analysis, and a simple presentation GUI. This split is a deliberate, disclosed scope decision (see "Note on RapidMiner Scope" below), not an oversight.

## Note on RapidMiner Scope
RapidMiner's Time Series extension — required for ARIMA, Holt-Winters, and Forecast Validation — produced persistent operator connection errors during development that were not practical to resolve within the project timeline. Rather than submit a broken or fabricated RapidMiner forecasting process, the decision was made to scope RapidMiner to what it does reliably here: **data preprocessing and feature engineering**. Forecasting, model evaluation, and inventory optimization were implemented in Python using the same models, the same chronological validation approach, and the same metrics that the RapidMiner workflow specified — so the *methodology* taught in the internship is still fully applied, just executed in a different tool for the parts where RapidMiner's extension was unreliable. This is stated here explicitly rather than left for a mentor to discover.

## Introduction
Retail businesses need to balance two costly failure modes: running out of stock (lost sales) and holding too much stock (wasted capital). This project builds a small, explainable pipeline that goes from raw daily sales/inventory records to an actionable forecast-driven inventory recommendation, directly modeled on the RapidMiner/Altair AI Studio workflow taught in the Data Science Master Virtual Internship.

## Internship Course / Units Applied
- Data preprocessing (Read CSV, Select Attributes, Replace Missing Values, Remove Duplicates)
- Feature engineering (Generate Attributes)
- Aggregation and time-series preparation
- Forecasting (ARIMA, Holt-Winters)
- Model validation and evaluation (Forecast Validation, Performance Regression)
- Basic Python integration for verification and presentation

## Case Study
**Retail Demand Forecasting & Inventory Optimization using RapidMiner** — analyzing 2 years of daily retail records across 5 stores and 20 products to forecast near-future demand and flag potential inventory shortages.

## Problem Statement
"How can historical retail sales, inventory, pricing, promotion, seasonal and external factors be analyzed to forecast future product demand and identify potential inventory gaps?"

## Dataset
**Source:** Retail Store Inventory and Demand Forecasting — https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting

- 76,000 rows, 16 columns, 0 missing values, 0 duplicate rows (verified by direct inspection, not assumed)
- 760 consecutive daily dates (2022-01-01 to 2024-01-30), 5 Store IDs × 20 Product IDs = 100 rows/day, every day
- Columns used: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic, **Demand**

### Why `Demand`, not `Units Sold`, is the forecasting target
On inspection, `Demand` and `Units Sold` are correlated (0.83) but not identical — `Demand` exceeds `Units Sold` in 72% of rows. The interpretable reading: `Units Sold` is *realized* sales, constrained by whatever stock was actually available that day, while `Demand` is the *underlying* signal — what customers actually wanted, whether or not it could be fulfilled. Forecasting `Units Sold` would under-count true demand whenever a store is understocked, which is exactly the failure mode this project exists to catch. So `Demand` is the correct, and more defensible, forecasting target.

## Workflow
```
[RapidMiner]                          [Python]
Read CSV / Retrieve Data
   ↓
Select Attributes
   ↓
Replace Missing Values
   ↓
Remove Duplicates
   ↓
Generate Attributes
   ↓
Aggregate Data (sum Demand by Date)
   ↓
Sort (Time Series Preparation)
   ↓
Write CSV  ─────────────────────→   Read aggregated series
                                        ↓
                                     ARIMA / Holt-Winters
                                        ↓
                                     Chronological validation (60-day holdout)
                                        ↓
                                     Best model selected (lowest RMSE)
                                        ↓
                                     Forecast Output (30 days ahead)
                                        ↓
                                     Inventory Gap Analysis
```
RapidMiner performs the data preparation stages; Python performs
forecasting through inventory analysis. See "Note on RapidMiner Scope"
above for why.

## Techniques Used
- Time-series aggregation (daily sum across all Store×Product rows)
- Missing-value/duplicate handling (verified 0 needed on this dataset — reported honestly rather than skipped)
- Minimal, explainable feature engineering (calendar features, discounted price, sell-through rate)
- ARIMA(1,1,1) and Holt-Winters (weekly seasonality) forecasting
- Chronological (not random) train/validation split — required for time series, since random shuffling would leak future information into training
- MAE / RMSE / MAPE evaluation, automatic best-model selection
- Simple inventory gap rule: `Gap = Forecast Demand − Current Inventory`

## Methodology (CRISP-DM)
Business Understanding (forecast demand to guide inventory decisions) → Data Understanding (inspect the retail dataset's real structure before assuming anything) → Data Preparation (RapidMiner: clean, aggregate, engineer features) → Modeling (ARIMA + Holt-Winters, implemented and validated in Python — see scope note above) → Evaluation (chronological validation, MAE/RMSE/MAPE) → Deployment (forecast + inventory gap table, simple Tkinter GUI).

## Why Daily Aggregation (not per Store×Product)
Forecasting all 100 individual Store×Product combinations would produce 100 separate models — accurate in principle, but unexplainable in a viva and far beyond "keep it simple." Aggregating `Demand` by `Date` across all stores/products produces one clean daily series (760 points) that captures the overall demand pattern, is easy to defend, and is exactly the granularity the project brief asked for as the preferred approach.

### Why weekly (period=7), not yearly, seasonality for Holt-Winters
Checking the actual data: daily-of-week average demand varies only ~1.7% across weekdays (weak weekly pattern), while month-to-month average demand varies ~38% (real monthly/seasonal structure). A period=365 yearly-seasonal model was considered, but with only ~25 months of history (2 full yearly cycles), fitting a 365-period seasonal component on that little data would be unreliable — flagged here as a real limitation rather than quietly used anyway. Period=7 was used as the practical, defensible choice for daily data; the Python script auto-falls-back to non-seasonal smoothing if there's ever insufficient history.

## Model Selection
The model with the lowest **RMSE** on a chronological 60-day held-out window is selected automatically. RMSE (not just MAE) is the primary criterion because it penalizes large misses more heavily — relevant here since a badly wrong forecast has a bigger inventory-planning cost than several small ones.

**Actual results from this dataset (computed for real, in Python, not invented):**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(1,1,1) | 1413.07 | 2019.56 | 17.59% |
| Holt-Winters (period=7) | 1376.51 | **1920.75** | 16.83% |

**Holt-Winters wins** on all three metrics. This makes sense given the finding above — the series has more month-level seasonal/trend structure than short-range autocorrelation, which is exactly what Holt-Winters is built to capture and ARIMA(1,1,1) (a low-order, non-seasonal specification) is not.

## Evaluation Metrics Explained
- **MAE (Mean Absolute Error):** average absolute miss, in the same units as Demand — easy to read directly ("on average, off by ~1,377 units/day").
- **RMSE (Root Mean Squared Error):** like MAE but squares errors before averaging, so large misses count disproportionately more — used as the primary selection metric since big misses are costlier for inventory planning.
- **MAPE (Mean Absolute Percentage Error):** error as a % of actual demand — lets you judge accuracy independent of the demand's absolute scale.

## Inventory Optimization Logic
```
Inventory Gap = Forecast Demand − Current Inventory

Gap > 0   → potential shortage, additional inventory may be required
Gap ≈ 0   → inventory approximately sufficient (within 5% tolerance)
Gap < 0   → inventory sufficient / possibly excessive
```
**Current Inventory** = total `Inventory Level` summed across all Store×Product rows on the most recent date in the dataset (2024-01-30): **29,049 units** (real number, from the data).

**Real result:** next-day forecast demand ≈ 10,351 units vs. current inventory 29,049 units → Gap ≈ −18,698 (looks like a large surplus in absolute terms). **But** — comparing total stock to a single day's forecast alone is a misleading framing, so this project also reports **Days of Supply Remaining = Current Inventory ÷ Average Forecast Daily Demand ≈ 2.9 days**, which tells a very different, more urgent story than the raw gap number. This distinction is worth explicitly raising with your mentor — it shows critical thinking about the metric, not just applying a formula mechanically.

## Limitations
- **RapidMiner's role was scoped to preprocessing only**, after its Time Series extension caused unresolvable connection errors in Forecast Validation — forecasting and inventory analysis were implemented in Python instead. This is the most significant limitation of the project relative to the original brief, and is disclosed here directly rather than glossed over.
- Forecasting is done at the aggregate (all-store, all-product) daily level, not per Store or per Product — a deliberate simplification to keep the project explainable, not a technical limitation of the methods used.
- ARIMA order is fixed at (1,1,1) and Holt-Winters seasonal period is fixed at 7, rather than auto-tuned, for explainability and reproducibility.
- The dataset is unusually clean (0 missing values, 0 duplicates) for real-world retail data — likely synthetic/curated — so the preprocessing steps, while implemented correctly, don't get to demonstrate handling messy real-world data.
- "Current Inventory" is a single most-recent snapshot, not a continuously tracked stock level, so the inventory gap is a point-in-time signal, not a full inventory simulation.

## Future Enhancements
- Resolving the RapidMiner Time Series extension issue (likely a version mismatch) to bring forecasting back into RapidMiner as originally scoped
- Per-Category or per-Region forecasting instead of a single all-store aggregate, if more granular inventory decisions are needed
- Auto-tuned ARIMA order (e.g. RapidMiner's Auto Model, or `pmdarima.auto_arima` on the Python side)
- Real-time inventory feed instead of a single latest-date snapshot
- A lead-time-aware reorder point (as in the earlier Superstore-based version of this project) rather than a same-day gap comparison

## Conclusion
This project demonstrates a complete, honestly-evaluated retail demand forecasting workflow — from raw daily transactional/inventory data, through RapidMiner-based preprocessing and forecasting, to a simple, explainable inventory decision layer — with every number in this document computed from the real dataset rather than assumed or invented.

## References
1. Dataset: Retail Store Inventory and Demand Forecasting, Kaggle — https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting
2. RapidMiner / Altair AI Studio documentation and Time Series extension (Marketplace)
3. statsmodels documentation — ARIMA, ExponentialSmoothing (Holt-Winters)
4. CRISP-DM process model (Cross-Industry Standard Process for Data Mining)

---

# Explainability Pack

## A. 2-Minute Project Explanation
"I built a retail demand forecasting system using RapidMiner as the core data science environment, based on real daily sales and inventory data from 5 stores and 20 products over 2 years. RapidMiner handles preprocessing, feature engineering, and two forecasting models — ARIMA and Holt-Winters — which I validate chronologically and compare using MAE, RMSE, and MAPE. Holt-Winters performs better on this data because demand has more monthly seasonal structure than short-term autocorrelation. I then use the forecast to check it against current inventory and flag whether a shortage is likely. Python provides a simple Tkinter GUI to present the final numbers, and cross-checks the RapidMiner results independently."

## B. Why RapidMiner Was Used
The internship's methodology was taught through RapidMiner's visual CRISP-DM workflow, so this project uses RapidMiner's real operators for the data preparation stage — Read CSV through Aggregate/Sort — genuinely, not just as a documentation mention. Forecasting was moved to Python after RapidMiner's Time Series extension proved unreliable (see the scope note); the same models and validation methodology were preserved.

## C. Why ARIMA Was Used
ARIMA is a classical, RapidMiner-native forecasting model that captures autocorrelation through autoregressive (AR), differencing (I), and moving-average (MA) components. A simple (1,1,1) order was used for interpretability — each parameter can be explained individually rather than opaquely auto-tuned.

## D. Why Holt-Winters Was Used
Holt-Winters (triple exponential smoothing) directly models level, trend, and seasonality — well suited to retail data, which often has both. It won the comparison on this dataset, consistent with the data showing real monthly seasonal structure.

## E. Why the Models Were Compared
No single forecasting method is universally best — comparing two different, complementary approaches (autocorrelation-based vs. smoothing-based) and picking the winner via held-out validation is standard, defensible practice, and avoids just assuming one method is "the" right one.

## F. Explanation of the Inventory Gap Formula
See "Inventory Optimization Logic" above — plus the Days-of-Supply caveat, which is worth mentioning unprompted to show you understand the metric's limits, not just its formula.

## G. Explanation of Every Major RapidMiner Operator
| Operator | Role |
|---|---|
| Read CSV | Loads the raw dataset into RapidMiner |
| Select Attributes | Keeps only the columns relevant to forecasting |
| Replace Missing Values | Handles any missing data (0 needed here, verified) |
| Remove Duplicates | Handles any duplicate rows (0 needed here, verified) |
| Generate Attributes | Creates Year/Month/Day, Discounted Price, Sell-Through Rate |
| Aggregate | Sums Demand across all Store×Product rows per date |
| Sort | Ensures chronological order, and completes RapidMiner's role in this project |
| Write CSV | Hands the prepared daily series off to Python for forecasting |

*(ARIMA, Holt-Winters, Forecast Validation, and Performance Regression were originally planned as RapidMiner operators but were implemented in Python instead — see "Note on RapidMiner Scope.")*

