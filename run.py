"""
run.py
-------
Runs the full Python-side pipeline in order:
    1. python/data_analysis.py          (preprocessing + features + EDA)
    2. python/forecasting_validation.py (ARIMA vs Holt-Winters, real numbers)
    3. python/inventory_analysis.py     (inventory gap + recommendation)

This does NOT run RapidMiner - RapidMiner is run separately, inside
Altair AI Studio, using the processes in rapidminer/. See
rapidminer/HOW_TO_BUILD.md for exact build instructions.

After this finishes, launch the GUI with:
    python python/gui.py
"""

import subprocess
import sys

STEPS = [
    ("Phase 2: Preprocessing, Feature Engineering & EDA", "python/data_analysis.py"),
    ("Phase 3: Forecasting (ARIMA vs Holt-Winters) & Evaluation", "python/forecasting_validation.py"),
    ("Phase 4: Inventory Gap Analysis", "python/inventory_analysis.py"),
]


def main():
    for label, script in STEPS:
        print("\n" + "=" * 65)
        print(label)
        print("=" * 65)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStep failed: {script}. Stopping pipeline.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE. All outputs saved to outputs/")
    print("Now run:  python python/gui.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
