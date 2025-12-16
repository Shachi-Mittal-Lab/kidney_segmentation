
#!/usr/bin/env python3
"""
Plot horizontal boxplots of fib_score grouped by pathologist_score.

Expected CSV columns:
- file_name
- fib_score
- pathologist_score (can be float like 1.0, 2.0, 3.0)

Usage:
    python plot_fib_boxplot.py path/to/data.csv [--out output.png]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Boxplot of fib_score by pathologist_score")
    parser.add_argument("csv", help="Path to the input CSV file")
    parser.add_argument("--out", default="fib_score_boxplot.png", help="Output image file name (PNG)")
    args = parser.parse_args()

    # --- Load data ---
    df = pd.read_csv(args.csv)

    # --- Basic validation ---
    required_cols = {"file_name", "fib_score", "pathologist_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure fib_score is numeric
    df["fib_score"] = pd.to_numeric(df["fib_score"], errors="coerce")

    # Read pathologist_score as numeric (float) then convert to int where valid
    # Valid values are 0.0, 1.0, 2.0, 3.0 -> 0, 1, 2, 3. Everything else (including NaN) -> 'None'
    df["pathologist_score"] = pd.to_numeric(df["pathologist_score"], errors="coerce")
    def to_category(x):
        if pd.isna(x):
            return "None"
        # Handle floats cleanly (including small FP noise)
        xi = int(round(float(x)))
        if xi in (0, 1, 2, 3):
            return str(xi)
        return "None"

    df["pathologist_score_cat"] = df["pathologist_score"].apply(to_category)

    # Drop rows with missing fib_score, since they can't be plotted
    df = df.dropna(subset=["fib_score"])

    # Define category order to ensure five boxes appear in this order
    cat_order = ["0", "1", "2", "3", "None"]
    df["pathologist_score_cat"] = pd.Categorical(
        df["pathologist_score_cat"], categories=cat_order, ordered=True
    )

    # --- Plot ---
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(
        data=df,
        x="fib_score",
        y="pathologist_score_cat",
        order=cat_order,
        orient="h",
        showfliers=True,
        width=0.6
    )

    # Optional: overlay jittered points for added visibility of individual samples
    sns.stripplot(
        data=df,
        x="fib_score",
        y="pathologist_score_cat",
        order=cat_order,
        orient="h",
        color="black",
        alpha=0.35,
        size=3,
        jitter=True
    )

    ax.set_xlabel("Fib Score")
    ax.set_ylabel("Pathologist Score")
    ax.set_title("Fib Score Distribution by Pathologist Score")

    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=300)
    print(f"Saved box plot to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
