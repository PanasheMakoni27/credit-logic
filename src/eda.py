"""Simple Exploratory Data Analysis (EDA) script.

Loads the original `Loan.csv` and writes a set of EDA artifacts to
`reports/eda/`: missingness, numeric summary, categorical counts,
correlation heatmap, histograms, boxplots and a small outliers CSV.

Run:
    python -u src/eda.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "Loan.csv"
OUT_DIR = ROOT / "reports" / "eda"


def run_eda(csv_path: Path = CSV_PATH, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Place your dataset at project root as Loan.csv")
        return 1

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Rows: {n}, Columns: {len(df.columns)}")

    # 1) Missingness
    miss = df.isnull().sum().rename("missing").to_frame()
    miss["pct_missing"] = miss["missing"] / n
    miss.sort_values("pct_missing", ascending=False, inplace=True)
    miss_path = out_dir / "missingness.csv"
    miss.to_csv(miss_path)
    print(f"Saved missingness -> {miss_path}")

    # 2) Numeric summary
    num = df.select_dtypes(include=[np.number])
    num_desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    num_desc_path = out_dir / "numeric_summary.csv"
    num_desc.to_csv(num_desc_path)
    print(f"Saved numeric summary -> {num_desc_path}")

    # 3) Categorical counts (top values)
    cat = df.select_dtypes(include=["object", "category"]).copy()
    cat_counts = []
    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False).head(50).rename_axis(c).reset_index(name="count")
        vc["column"] = c
        cat_counts.append(vc)
    if cat_counts:
        cat_all = pd.concat(cat_counts, ignore_index=True)
        cat_path = out_dir / "categorical_counts_top50.csv"
        cat_all.to_csv(cat_path, index=False)
        print(f"Saved categorical top counts -> {cat_path}")
    else:
        print("No categorical columns found.")

    # 4) Correlation heatmap (numeric)
    if not num.empty:
        corr = num.corr()
        corr_path = out_dir / "correlation.csv"
        corr.to_csv(corr_path)

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Numeric Correlation Matrix")
        plt.tight_layout()
        heatmap_png = out_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_png, bbox_inches="tight")
        plt.close()
        print(f"Saved correlation heatmap -> {heatmap_png}")
    else:
        print("No numeric columns for correlation.")

    # 5) Histograms for numeric features (multi-plot)
    if not num.empty:
        try:
            ax = num.hist(figsize=(12, int(max(2, len(num.columns) / 3))), bins=30)
            # When num.hist returns array of axes, save figure from first axis's figure
            fig = ax.flatten()[0].get_figure()
            hist_png = out_dir / "numeric_histograms.png"
            fig.savefig(hist_png, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved numeric histograms -> {hist_png}")
        except Exception:
            # fallback: individual histograms
            for c in num.columns:
                plt.figure()
                num[c].hist(bins=50)
                plt.title(c)
                plt.xlabel(c)
                plt.ylabel("count")
                p = out_dir / f"hist_{c}.png"
                plt.savefig(p, bbox_inches="tight")
                plt.close()
    
    # 6) Boxplot for numeric features
    if not num.empty:
        try:
            fig2 = plt.figure(figsize=(12, 6))
            num.boxplot(rot=45)
            bp_png = out_dir / "numeric_boxplots.png"
            fig2.savefig(bp_png, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved numeric boxplots -> {bp_png}")
        except Exception:
            pass

    # 7) Simple outlier finder (mean + 4*std)
    outliers = []
    for c in num.columns:
        col = num[c]
        mean = col.mean()
        std = col.std()
        if std == 0 or np.isnan(std):
            continue
        thr = mean + 4 * std
        mask = col > thr
        if mask.any():
            rows = df.loc[mask, :].copy()
            rows["outlier_col"] = c
            rows["outlier_threshold"] = thr
            outliers.append(rows)
    if outliers:
        out_df = pd.concat(outliers, ignore_index=True)
        out_path = out_dir / "outliers_mean4std.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved outliers -> {out_path}")
    else:
        print("No extreme outliers found (mean + 4*std).")

    # 8) Small sample for quick inspection
    sample_path = out_dir / "sample_head.csv"
    df.head(200).to_csv(sample_path, index=False)
    print(f"Saved sample head -> {sample_path}")

    print("EDA complete.")
    return 0


if __name__ == "__main__":
    run_eda()
