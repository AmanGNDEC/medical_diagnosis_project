"""
data_prep.py  ── STEP 1 : Dataset & Preprocessing
===================================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant
Concept  : Naive Bayes + Explainable AI (XAI)

What this file does
-------------------
1. Loads (or auto-generates) the disease-symptom CSV dataset.
2. Converts it to a binary symptom matrix  (1 = present, 0 = absent).
3. Shows disease distribution and top-symptom frequency bar charts.
4. Performs 80 / 20 train-test split.
5. Exports  X_train, X_test, y_train, y_test, symptom_cols  so every
   other module can import them without re-reading the file.

Run standalone:
    python data_prep.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
# NOTE: Do NOT force a backend here — app.py sets TkAgg once for the whole app.
# When running this file standalone, matplotlib will auto-detect the right backend.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, "disease_symptom_dataset.csv")
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 20 % held-out test set


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1-A  ─  Load / Generate Dataset
# ════════════════════════════════════════════════════════════════════════════
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Loads the CSV.  If it doesn't exist, auto-generates it by calling
    dataset_generator.py so the project works without a Kaggle account.
    """
    if not os.path.exists(path):
        print("⚠️  Dataset not found.  Auto-generating …")
        gen_path = os.path.join(PROJECT_DIR, "dataset_generator.py")
        if not os.path.exists(gen_path):
            sys.exit("❌  dataset_generator.py missing.  Please re-download the project.")
        # Run generator as a sub-process (safe, only touches local files)
        import subprocess
        subprocess.run([sys.executable, gen_path], check=True)

    df = pd.read_csv(path)
    print(f"\n✅  Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1-B  ─  Inspect & Clean
# ════════════════════════════════════════════════════════════════════════════
def inspect_dataset(df: pd.DataFrame) -> None:
    """Prints shape, disease counts, missing-value report, and sample rows."""
    print("\n" + "═" * 60)
    print("  DATASET INSPECTION")
    print("═" * 60)
    print(f"  Shape          : {df.shape}")
    print(f"  Disease column : {df['Disease'].nunique()} unique diseases")
    print(f"  Symptom cols   : {df.shape[1] - 1}")

    print("\n  ── Disease distribution ──────────────────────────────────")
    counts = df["Disease"].value_counts()
    for disease, cnt in counts.items():
        bar = "█" * (cnt // 5)          # mini ASCII bar  (30 blocks ≈ 150)
        print(f"  {disease:<35} {cnt:>4}  {bar}")

    missing = df.isnull().sum().sum()
    print(f"\n  Missing values : {missing}  {'✅ None!' if missing == 0 else '⚠️  Handle before training'}")

    print("\n  ── First 3 rows (Disease + first 10 symptoms) ───────────")
    symptom_cols = [c for c in df.columns if c != "Disease"]
    print(df[["Disease"] + symptom_cols[:10]].head(3).to_string(index=False))
    print()


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1-C  ─  Build Binary Feature Matrix
# ════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(df: pd.DataFrame):
    """
    ML concept: Feature Engineering
    ────────────────────────────
    We already have a binary matrix (1/0) from the CSV.
    X = symptom columns (features)
    y = Disease labels  (target)

    NOTE: We use .to_numpy() instead of .values to guarantee a plain
    NumPy ndarray on ALL pandas versions (including 2.x with PyArrow
    backend used on Streamlit Cloud / Python 3.12+).
    """
    symptom_cols = [c for c in df.columns if c != "Disease"]
    X = df[symptom_cols].to_numpy().astype(np.float64)  # always a numpy ndarray
    y = df["Disease"].to_numpy(dtype=str)               # always a numpy ndarray
    return X, y, symptom_cols


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1-D  ─  Train / Test Split
# ════════════════════════════════════════════════════════════════════════════
def split_data(X: np.ndarray, y: np.ndarray):
    """
    ML concept: Train-Test Split
    ────────────────────────────
    80 % of data → training (model learns from this)
    20 % of data → testing  (we evaluate on UNSEEN samples)

    stratify=y ensures every disease class has the same ratio in
    both splits (important for imbalanced datasets).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y             # ← keeps class balance identical in both splits
    )
    print("  ── Train / Test split ────────────────────────────────────")
    print(f"  Train set : {X_train.shape}  ({(1-TEST_SIZE)*100:.0f} %)")
    print(f"  Test  set : {X_test.shape}  ({TEST_SIZE*100:.0f} %)")
    print(f"  Unique classes in train : {len(set(y_train))}")
    print(f"  Unique classes in test  : {len(set(y_test))}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1-E  ─  Visualisations
# ════════════════════════════════════════════════════════════════════════════
def plot_disease_distribution(df: pd.DataFrame) -> None:
    """Bar chart: how many patient records exist per disease."""
    counts = df["Disease"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", len(counts))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=palette[::-1], edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    ax.set_xlabel("Number of Patient Records", fontsize=11)
    ax.set_title("📊  Disease Distribution in Dataset", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, counts.max() + 20)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "plot_disease_distribution.png"), dpi=150)
    plt.show()
    print("  ✅  Saved: plot_disease_distribution.png")


def plot_top_symptoms(df: pd.DataFrame, top_n: int = 20) -> None:
    """Bar chart: which symptoms appear most frequently across all records."""
    symptom_cols = [c for c in df.columns if c != "Disease"]
    freq = df[symptom_cols].sum().sort_values(ascending=False).head(top_n)
    readable = [s.replace("_", " ").title() for s in freq.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("magma", top_n)
    bars = ax.bar(readable, freq.values, color=palette, edgecolor="white")

    for bar, val in zip(bars, freq.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(val)), ha="center", fontsize=7.5)

    ax.set_xticklabels(readable, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Total Occurrences", fontsize=11)
    ax.set_title(f"🔬  Top {top_n} Most Frequent Symptoms", fontsize=14, fontweight="bold", pad=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "plot_top_symptoms.png"), dpi=150)
    plt.show()
    print("  ✅  Saved: plot_top_symptoms.png")


def plot_symptom_heatmap(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Heatmap: average symptom prevalence per disease.
    Darker cell = symptom more commonly associated with that disease.
    Great for understanding which symptoms are 'discriminative'.
    """
    symptom_cols = [c for c in df.columns if c != "Disease"]

    # Pick the top-N most frequent symptoms for readability
    top_symptoms = df[symptom_cols].sum().sort_values(ascending=False).head(top_n).index.tolist()

    # Group by disease and compute mean presence (= probability P(s|d))
    heatmap_data = df.groupby("Disease")[top_symptoms].mean()

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        ax=ax,
        cbar_kws={"label": "P(symptom | disease)"},
    )
    # Pretty column labels
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in top_symptoms],
        rotation=40, ha="right", fontsize=8
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.set_title("🔥  Symptom–Disease Heatmap  (P(symptom | disease))",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "plot_heatmap.png"), dpi=150)
    plt.show()
    print("  ✅  Saved: plot_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  ─  imported by model.py, xai.py, app.py
# ════════════════════════════════════════════════════════════════════════════
def get_prepared_data():
    """
    One-stop function used by other modules.
    Returns
    -------
    X_train, X_test, y_train, y_test  : numpy arrays
    symptom_cols                       : list of symptom column names
    df                                 : full DataFrame (for reference)
    """
    df            = load_dataset()
    X, y, symptom_cols = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, symptom_cols, df


# ════════════════════════════════════════════════════════════════════════════
#  MAIN  ─  run this file directly to verify Step 1
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   AI Medical Diagnosis Assistant  ─  STEP 1: DATA PREP  ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. Load ───────────────────────────────────────────────────────────
    df = load_dataset()

    # ── 2. Inspect ────────────────────────────────────────────────────────
    inspect_dataset(df)

    # ── 3. Feature matrix ─────────────────────────────────────────────────
    X, y, symptom_cols = build_feature_matrix(df)
    print("\n  ── Feature matrix ────────────────────────────────────────")
    print(f"  X shape        : {X.shape}  (samples × symptoms)")
    print(f"  y shape        : {y.shape}  (disease labels)")
    print(f"  Sample features: {symptom_cols[:8]} …")

    # ── 4. Split ──────────────────────────────────────────────────────────
    print()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── 5. Visualise ──────────────────────────────────────────────────────
    print("\n  ── Generating plots … ────────────────────────────────────")
    plot_disease_distribution(df)
    plot_top_symptoms(df)
    plot_symptom_heatmap(df)

    print("\n" + "═" * 60)
    print("  ✅  STEP 1 COMPLETE.  Ready to move to STEP 2 → model.py")
    print("═" * 60 + "\n")
