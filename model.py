"""
model.py  ── STEP 2 : Naive Bayes Classifier
=============================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant

What this file does
-------------------
1. Implements Naive Bayes FROM SCRATCH using Bayes' rule + Laplace smoothing.
2. Also trains sklearn GaussianNB for comparison.
3. Evaluates both: accuracy, precision, recall, F1, confusion matrix.
4. Exposes predict_top_diseases(symptoms_list) → top-5 diseases + probabilities.

Key ML Concepts (for viva)
--------------------------
• Bayes' Theorem:  P(Disease | Symptoms) ∝ P(Symptoms | Disease) × P(Disease)
• Naive assumption: symptoms are conditionally independent given the disease
• Laplace smoothing: add 1 to every count to avoid zero-probability problem
• Log probabilities: multiply small numbers → add logs to avoid underflow
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ── Project module ────────────────────────────────────────────────────────────
from data_prep import get_prepared_data

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM NAIVE BAYES  (Bernoulli / Binary features)
# ════════════════════════════════════════════════════════════════════════════
class NaiveBayesScratch:
    """
    Binary (Bernoulli) Naive Bayes implemented from scratch.

    Bayes' Theorem
    ──────────────
        P(D | S₁…Sₙ) ∝  P(D) × ∏ P(Sᵢ | D)

    With Laplace smoothing (α = 1):
        P(Sᵢ = 1 | D) = (count(Sᵢ=1 in D) + 1) / (count(D) + 2)
        P(Sᵢ = 0 | D) = 1 − P(Sᵢ=1 | D)

    We work in log-space to avoid floating-point underflow:
        log P(D|S) ∝ log P(D) + Σ log P(Sᵢ | D)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha          # Laplace smoothing parameter
        self.classes_     = None   # unique disease names
        self.log_prior_   = None   # log P(disease)   shape: (n_classes,)
        self.log_prob_1_  = None   # log P(s=1|disease) shape: (n_classes, n_features)
        self.log_prob_0_  = None   # log P(s=0|disease)

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate class priors and conditional probabilities from training data.

        Parameters
        ----------
        X : binary matrix  (n_samples × n_features)
        y : label array    (n_samples,)
        """
        self.classes_ = np.unique(y)
        n_classes   = len(self.classes_)
        n_features  = X.shape[1]
        n_samples   = X.shape[0]

        # Allocate probability tables
        self.log_prior_  = np.zeros(n_classes)
        self.log_prob_1_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]          # all rows belonging to this disease
            n_cls = X_cls.shape[0]

            # ── Prior: P(disease) = count(disease) / total_samples
            self.log_prior_[idx] = np.log(n_cls / n_samples)

            # ── Likelihood with Laplace smoothing
            # P(symptom_i = 1 | disease) = (Σ X[:,i] + α) / (n_cls + 2α)
            count_present = X_cls.sum(axis=0)     # how many times symptom present
            self.log_prob_1_[idx] = np.log(
                (count_present + self.alpha) / (n_cls + 2 * self.alpha)
            )

        # P(s=0|d) = 1 - P(s=1|d)  →  in log-space: log(1 - exp(log_prob_1))
        self.log_prob_0_ = np.log(1 - np.exp(self.log_prob_1_))
        return self

    # ── Inference ────────────────────────────────────────────────────────────
    def _compute_log_posterior(self, x: np.ndarray) -> np.ndarray:
        """
        For a single sample x (binary vector), compute log P(disease | x)
        for every disease class.

        log P(D|x) ∝ log P(D)
                     + Σ [xᵢ · log P(sᵢ=1|D)  +  (1-xᵢ) · log P(sᵢ=0|D)]
        """
        # present symptoms contribute log_prob_1, absent ones log_prob_0
        log_likelihoods = (x * self.log_prob_1_) + ((1 - x) * self.log_prob_0_)
        return self.log_prior_ + log_likelihoods.sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely disease for each row in X."""
        log_posteriors = np.array([self._compute_log_posterior(x) for x in X])
        return self.classes_[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalised probability for each class.
        We convert log-posteriors → probabilities via softmax:
            prob[k] = exp(log_post[k]) / Σ exp(log_post[j])
        """
        log_posteriors = np.array([self._compute_log_posterior(x) for x in X])
        # Subtract max for numerical stability before exp
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors)
        return posteriors / posteriors.sum(axis=1, keepdims=True)

    # ── Convenience: top-k predictions ──────────────────────────────────────
    def predict_top_k(self, x: np.ndarray, k: int = 5):
        """Return top-k (disease, probability) pairs for a single sample."""
        proba = self.predict_proba(x.reshape(1, -1))[0]
        top_idx = np.argsort(proba)[::-1][:k]
        return [(self.classes_[i], float(proba[i])) for i in top_idx]


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> dict:
    """
    Prints and returns classification metrics.
    Metrics explained (for viva):
    ─────────────────────────────
    • Accuracy  : fraction of all predictions that are correct
    • Precision : of all predicted as disease D, how many actually had D?
    • Recall    : of all who actually had D, how many did we detect?
    • F1 Score  : harmonic mean of precision and recall (balanced metric)
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  ── {model_name} ───────────────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    return {"name": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, classes, model_name: str) -> None:
    """Normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalize

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_norm,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        fmt=".2f",
        annot=True,
        annot_kws={"size": 7},
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Normalised count"},
    )
    ax.set_xlabel("Predicted Disease", fontsize=11)
    ax.set_ylabel("Actual Disease", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    fname = os.path.join(PROJECT_DIR, f"plot_cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  ✅  Saved: {os.path.basename(fname)}")


def plot_metrics_comparison(results: list) -> None:
    """Side-by-side bar chart comparing both models on 4 metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#4C72B0", "#DD8452"]

    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res["name"],
                      color=colors[i], edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison: Custom NB vs sklearn BernoulliNB",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    fname = os.path.join(PROJECT_DIR, "plot_model_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  ✅  Saved: {os.path.basename(fname)}")


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════
# These are loaded once and reused by xai.py and app.py
_model       = None   # custom NaiveBayesScratch instance
_symptom_cols = None  # ordered list of symptom names


def _ensure_model_loaded():
    """Lazy-load: train model on first call, cache for subsequent calls."""
    global _model, _symptom_cols
    if _model is not None:
        return
    X_train, X_test, y_train, y_test, symptom_cols, df = get_prepared_data()
    _symptom_cols = symptom_cols
    _model = NaiveBayesScratch(alpha=1.0)
    _model.fit(X_train, y_train)


def predict_top_diseases(symptoms_list: list, k: int = 5) -> list:
    """
    Public function used by app.py and xai.py.

    Parameters
    ----------
    symptoms_list : list of symptom name strings that ARE present
    k             : how many top diseases to return

    Returns
    -------
    list of (disease_name, probability_float) tuples, sorted desc by prob
    """
    _ensure_model_loaded()

    # Build binary feature vector
    x = np.zeros(len(_symptom_cols))
    for s in symptoms_list:
        if s in _symptom_cols:
            x[_symptom_cols.index(s)] = 1.0

    return _model.predict_top_k(x, k=k)


def get_model_and_symptoms():
    """Returns (trained_model, symptom_cols) for use by xai.py and app.py."""
    _ensure_model_loaded()
    return _model, _symptom_cols


# ════════════════════════════════════════════════════════════════════════════
#  MAIN  ─  run standalone to verify Step 2
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║   AI Medical Diagnosis Assistant  ─  STEP 2: MODEL       ║")
    print("╚" + "═" * 60 + "╝")

    # ── Load data ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, symptom_cols, df = get_prepared_data()
    classes = sorted(df["Disease"].unique())

    # ── Train custom model ────────────────────────────────────────────────
    print("\n  Training Custom Naive Bayes (from scratch) …")
    custom_nb = NaiveBayesScratch(alpha=1.0)
    custom_nb.fit(X_train, y_train)
    res_custom = evaluate_model(custom_nb, X_test, y_test, "Custom NaiveBayes")

    # ── Train sklearn BernoulliNB for comparison ──────────────────────────
    print("\n  Training sklearn BernoulliNB …")
    sklearn_nb = BernoulliNB(alpha=1.0)
    sklearn_nb.fit(X_train, y_train)
    res_sklearn = evaluate_model(sklearn_nb, X_test, y_test, "sklearn BernoulliNB")

    # ── Confusion matrices ────────────────────────────────────────────────
    print("\n  Generating confusion matrices …")
    plot_confusion_matrix(y_test, res_custom["y_pred"],  classes, "Custom NaiveBayes")
    plot_confusion_matrix(y_test, res_sklearn["y_pred"], classes, "sklearn BernoulliNB")

    # ── Comparison bar chart ──────────────────────────────────────────────
    plot_metrics_comparison([res_custom, res_sklearn])

    # ── Demo: predict for sample symptoms ─────────────────────────────────
    sample = ["high_fever", "chills", "sweating", "headache", "body_ache"]
    print(f"\n  Demo — symptoms: {sample}")
    print("  Top-5 predicted diseases:")
    results = custom_nb.predict_top_k(
        np.array([1.0 if s in sample else 0.0 for s in symptom_cols]),
        k=5
    )
    for rank, (disease, prob) in enumerate(results, 1):
        bar = "▓" * int(prob * 40)
        print(f"    {rank}. {disease:<35} {prob*100:6.2f}%  {bar}")

    print("\n" + "═" * 62)
    print("  ✅  STEP 2 COMPLETE.  Ready to move to STEP 3 → xai.py")
    print("═" * 62 + "\n")
