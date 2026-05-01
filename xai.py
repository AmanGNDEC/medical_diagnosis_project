"""
xai.py  ── STEP 3 : Explainable AI (XAI) Layer
================================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant

What this file does
-------------------
1. explain_prediction(disease, symptoms)
       → For each symptom, computes its log-likelihood contribution.
       → Returns sorted list so we know WHICH symptoms drove the diagnosis.

2. what_if_analysis(symptoms, add=[], remove=[])
       → Re-runs prediction with modified symptom set.
       → Enables the "What-If" toggle in the UI.

3. plot_xai_chart(contributions, disease)
       → Horizontal bar chart showing symptom contributions.

XAI Concept (for viva)
-----------------------
Naive Bayes is inherently interpretable because the total score is a SUM:
    score(D) = log P(D) + Σᵢ contribution(symptom_i, D)

where:
    contribution(symptom_i = 1, D) = log P(symptom_i=1 | D)
    contribution(symptom_i = 0, D) = log P(symptom_i=0 | D)

Positive contribution → symptom SUPPORTS the diagnosis.
Negative contribution → symptom ARGUES AGAINST it.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Project modules ───────────────────────────────────────────────────────────
from model import get_model_and_symptoms, predict_top_diseases

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════════
#  CORE XAI FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def explain_prediction(disease: str, symptoms_present: list) -> list:
    """
    Computes the log-likelihood contribution of each symptom
    for a given predicted disease.

    Parameters
    ----------
    disease          : the disease whose prediction we want to explain
    symptoms_present : list of symptom name strings that ARE present

    Returns
    -------
    List of (symptom_name, contribution_float, is_present_bool) tuples
    sorted by |contribution| descending (most impactful first).

    XAI Math
    --------
    For symptom sᵢ:
        if sᵢ is present:  contribution = log P(sᵢ=1 | disease)  [positive → supports]
        if sᵢ absent   :  contribution = log P(sᵢ=0 | disease)  [negative → against]

    Larger magnitude = more evidence for/against the disease.
    """
    model, symptom_cols = get_model_and_symptoms()

    # Find the index of the disease in the model's class list
    if disease not in model.classes_:
        raise ValueError(f"Unknown disease: {disease}")
    d_idx = list(model.classes_).index(disease)

    contributions = []
    for sym_idx, sym_name in enumerate(symptom_cols):
        is_present = sym_name in symptoms_present

        if is_present:
            # symptom present → log P(s=1 | disease)
            contrib = float(model.log_prob_1_[d_idx, sym_idx])
        else:
            # symptom absent → log P(s=0 | disease)
            contrib = float(model.log_prob_0_[d_idx, sym_idx])

        contributions.append((sym_name, contrib, is_present))

    # Sort by absolute value (most influential first)
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions


def get_top_contributions(disease: str, symptoms_present: list,
                          top_n: int = 10) -> list:
    """
    Returns only the top-N most impactful symptom contributions.
    Used by the UI to avoid cluttering the chart.
    """
    all_contribs = explain_prediction(disease, symptoms_present)
    # Prefer to show present symptoms first (they are the positive signals)
    present   = [(s, c, p) for s, c, p in all_contribs if p]
    absent    = [(s, c, p) for s, c, p in all_contribs if not p]
    combined  = present[:top_n // 2 + 1] + absent[:top_n // 2]
    combined.sort(key=lambda t: abs(t[1]), reverse=True)
    return combined[:top_n]


# ════════════════════════════════════════════════════════════════════════════
#  WHAT-IF ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def what_if_analysis(symptoms: list, add: list = None, remove: list = None) -> list:
    """
    Re-runs the top-disease prediction after adding or removing symptoms.

    Parameters
    ----------
    symptoms : current list of symptom names (strings)
    add      : symptoms to add
    remove   : symptoms to remove

    Returns
    -------
    List of (disease, probability) tuples — same format as predict_top_diseases()

    Use case (UI "What-If" toggle)
    ───────────────────────────────
    User unchecks 'fever' → remove=['fever'] → confidence for Malaria drops.
    User checks  'rash'  → add=['rash']    → Dengue probability rises.
    """
    add    = add    or []
    remove = remove or []

    modified = set(symptoms)
    modified.update(add)
    modified.difference_update(remove)

    return predict_top_diseases(list(modified), k=5)


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
def plot_xai_chart(contributions: list, disease: str,
                   ax=None, save_path: str = None) -> None:
    """
    Horizontal bar chart showing symptom log-likelihood contributions.

    • Green bars (rightward) = symptom supports the diagnosis (present)
    • Red   bars (leftward)  = symptom argues against / is absent

    Parameters
    ----------
    contributions : output of get_top_contributions()
    disease       : disease name (used for title)
    ax            : optional matplotlib Axes (for embedding in tkinter)
    save_path     : if given, saves the figure to this path
    """
    names    = [c[0].replace("_", " ").title() for c in contributions]
    values   = [c[1] for c in contributions]
    present  = [c[2] for c in contributions]
    colors   = ["#2ecc71" if p else "#e74c3c" for p in present]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", height=0.6)

    # Add value labels
    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width() + (0.05 if val >= 0 else -0.05)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Log-likelihood contribution", fontsize=10)
    ax.set_title(f"🔍 XAI — Symptom Contributions for: {disease}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(min(values) - 0.5, max(values) + 0.5)

    # Legend
    patch_pos = mpatches.Patch(color="#2ecc71", label="Symptom present  → supports")
    patch_neg = mpatches.Patch(color="#e74c3c", label="Symptom absent   → against")
    ax.legend(handles=[patch_pos, patch_neg], fontsize=9, loc="lower right")

    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  ✅  Saved: {os.path.basename(save_path)}")
        plt.show()


def plot_what_if_comparison(before: list, after: list,
                             added: list, removed: list) -> None:
    """
    Side-by-side comparison of top-5 predictions before and after
    a what-if modification.
    """
    # Gather all diseases that appear in either list
    all_diseases = list({d for d, _ in before + after})
    prob_before  = {d: p for d, p in before}
    prob_after   = {d: p for d, p in after}

    x      = np.arange(len(all_diseases))
    width  = 0.38
    colors = ["#3498db", "#e67e22"]

    fig, ax = plt.subplots(figsize=(11, 5))
    b_vals  = [prob_before.get(d, 0) * 100 for d in all_diseases]
    a_vals  = [prob_after.get(d,  0) * 100 for d in all_diseases]

    ax.bar(x - width / 2, b_vals, width, label="Before",
           color=colors[0], alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, a_vals, width, label="After",
           color=colors[1], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_diseases, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Confidence (%)", fontsize=11)

    change_str = ""
    if added:
        change_str += f"Added: {', '.join(added)}  "
    if removed:
        change_str += f"Removed: {', '.join(removed)}"
    ax.set_title(f"🔄 What-If Analysis\n{change_str}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    fname = os.path.join(PROJECT_DIR, "plot_what_if.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  ✅  Saved: {os.path.basename(fname)}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN  ─  run standalone to verify Step 3
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║   AI Medical Diagnosis Assistant  ─  STEP 3: XAI         ║")
    print("╚" + "═" * 60 + "╝")

    # ── Sample patient symptoms ───────────────────────────────────────────
    sample_symptoms = ["high_fever", "chills", "sweating", "headache", "body_ache"]
    print(f"\n  Patient symptoms : {sample_symptoms}")

    # ── Step 1: predict top diseases ─────────────────────────────────────
    top = predict_top_diseases(sample_symptoms, k=5)
    print("\n  Top-5 Predicted Diseases:")
    for rank, (disease, prob) in enumerate(top, 1):
        bar = "▓" * int(prob * 40)
        print(f"    {rank}. {disease:<35} {prob*100:6.2f}%  {bar}")

    # ── Step 2: explain top disease ───────────────────────────────────────
    top_disease = top[0][0]
    print(f"\n  Explaining prediction for: {top_disease}")
    contribs = get_top_contributions(top_disease, sample_symptoms, top_n=12)
    print("\n  Symptom Contributions (log-likelihood):")
    for sym, val, present in contribs:
        tag  = "✅ present" if present else "❌ absent"
        bar  = ("+" if val > 0 else "") + "█" * int(abs(val) * 5)
        print(f"    {sym:<35} {val:+.4f}  {tag:<12}  {bar}")

    # ── Step 3: plot XAI chart ────────────────────────────────────────────
    plot_xai_chart(
        contribs, top_disease,
        save_path=os.path.join(PROJECT_DIR, "plot_xai.png")
    )

    # ── Step 4: what-if demo ──────────────────────────────────────────────
    print("\n  What-If: remove 'chills', add 'rash' + 'joint_pain'")
    before = predict_top_diseases(sample_symptoms, k=5)
    after  = what_if_analysis(sample_symptoms,
                               add=["rash", "joint_pain"],
                               remove=["chills"])
    print("\n  Before modification:")
    for d, p in before:
        print(f"    {d:<35} {p*100:6.2f}%")
    print("\n  After  modification:")
    for d, p in after:
        print(f"    {d:<35} {p*100:6.2f}%")

    plot_what_if_comparison(before, after,
                             added=["rash", "joint_pain"],
                             removed=["chills"])

    print("\n" + "═" * 62)
    print("  ✅  STEP 3 COMPLETE.  Ready to move to STEP 4 → app.py")
    print("═" * 62 + "\n")
