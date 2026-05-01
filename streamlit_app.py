"""
streamlit_app.py  ──  Web-Deployable Version
=============================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant

Deploy FREE to Streamlit Cloud:
  1. Push this entire folder to a GitHub repo
  2. Go to https://streamlit.io/cloud
  3. Sign in with GitHub → "New App"
  4. Select your repo, set main file = streamlit_app.py
  5. Click Deploy → live in ~2 minutes!

Run locally:
  pip install streamlit
  streamlit run streamlit_app.py
"""

import io
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ── Project modules ──────────────────────────────────────────────────────────
from data_prep import get_prepared_data
from model     import get_model_and_symptoms, predict_top_diseases
from xai       import get_top_contributions

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  (dark medical theme)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* Dark base */
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
  [data-testid="stHeader"]           { background: #161b22; border-bottom: 1px solid #30363d; }

  /* Text */
  h1,h2,h3,h4,p,div,label,span { color: #e6edf3 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 10px; padding: 12px !important;
  }

  /* Sidebar inputs */
  .stSelectbox label, .stNumberInput label, .stCheckbox label { color: #8b949e !important; }

  /* Divider */
  hr { border-color: #30363d; }

  /* Disclaimer */
  .disclaimer {
    background: #1c2128; border: 1px solid #d29922;
    border-radius: 8px; padding: 10px 16px;
    font-size: 0.82rem; color: #d29922 !important;
    margin-top: 20px;
  }

  /* Disease card */
  .disease-card {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
  }
  .rank-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 5px; font-weight: 700; font-size: 0.9rem;
    color: #fff; margin-right: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS  (run once, cached across sessions)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠  Loading AI model…")
def load_model():
    model, symptoms = get_model_and_symptoms()
    X_train, X_test, y_train, y_test, _, _ = get_prepared_data()
    accuracy = float(np.mean(model.predict(X_test) == y_test))
    return model, symptoms, accuracy


MODEL, SYMPTOM_COLS, TEST_ACCURACY = load_model()


# ════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════
def make_xai_figure(contributions: list, disease: str) -> plt.Figure:
    """Returns a dark-themed matplotlib Figure for the XAI bar chart."""
    names   = [c[0].replace("_", " ").title() for c in contributions]
    values  = [c[1] for c in contributions]
    present = [c[2] for c in contributions]
    colors  = ["#2ecc71" if p else "#e74c3c" for p in present]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5)),
                           facecolor="#1c2128")
    ax.set_facecolor("#1c2128")

    bars = ax.barh(names[::-1], values[::-1],
                   color=colors[::-1], edgecolor="#30363d", height=0.65)

    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width() + (0.03 if val >= 0 else -0.03)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="#e6edf3")

    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Log-likelihood contribution", fontsize=9, color="#8b949e")
    ax.set_title(f"Why → {disease}", fontsize=12, fontweight="bold",
                 color="#e6edf3", pad=10)

    patch_pos = mpatches.Patch(color="#2ecc71", label="Present  → supports diagnosis")
    patch_neg = mpatches.Patch(color="#e74c3c", label="Absent   → argues against")
    ax.legend(handles=[patch_pos, patch_neg], fontsize=8,
              facecolor="#21262d", labelcolor="#e6edf3", edgecolor="#30363d")

    ax.tick_params(colors="#e6edf3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout(pad=1.5)
    return fig


def confidence_color(prob: float) -> str:
    if prob >= 0.70: return "#f85149"
    if prob >= 0.40: return "#d29922"
    return "#3fb950"


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  ─ Patient info + Symptom checklist
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 AI Medical Diagnosis")
    st.markdown(f"**Model Accuracy:** `{TEST_ACCURACY*100:.1f}%`")
    st.divider()

    # Patient info
    st.markdown("### 👤 Patient Information")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
    with col_b:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.divider()

    # Symptom search + checklist
    st.markdown("### ☑ Select Symptoms")
    search = st.text_input("🔍 Search symptoms…", placeholder="Type to filter")

    filtered = [s for s in SYMPTOM_COLS if search.lower() in s.lower()] \
               if search else SYMPTOM_COLS

    # Use session state to track selections
    if "selections" not in st.session_state:
        st.session_state.selections = set()

    selected_symptoms = []
    for sym in filtered:
        label = sym.replace("_", " ").title()
        checked = st.checkbox(label, value=(sym in st.session_state.selections),
                              key=f"cb_{sym}")
        if checked:
            selected_symptoms.append(sym)
            st.session_state.selections.add(sym)
        else:
            st.session_state.selections.discard(sym)

    st.divider()

    # Buttons
    col1, col2 = st.columns(2)
    diagnose_clicked = col1.button("🔬 DIAGNOSE", type="primary",
                                   use_container_width=True)
    if col2.button("🗑 Clear", use_container_width=True):
        st.session_state.selections = set()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏥 AI Medical Diagnosis Assistant")
st.markdown("*B.Tech Mini-Project — Naive Bayes + Explainable AI (XAI)*")
st.divider()

# Trigger diagnosis
if diagnose_clicked:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom before diagnosing.")
        st.stop()
    st.session_state["last_symptoms"] = selected_symptoms
    st.session_state["last_results"] = predict_top_diseases(selected_symptoms, k=5)

# Show results if available
if "last_results" in st.session_state:
    results   = st.session_state["last_results"]
    symptoms  = st.session_state.get("last_symptoms", [])
    top3      = results[:3]
    top_prob  = top3[0][1] if top3 else 0

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔬 Top Diagnosis",  top3[0][0] if top3 else "—")
    m2.metric("📊 Confidence",    f"{top_prob*100:.1f}%")
    m3.metric("🩺 Symptoms Used", str(len(symptoms)))
    m4.metric("✅ Model Accuracy", f"{TEST_ACCURACY*100:.1f}%")

    st.divider()

    # ── Two-column layout: cards (left) + XAI chart (right) ──────────────
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("### 🏆 Top Predicted Diseases")
        rank_colors = ["#f85149", "#d29922", "#3fb950"]
        rank_labels = ["🥇", "🥈", "🥉"]

        for i, (disease, prob) in enumerate(top3):
            col = confidence_color(prob)
            pct = prob * 100
            st.markdown(f"""
            <div class="disease-card">
              <span class="rank-badge" style="background:{rank_colors[i]}">{rank_labels[i]} #{i+1}</span>
              <strong>{disease}</strong>
              <div style="margin-top:6px; color:{col}; font-size:0.9rem">{pct:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(prob))

        # Recommendation
        st.markdown("### 💊 Recommendation")
        if top_prob >= 0.70:
            st.error("🔴 **High confidence.** Please consult a doctor immediately.")
        elif top_prob >= 0.40:
            st.warning("🟡 **Moderate confidence.** Consider seeing a healthcare professional.")
        else:
            st.success("🟢 **Low confidence.** Monitor symptoms and see a doctor if they worsen.")

    with right:
        st.markdown("### 🔍 Why This Diagnosis?")
        st.caption(f"Log-likelihood contributions for **{top3[0][0]}**")

        contribs = get_top_contributions(top3[0][0], symptoms, top_n=12)
        fig = make_xai_figure(contribs, top3[0][0])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── What-If section ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚡ What-If Analysis")
    st.caption("Add or remove symptoms below to see how the prediction changes live.")

    wif_col1, wif_col2 = st.columns(2)
    add_sym    = wif_col1.multiselect(
        "➕ Add symptoms",
        [s for s in SYMPTOM_COLS if s not in symptoms],
        key="wif_add"
    )
    remove_sym = wif_col2.multiselect(
        "➖ Remove symptoms",
        symptoms,
        key="wif_remove"
    )

    if add_sym or remove_sym:
        modified = list((set(symptoms) | set(add_sym)) - set(remove_sym))
        new_results = predict_top_diseases(modified, k=5)

        wif_before, wif_after = st.columns(2)
        with wif_before:
            st.markdown("**Before**")
            for d, p in results[:5]:
                st.markdown(f"`{p*100:.1f}%` — {d}")
        with wif_after:
            st.markdown("**After**")
            for d, p in new_results[:5]:
                st.markdown(f"`{p*100:.1f}%` — {d}")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #8b949e;">
      <div style="font-size: 5rem;">🩺</div>
      <div style="font-size: 1.1rem; margin-top: 16px;">
        Select symptoms in the left panel<br>and press <strong style="color:#58a6ff">DIAGNOSE</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>DISCLAIMER:</strong> This tool is for <strong>EDUCATIONAL PURPOSES ONLY</strong>
  and does NOT constitute medical advice. Always consult a qualified healthcare
  professional for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
