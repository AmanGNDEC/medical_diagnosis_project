# 🏥 AI Medical Diagnosis Assistant
### B.Tech Mini-Project — Naive Bayes + Explainable AI (XAI)

A desktop/web application that takes patient symptoms as input, uses **Naive Bayes** 
(implemented from scratch) to predict the most likely diseases, and provides 
**Explainable AI** showing *why* each disease was predicted.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## ✨ Features
- 🔬 **Naive Bayes from scratch** — Bayes' theorem with Laplace smoothing & log-space
- 📊 **Top-3 disease predictions** with confidence percentages
- 🔍 **Explainable AI (XAI)** — log-likelihood contribution chart per symptom
- ⚡ **What-If analysis** — add/remove symptoms and see live prediction changes
- 💊 **Recommendations** based on confidence level
- 🌐 **Web deployable** via Streamlit Cloud

## 🛠 Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Algorithm | Naive Bayes (scratch + sklearn BernoulliNB) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Web UI | Streamlit |
| Dataset | Synthetic disease-symptom (20 diseases, 50 symptoms, 3000 rows) |

## 📁 Project Structure
```
medical_diagnosis_ai/
├── dataset_generator.py   # Generates the disease-symptom CSV
├── data_prep.py           # STEP 1 — Data loading & preprocessing
├── model.py               # STEP 2 — Naive Bayes classifier
├── xai.py                 # STEP 3 — XAI log-likelihood contributions
├── streamlit_app.py       # STEP 4 — Web UI (Streamlit)
├── app.py                 # Alternative local browser UI
├── main.py                # Entry point for local run
└── requirements.txt
```

## 🏃 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd medical_diagnosis_ai
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ☁️ Deploy to Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub → **New App**
4. Select your repo, set **Main file** = `streamlit_app.py`
5. Click **Deploy** — live in ~2 minutes!

## 🎓 AI Concepts (Viva Prep)
| Concept | Implementation |
|---|---|
| Bayes' Theorem | `P(D\|S) ∝ P(D) × Π P(Sᵢ\|D)` |
| Laplace Smoothing | `(count+α) / (n+2α)` — avoids zero probability |
| Log-space arithmetic | Prevents float underflow on 50+ features |
| XAI | `contribution = log P(symptomᵢ \| disease)` |
| What-If | Re-runs prediction with modified symptom set |

## ⚠️ Disclaimer
This tool is for **EDUCATIONAL PURPOSES ONLY** and does NOT constitute medical advice.
