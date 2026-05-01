"""
main.py  ── Entry Point (runs the full application)
====================================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant

Usage:
    python main.py

What it does:
1. Checks that the dataset exists (auto-generates if not).
2. Pre-trains the model and warms up the cache.
3. Launches the Tkinter desktop application.
"""

import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP BANNER
# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║      🏥  AI-Powered Medical Diagnosis Assistant  🏥          ║
║      B.Tech Mini-Project  |  Naive Bayes + XAI              ║
╠══════════════════════════════════════════════════════════════╣
║  TECH STACK                                                  ║
║    • Python 3             • scikit-learn (BernoulliNB)       ║
║    • Naive Bayes (scratch) with Laplace smoothing            ║
║    • Explainable AI (log-likelihood contributions)           ║
║    • matplotlib / seaborn  (embedded charts)                 ║
║    • tkinter               (desktop GUI)                     ║
╚══════════════════════════════════════════════════════════════╝
"""


def check_dependencies():
    """Verify required packages are installed.
    NOTE: tkinter is NOT required — the UI runs in your web browser.
    """
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }
    missing = []
    for import_name, pkg_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print("❌  Missing packages detected:")
        for pkg in missing:
            print(f"     pip install {pkg}")
        print("\n  Run:  pip install -r requirements.txt")
        sys.exit(1)


def ensure_dataset():
    """Generate dataset if the CSV is not present."""
    dataset_path = os.path.join(PROJECT_DIR, "disease_symptom_dataset.csv")
    if not os.path.exists(dataset_path):
        print("  ⚙️  Dataset not found — generating …")
        import subprocess
        gen = os.path.join(PROJECT_DIR, "dataset_generator.py")
        subprocess.run([sys.executable, gen], check=True)
        print("  ✅  Dataset generated.\n")
    else:
        print("  ✅  Dataset found.\n")


def warmup_model():
    """Pre-train and cache the model so the UI opens instantly."""
    from model import get_model_and_symptoms
    print("  ⚙️  Training Naive Bayes model …", end="", flush=True)
    t0 = time.time()
    model, symptoms = get_model_and_symptoms()
    elapsed = time.time() - t0
    print(f" done ({elapsed:.2f}s)  |  {len(model.classes_)} diseases  |"
          f"  {len(symptoms)} symptoms")


def main():
    print(BANNER)

    # ── Step 0: dependency check ──────────────────────────────────────────
    print("  Checking dependencies …")
    check_dependencies()
    print("  ✅  All dependencies satisfied.\n")

    # ── Step 1: ensure dataset ────────────────────────────────────────────
    print("  Checking dataset …")
    ensure_dataset()

    # ── Step 2: warm up model ─────────────────────────────────────────────
    warmup_model()
    print()

    # ── Step 3: launch GUI ────────────────────────────────────────────────
    print("  🚀  Launching GUI …\n")
    from app import run_app
    run_app()


if __name__ == "__main__":
    main()
