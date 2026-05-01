"""
dataset_generator.py
====================
Generates a realistic disease-symptom CSV that mirrors the Kaggle
"Disease Symptom Prediction" dataset.  Run this ONCE to create
disease_symptom_dataset.csv in the same folder.

Why we generate it:  Kaggle requires an account/login.  For a
self-contained B.Tech project this synthetic set is sufficient.
The data follows the same column format so the rest of the code
works identically with the real Kaggle file if you download it.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducible randomness ──────────────────────────────────────────────────
np.random.seed(42)

# ── Disease → typical symptom profiles ───────────────────────────────────────
# Each disease has "core" symptoms (high prob) and "minor" ones (lower prob).
DISEASE_PROFILES = {
    "Common Cold": {
        "core": ["runny_nose", "sneezing", "sore_throat", "cough", "mild_fever"],
        "minor": ["headache", "fatigue", "body_ache", "nasal_congestion"],
    },
    "Influenza": {
        "core": ["high_fever", "body_ache", "fatigue", "headache", "cough"],
        "minor": ["sore_throat", "runny_nose", "chills", "vomiting", "diarrhea"],
    },
    "COVID-19": {
        "core": ["fever", "dry_cough", "fatigue", "loss_of_smell", "loss_of_taste"],
        "minor": ["headache", "body_ache", "shortness_of_breath", "sore_throat", "diarrhea"],
    },
    "Pneumonia": {
        "core": ["high_fever", "productive_cough", "shortness_of_breath", "chest_pain"],
        "minor": ["fatigue", "chills", "sweating", "nausea", "rapid_breathing"],
    },
    "Malaria": {
        "core": ["high_fever", "chills", "sweating", "headache", "body_ache"],
        "minor": ["nausea", "vomiting", "fatigue", "jaundice", "rapid_breathing"],
    },
    "Dengue": {
        "core": ["high_fever", "severe_headache", "joint_pain", "rash", "pain_behind_eyes"],
        "minor": ["nausea", "vomiting", "fatigue", "bleeding_gums", "body_ache"],
    },
    "Typhoid": {
        "core": ["prolonged_fever", "abdominal_pain", "headache", "weakness", "constipation"],
        "minor": ["rash", "loss_of_appetite", "sweating", "chills", "diarrhea"],
    },
    "Tuberculosis": {
        "core": ["persistent_cough", "weight_loss", "night_sweats", "fatigue", "fever"],
        "minor": ["chest_pain", "coughing_blood", "shortness_of_breath", "chills", "loss_of_appetite"],
    },
    "Asthma": {
        "core": ["shortness_of_breath", "wheezing", "chest_tightness", "cough"],
        "minor": ["fatigue", "rapid_breathing", "anxiety", "sleep_disturbance"],
    },
    "Diabetes": {
        "core": ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision"],
        "minor": ["slow_healing", "weight_loss", "numbness_in_feet", "dry_skin", "headache"],
    },
    "Hypertension": {
        "core": ["headache", "dizziness", "blurred_vision", "chest_pain"],
        "minor": ["shortness_of_breath", "nausea", "fatigue", "nosebleed", "palpitations"],
    },
    "Migraine": {
        "core": ["severe_headache", "nausea", "vomiting", "sensitivity_to_light"],
        "minor": ["dizziness", "visual_disturbances", "sensitivity_to_sound", "fatigue"],
    },
    "Gastroenteritis": {
        "core": ["diarrhea", "nausea", "vomiting", "abdominal_pain", "fever"],
        "minor": ["fatigue", "loss_of_appetite", "dehydration", "muscle_ache", "headache"],
    },
    "Appendicitis": {
        "core": ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite"],
        "minor": ["diarrhea", "constipation", "bloating", "rebound_tenderness", "fatigue"],
    },
    "Chickenpox": {
        "core": ["rash", "itching", "fever", "fatigue", "blisters"],
        "minor": ["headache", "loss_of_appetite", "sore_throat", "body_ache", "nausea"],
    },
    "Measles": {
        "core": ["high_fever", "rash", "cough", "runny_nose", "red_eyes"],
        "minor": ["sensitivity_to_light", "sore_throat", "white_spots_in_mouth", "fatigue", "body_ache"],
    },
    "Hepatitis B": {
        "core": ["jaundice", "fatigue", "abdominal_pain", "dark_urine", "nausea"],
        "minor": ["fever", "joint_pain", "vomiting", "loss_of_appetite", "pale_stools"],
    },
    "Urinary Tract Infection": {
        "core": ["frequent_urination", "burning_urination", "lower_abdominal_pain", "cloudy_urine"],
        "minor": ["fever", "fatigue", "blood_in_urine", "back_pain", "nausea"],
    },
    "Anemia": {
        "core": ["fatigue", "weakness", "pale_skin", "shortness_of_breath", "dizziness"],
        "minor": ["headache", "cold_hands", "chest_pain", "irregular_heartbeat", "brittle_nails"],
    },
    "Depression": {
        "core": ["persistent_sadness", "loss_of_interest", "fatigue", "sleep_disturbance", "concentration_problems"],
        "minor": ["appetite_changes", "worthlessness", "anxiety", "irritability", "headache"],
    },
}

# Collect all unique symptoms
ALL_SYMPTOMS = sorted(set(
    s for profile in DISEASE_PROFILES.values()
    for s in profile["core"] + profile["minor"]
))

SAMPLES_PER_DISEASE = 150   # 150 × 20 diseases = 3 000 rows total


def generate_row(disease: str) -> dict:
    """Generate one patient row for the given disease."""
    profile = DISEASE_PROFILES[disease]
    row = {"Disease": disease}

    for symptom in ALL_SYMPTOMS:
        if symptom in profile["core"]:
            # Core symptom: 75–95 % chance of being present
            row[symptom] = int(np.random.random() < np.random.uniform(0.75, 0.95))
        elif symptom in profile["minor"]:
            # Minor symptom: 25–55 % chance
            row[symptom] = int(np.random.random() < np.random.uniform(0.25, 0.55))
        else:
            # Unrelated symptom: 2–8 % noise
            row[symptom] = int(np.random.random() < np.random.uniform(0.02, 0.08))

    return row


def build_dataset() -> pd.DataFrame:
    rows = []
    for disease in DISEASE_PROFILES:
        for _ in range(SAMPLES_PER_DISEASE):
            rows.append(generate_row(disease))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "disease_symptom_dataset.csv")
    print("Generating dataset …")
    df = build_dataset()
    df.to_csv(output_path, index=False)
    print(f"✅  Dataset saved → {output_path}")
    print(f"   Shape  : {df.shape}  (rows × columns)")
    print(f"   Diseases: {df['Disease'].nunique()}")
    print(f"   Symptoms: {len(ALL_SYMPTOMS)}")
    print(df.head(3))
