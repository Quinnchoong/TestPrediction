# Final minimal, single-page version of the app

minimal_app_code = """
# -*- coding: utf-8 -*-
# student_employability_app_final.py - Minimal Single-Page Version

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# --- Streamlit App Setup ---
st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")

# --- CSS: Light blue background & compact layout ---
st.markdown(\"\"\"
<style>
.stApp { background-color: #e6f2ff; }
html, body, [class*="css"] { font-size: 14px; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
\"\"\", unsafe_allow_html=True)

# --- Load Model & Scaler ---
@st.cache_resource
def load_model():
    try:
        with open("employability_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()
if model is None or scaler is None:
    st.error("⚠️ Model or scaler file not found.")
    st.stop()

# --- Header Image ---
try:
    image = Image.open("group-business-people-silhouette-businesspeople-abstract-background_656098-461.avif")
    st.image(image, use_container_width=True)
except:
    pass

st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>", unsafe_allow_html=True)
st.markdown("Fill in the input features below and click **Predict** to see results.")

feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

# --- Input Form ---
col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    inputs['GENDER'] = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female", index=1)
    inputs['GENERAL_APPEARANCE'] = st.slider("Appearance (1-5)", 1, 5, 3)
    inputs['GENERAL_POINT_AVERAGE'] = st.number_input("GPA (0.0-4.0)", 0.0, 4.0, 3.0, 0.01)
    inputs['MANNER_OF_SPEAKING'] = st.slider("Speaking (1-5)", 1, 5, 3)

with col2:
    inputs['PHYSICAL_CONDITION'] = st.slider("Physical (1-5)", 1, 5, 3)
    inputs['MENTAL_ALERTNESS'] = st.slider("Alertness (1-5)", 1, 5, 3)
    inputs['SELF-CONFIDENCE'] = st.slider("Confidence (1-5)", 1, 5, 3)
    inputs['ABILITY_TO_PRESENT_IDEAS'] = st.slider("Ideas (1-5)", 1, 5, 3)

with col3:
    inputs['COMMUNICATION_SKILLS'] = st.slider("Communication (1-5)", 1, 5, 3)
    inputs['STUDENT_PERFORMANCE_RATING'] = st.slider("Performance (1-5)", 1, 5, 3)
    inputs['NO_SKILLS'] = st.radio("Has No Skills", [0,1], format_func=lambda x: "No" if x==0 else "Yes", index=0)
    inputs['Year_of_Graduate'] = st.number_input("Graduation Year", 2019, 2025, 2022)

input_df = pd.DataFrame([inputs])[feature_columns]

# --- Predict ---
if st.button("🔮 Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.write("🎓 The system predicts if the student is employable or less employable.")

    if prediction == 1:
        result_text = "✅ The student is predicted to be **Employable**"
        result_color = "green"
    else:
        result_text = "⚠️ The student is predicted to be **Less Employable**"
        result_color = "red"

    st.markdown(f"<h3 style='color:{result_color}'>{result_text}</h3>", unsafe_allow_html=True)
    st.info(f"Probability of being Employable: {proba[1]*100:.2f}%")
    st.info(f"Probability of being Less Employable: {proba[0]*100:.2f}%")

st.markdown("---")
st.caption("© 2025 CHOONG MUH IN | Graduate Employability Prediction App | For research purposes only.")
"""
