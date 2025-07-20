# Combine the best parts of both versions into a clean, final app

combined_app_code = """
# -*- coding: utf-8 -*-
# student_employability_app_final.py - Final Combined Version

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
from fpdf import FPDF
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

# --- Utility Functions ---
def generate_pdf_report(data, result, confidence, proba):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Employability Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for k, v in data.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Probabilities: Employable {proba[1]*100:.2f}%, Less Employable {proba[0]*100:.2f}%", ln=True)
    file_path = "prediction_report.pdf"
    pdf.output(file_path)
    return file_path

def get_pdf_download_link(file_path):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📄 Download PDF Report</a>'
    return href

# --- Header Image ---
try:
    image = Image.open("group-business-people-silhouette-businesspeople-abstract-background_656098-461.avif")
    st.image(image, use_container_width=True)
except:
    pass

st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>", unsafe_allow_html=True)
st.markdown("Fill in the input features to predict employability.")

tab1, tab2, tab3 = st.tabs(["📋 Input Form", "📊 Feature Insights", "📄 Report"])

feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

# ---------------- Tab 1: Input Form ----------------
with tab1:
    st.header("📋 Student Profile Input")

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

    if st.button("🔮 Predict"):
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.write("🎓 The system predicts if the student is employable or less employable.")

        if prediction == 1:
            result_text = "✅ The student is predicted to be **Employable**"
            result_color = "green"
        else:
            result_text = "⚠️ The student is predicted to be **Less Employable**"
            result_color = "red"

        confidence = proba[prediction] * 100

        st.session_state['data'] = dict(zip(feature_columns, input_df.iloc[0]))
        st.session_state['result'] = result_text
        st.session_state['confidence'] = confidence
        st.session_state['proba'] = proba

        st.markdown("---")
        st.markdown(f"<h3 style='color:{result_color}'>{result_text}</h3>", unsafe_allow_html=True)
        st.info(f"Probability of being Employable: {proba[1]*100:.2f}%")
        st.info(f"Probability of being Less Employable: {proba[0]*100:.2f}%")

# ---------------- Tab 2: Feature Insights ----------------
with tab2:
    st.header("📊 Feature Contribution")

    if 'data' in st.session_state:
        df = pd.DataFrame([st.session_state['data']])
        df.T.plot(kind="barh", legend=False, figsize=(8, 6), color='skyblue')
        plt.xlabel("Feature Value")
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

# ---------------- Tab 3: Report ----------------
with tab3:
    st.header("📄 Downloadable Prediction Report")

    if 'result' in st.session_state:
        pdf_path = generate_pdf_report(
            st.session_state['data'],
            st.session_state['result'],
            st.session_state['confidence'],
            st.session_state['proba']
        )
        st.markdown(get_pdf_download_link(pdf_path), unsafe_allow_html=True)
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

st.markdown("---")
st.caption("© 2025 CHOONG MUH IN | Graduate Employability Prediction App | For research purposes only.")
"""

with open("/mnt/data/student_employability_app_final.py", "w") as f:
    f.write(combined_app_code)

"/mnt/data/student_employability_app_final.py"
