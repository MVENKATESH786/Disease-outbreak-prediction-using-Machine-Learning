import streamlit as st
import pickle
import os
import numpy as np
import time

# Set page configuration
st.set_page_config(page_title="Disease Prediction Model",
                   layout="wide", page_icon="🩺")

# Function to load models safely


def load_model(filepath):
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: {filepath} not found. Please check the file path.")
        return None


# Load models and scalers
heart_model = load_model("Model/heart_disease_model.sav")
diabetes_model = load_model("Model/diabetes_model.sav")
parkinson_model = load_model("Model/parkinsons_model.sav")

heart_scaler = load_model("Model/scaler_heart.sav")
diabetes_scaler = load_model("Model/scaler_diabetes.sav")
parkinson_scaler = load_model("Model/scaler_parkinsons.sav")

# Function to add logo


def add_logo():
    logo_path = "Images/Logo.jpg"
    if os.path.exists(logo_path):
        col1, col2 = st.columns([5, 1.5])
        with col1:
            st.title("🩺 Disease Prediction Model")
        with col2:
            st.image(logo_path, width=200)
    else:
        st.warning("Logo image not found. Please check the file path.")


add_logo()

# Function to predict heart disease


def predict_heart_disease(features):
    if heart_model and heart_scaler:
        features_scaled = heart_scaler.transform([features])
        return heart_model.predict(features_scaled)
    return None

# Function to predict diabetes


def predict_diabetes(features):
    if diabetes_model and diabetes_scaler:
        features_scaled = diabetes_scaler.transform([features])
        return diabetes_model.predict(features_scaled)
    return None

# Function to predict Parkinson's disease


def predict_parkinson(features):
    if parkinson_model and parkinson_scaler:
        features_scaled = parkinson_scaler.transform([features])
        return parkinson_model.predict(features_scaled)
    return None


# Define app tabs
tabs = st.tabs(["🏠 Home", "❤️ Heart Disease",
               "🩸 Diabetes", "🧠 Parkinson's Disease"])

# Home Tab
with tabs[0]:
    st.header("Welcome to the Disease Prediction Web App")
    st.markdown("""
    ### 🌟 About the Web App
    This application uses **Machine Learning models** to predict:
    - 🫀 **Heart Disease**
    - 🩸 **Diabetes**
    - 🧠 **Parkinson's Disease**

    ### 🛠 How to Use
    1. Select the disease prediction tab.
    2. Enter the required details.
    3. Click **Diagnose** to see results.

    ⚠️ **Disclaimer:** This tool is for informational purposes only. Consult a medical professional for diagnosis.
    """)

    # Display background image
    bg_path = "Images/Background.jpg"
    if os.path.exists(bg_path):
        st.image(bg_path, use_column_width=True, caption="YOUR HEALTH MATTERS")
    else:
        st.warning("Background image not found. Please check the file path.")

# Heart Disease Prediction Tab
with tabs[1]:
    st.header("❤️ Heart Disease Prediction")
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input(
                "Resting Blood Pressure", min_value=0, step=1)
            chol = st.number_input("Serum Cholesterol", min_value=0, step=1)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [0, 1])
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

        with col2:
            thalach = st.number_input("Max Heart Rate", min_value=0, step=1)
            exang = st.selectbox(
                "Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
            oldpeak = st.number_input("ST Depression", min_value=0.0, step=0.1)
            slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
            ca = st.selectbox(
                "Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [1, 2, 3])

        diagnose_button = st.form_submit_button("🩺 Diagnose")
        if diagnose_button:
            with st.spinner("Analyzing..."):
                time.sleep(2)
                features = [age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]
                prediction = predict_heart_disease(features)
                if prediction is not None:
                    if prediction == 1:
                        st.error("⚠️ High risk of Heart Disease!")
                    else:
                        st.success("✅ No risk of Heart Disease detected.")

# Diabetes Prediction Tab
with tabs[2]:
    st.header("🩸 Diabetes Prediction")
    with st.form(key='diabetes_form'):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
            glucose = st.number_input("Glucose Level", min_value=0, step=1)
            blood_pressure = st.number_input(
                "Blood Pressure", min_value=0, step=1)
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0, step=1)
            insulin = st.number_input("Insulin Level", min_value=0, step=1)

        with col2:
            bmi = st.number_input("Body Mass Index (BMI)",
                                  min_value=0.0, step=0.1)
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function", min_value=0.0, step=0.1)
            age = st.number_input("Age", min_value=1, max_value=120, step=1)

        diagnose_button = st.form_submit_button("🩺 Diagnose")
        if diagnose_button:
            with st.spinner("Analyzing..."):
                time.sleep(2)
                features = [pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, diabetes_pedigree, age]
                prediction = predict_diabetes(features)
                if prediction is not None:
                    if prediction == 1:
                        st.error("⚠️ High risk of Diabetes!")
                    else:
                        st.success("✅ No risk of Diabetes detected.")


# Parkinson's Disease Prediction Tab
with tabs[3]:
    st.header("Parkinson's Disease Prediction🧠")
    with st.form(key='parkinson_form'):
        # User input fields evenly split across two columns
        col1, col2 = st.columns(2)
        with col1:
            MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
            MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
            MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
            MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001, format="%.6f")
            MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, step=0.001, format="%.6f")
            MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0, step=0.001, format="%.6f")
            MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0, step=0.001, format="%.6f")
            Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0, step=0.001, format="%.6f")
            MDVP_Shim = st.number_input("MDVP:Shimmer", min_value=0.0, step=0.001, format="%.6f")
            MDVP_Shim_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, step=0.1)
            Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, step=0.001, format="%.6f")
        with col2:
            Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, step=0.001, format="%.6f")
            MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0, step=0.001, format="%.6f")
            Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0, step=0.001, format="%.6f")
            NHR = st.number_input("NHR", min_value=0.0, step=0.001, format="%.6f")
            HNR = st.number_input("HNR", min_value=0.0, step=0.1)
            RPDE = st.number_input("RPDE", min_value=0.0, max_value=1.0, step=0.001, format="%.6f")
            DFA = st.number_input("DFA", min_value=0.0, max_value=1.0, step=0.001, format="%.6f")
            spread1 = st.number_input("Spread1", min_value=-10.0, max_value=1.0, step=0.001, format="%.6f")
            spread2 = st.number_input("Spread2", min_value=-1.0, max_value=1.0, step=0.001, format="%.6f")
            D2 = st.number_input("D2", min_value=0.0, step=0.001, format="%.6f")
            PPE = st.number_input("PPE", min_value=0.0, step=0.001, format="%.6f")

        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)  # Simulate processing time
                features = [
                    MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ,
                    Jitter_DDP, MDVP_Shim, MDVP_Shim_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
                    NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
                ]
                prediction = predict_parkinson(features)
                if prediction == 1:
                    st.error("The Person has a risk of Parkinson's Disease", icon="⚠️")
                else:
                    st.success("The Person does not have a risk of Parkinson's Disease", icon="✅")

# Footer
st.markdown("---")
st.markdown(
    "💡 Developed by Maisagalla Venkatesh ❤️ using **Machine Learning & Streamlit** | 🚀 Stay Healthy!")
