import streamlit as st
import pickle
import os
import numpy as np
import time

# Set page configuration
st.set_page_config(page_title="Disease Prediction Model", layout="wide", page_icon="Images/Icon.png")

def add_logo():
    logo = "Images/logo.png" 
    col1, col2 = st.columns([5, 1.5]) 
    with col1:
        st.title("Disease Prediction Model🩺")  
    with col2:
        st.image(logo, width=200)  

add_logo()

# Load the saved models and scalers
heart_model = pickle.load(open('Saved Models/heart_disease_model.sav', 'rb'))
diabetes_model = pickle.load(open('Saved Models/diabetes_model.sav', 'rb'))
parkinson_model = pickle.load(open('Saved Models/parkinsons_model.sav', 'rb'))

heart_scaler = pickle.load(open('Saved Models/scaler_heart.sav', 'rb'))
diabetes_scaler = pickle.load(open('Saved Models/scaler_diabetes.sav', 'rb'))
parkinson_scaler = pickle.load(open('Saved Models/scaler_parkinsons.sav', 'rb'))

# Function to predict heart disease
def predict_heart_disease(features):
    features_scaled = heart_scaler.transform([features])
    prediction = heart_model.predict(features_scaled)
    return prediction

# Function to predict diabetes
def predict_diabetes(features):
    features_scaled = diabetes_scaler.transform([features])
    prediction = diabetes_model.predict(features_scaled)
    return prediction

# Function to predict Parkinson's disease
def predict_parkinson(features):
    features_scaled = parkinson_scaler.transform([features])
    prediction = parkinson_model.predict(features_scaled)
    return prediction

# App interface
tabs = st.tabs(["Home", "Heart Disease Prediction", "Diabetes Prediction", "Parkinson's Prediction"])

with tabs[0]:
    st.title("Welcome to the Disease Prediction Web App")
    st.markdown("""
    ### About the Web App
    This application uses Machine Learning models to predict the likelihood of:
    - **Heart Disease**
    - **Diabetes**
    - **Parkinson's Disease**
    
    ### How to Use the Web App
    1. Navigate to the respective tabs for Heart, Diabetes, or Parkinson's predictions.
    2. Fill in the required input features in the form.
    3. Click **Diagnose** to see the result.

    ### Purpose
    This app aims to assist medical professionals and individuals in identifying potential risks early, enabling timely medical intervention.
    """)
    
    # Updated image rendering
    st.image("Images/Background.jpg", use_container_width=True, caption="YOUR HEALTH MATTERS")

# Heart Disease Prediction Tab
with tabs[1]:
    st.header("Heart Disease Prediction🫀")
    with st.form(key='heart_form'):
        # User input fields
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, step=1)
            sex = st.selectbox("Sex", [0, 1])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", min_value=0, step=1)
            chol = st.number_input("Serum Cholesterol", min_value=0, step=1)
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
            restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("Depression Induced by Exercise", min_value=0.0, step=0.1)
            slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [1, 2, 3])
        
        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)  # Simulate processing time
                features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                prediction = predict_heart_disease(features)
                if prediction == 1:
                    st.error("The Person has a risk of Heart disease", icon="⚠️")
                else:
                    st.success("The Person does not have a risk of Heart disease", icon="✅")

# Diabetes Prediction Tab
with tabs[2]:
    st.header("Diabetes Prediction🩸")
    with st.form(key='diabetes_form'):
        # User input fields
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
            glucose = st.number_input("Glucose", min_value=0, step=1)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
            insulin = st.number_input("Insulin", min_value=0, step=1)
        with col2:
            bmi = st.number_input("BMI", min_value=0.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
            age = st.number_input("Age", min_value=0, max_value=100, step=1)

        diagnose_button = st.form_submit_button(label="Diagnose")
        if diagnose_button:
            with st.spinner('Analyzing... Please wait.'):
                time.sleep(2)  # Simulate processing time
                features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
                prediction = predict_diabetes(features)
                if prediction == 1:
                    st.error("The Person has a risk of Diabetes", icon="⚠️")
                else:
                    st.success("The Person does not have a risk of Diabetes", icon="✅")

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
