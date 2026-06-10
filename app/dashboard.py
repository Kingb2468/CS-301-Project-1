import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(base_dir, 'models', 'mlr_model.joblib'))
scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.joblib'))

# Page config
st.set_page_config(
    page_title="Kitwe Solar Irradiance Predictor",
    page_icon="☀️",
    layout="centered"
)

# Title
st.title("☀️ Solar Irradiance Predictor")
st.subheader("Kitwe, Copperbelt — Zambia")
st.markdown("Predict **Global Horizontal Irradiance (GHI)** using climate variables.")

st.divider()

# Input section
st.header("🌡️ Enter Climate Variables")

col1, col2, col3 = st.columns(3)

with col1:
    t2m = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=22.0,
        step=0.1
    )

with col2:
    rh2m = st.number_input(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.1
    )

with col3:
    cloud_amt = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

st.divider()

# Predict button
if st.button("🔮 Predict GHI", use_container_width=True):
    input_data = np.array([[t2m, rh2m, cloud_amt]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### ☀️ Predicted GHI: {prediction:.4f} kWh/m²/day")

    # Interpretation
    st.header("📊 Interpretation")
    if prediction < 3:
        st.warning("🌧️ Low solar irradiance — heavy cloud cover or rain likely.")
    elif prediction < 5:
        st.info("⛅ Moderate solar irradiance — partly cloudy conditions.")
    else:
        st.success("🌞 High solar irradiance — clear sunny conditions!")

    # Bar chart
    st.header("📈 Input Variables Summary")
    fig, ax = plt.subplots(figsize=(8, 4))
    variables = ['Temperature (°C)', 'Humidity (%)', 'Cloud Cover (%)']
    values = [t2m, rh2m, cloud_amt]
    colors = ['red', 'blue', 'gray']
    ax.bar(variables, values, color=colors)
    ax.set_title('Input Climate Variables')
    ax.set_ylabel('Value')
    st.pyplot(fig)

st.divider()
st.caption("CS 301 Project — Group 10 | Copperbelt University | Data: NASA POWER")