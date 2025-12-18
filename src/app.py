import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="SpaceX Landing Predictor", page_icon="🚀")

st.title("🚀 SpaceX Falcon 9 Landing Predictor")
st.write("This app predicts if a SpaceX booster will land successfully based on mission data.")

try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df_template = pd.read_csv('dataset_part_3.csv')
    st.success("Model and Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop() 

st.header("Enter Mission Details")

col1, col2 = st.columns(2)

with col1:
    flight_num = st.number_input("Flight Number", value=90)
    payload = st.number_input("Payload Mass (kg)", value=5000)
    flights = st.number_input("Booster Flights Count", value=1)
    block = st.slider("Booster Block Version", 1.0, 5.0, 5.0)

with col2:
    orbit = st.selectbox("Orbit Type", ["LEO", "GTO", "ISS", "PO", "VLEO", "SSO", "MEO", "GEO", "HEO", "ES-L1", "SO"])
    site = st.selectbox("Launch Site", ["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"])
    grid_fins = st.checkbox("Grid Fins Used", value=True)
    legs = st.checkbox("Landing Legs Used", value=True)

if st.button("Predict Landing Result"):
    
    input_data = pd.DataFrame(columns=df_template.columns)
    input_data.loc[0] = 0.0  

    input_data['FlightNumber'] = float(flight_num)
    input_data['PayloadMass'] = float(payload)
    input_data['Flights'] = float(flights)
    input_data['Block'] = float(block)
    input_data['GridFins'] = 1.0 if grid_fins else 0.0
    input_data['Legs'] = 1.0 if legs else 0.0

    orbit_column = f"Orbit_{orbit}"
    site_column = f"LaunchSite_{site}"

    if orbit_column in input_data.columns:
        input_data[orbit_column] = 1.0
    if site_column in input_data.columns:
        input_data[site_column] = 1.0

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("🚀 SUCCESS: The booster is predicted to land successfully!")
        st.balloons()
    else:
        st.error("💥 FAILURE: The booster is predicted to crash or fail landing.")

if st.checkbox("Show historical data (Part 2)"):
    df2 = pd.read_csv('dataset_part_2.csv')
    st.write(df2.head())