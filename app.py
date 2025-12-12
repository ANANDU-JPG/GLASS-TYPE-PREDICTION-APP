import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("glass_model.pkl")      
scaler = joblib.load("scaler.pkl")         

st.title(" Glass Type Prediction App")
st.write("Enter the chemical composition to predict the glass type.")


RI = st.number_input("Refractive Index (RI)", value=1.51, format="%.5f")
Na = st.number_input("Na (Sodium Content)", value=13.0)
Mg = st.number_input("Mg (Magnesium Content)", value=3.0)
Al = st.number_input("Al (Aluminum Content)", value=1.0)
Si = st.number_input("Si (Silicon Content)", value=72.0)
K  = st.number_input("K (Potassium Content)", value=0.5)
Ca = st.number_input("Ca (Calcium Content)", value=8.0)
Ba = st.number_input("Ba (Barium Content)", value=0.0)
Fe = st.number_input("Fe (Iron Content)", value=0.0)

input_data = pd.DataFrame(
    [[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]],
    columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
)
if st.button("Predict Glass Type"):
    
    scaled_input = scaler.transform(input_data)

    

    prediction = model.predict(scaled_input)[0]

    st.subheader(" Prediction:")
    st.success(f"The predicted Glass Type is: **{prediction}**")

    

