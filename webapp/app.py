import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(page_title="Rain Prediction – Input Explorer", layout="centered")

st.title("☔ Rain Prediction App")
st.subheader("🔍 Exploración de variables seleccionadas para el modelo")


features_path = 'data/model_output/selected_features.pkl'
data_path = 'data/processed/data_clean.csv'

# Obtener información del rango de valores para cada columna
feature_ranges = {}

if not os.path.exists(features_path) or not os.path.exists(data_path):
    st.error("❌ Asegurate de haber ejecutado el entrenamiento del modelo y guardado los archivos necesarios.")
else:
    selected_features = joblib.load(features_path)
    df_clean = pd.read_csv(data_path)
    df_subset = df_clean[selected_features]

    for col in df_subset.columns:
        if df_subset[col].dtype == 'object' or df_subset[col].dtype.name == 'category':
            feature_ranges[col] = df_subset[col].unique().tolist()
        else:
            feature_ranges[col] = {
                'min': df_subset[col].min(),
                'max': df_subset[col].max()
            }

