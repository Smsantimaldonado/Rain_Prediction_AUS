import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(page_title='Rain Prediction – Input Explorer', layout='centered')
st.title('☔ Australia Rain Prediction App')
st.subheader("🔍 Will tomorrow rain? Let's find out")

# ============================
# Paths
features_path = 'data/model_output/selected_features.pkl'
data_path = 'data/processed/data_clean.csv'
model_path = 'data/model_output/best_model_.pkl'
encoder_path = 'data/model_output/label_encoder.pkl'

# ============================
# One time load
@st.cache_resource
def load_resources():
    selected_features = joblib.load(features_path)
    df_clean = pd.read_csv(data_path)
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return selected_features, df_clean, model, encoder

# ============================
# Predict function
def predict_rain(user_inputs_dict, model, encoder, selected_features):
    df_user = pd.DataFrame([user_inputs_dict])

    # Verify columns
    missing_cols = []
    for col in selected_features:
        if col not in df_user.columns:
            missing_cols.append(col)

    if missing_cols:
        st.error(f'❌ Missing columns: {missing_cols}')
        return

    df_user = df_user[selected_features]

    pred = model.predict(df_user)[0]
    pred_label = encoder.inverse_transform([pred])[0]

    st.success(f'🌧️ Prediction: Tomorrow {'WILL RAIN' if pred_label == 1 else 'WILL NOT RAIN'}')

# ============================
# Main execution
if not os.path.exists(features_path) or not os.path.exists(data_path) or not os.path.exists(model_path):
    st.error('❌ Missing model or data files.')
else:
    selected_features, df_clean, model, encoder = load_resources()
    df_subset = df_clean[selected_features]

    # Get ranges
    feature_ranges = {}
    for col in df_subset.columns:
        if df_subset[col].dtype == 'object' or df_subset[col].dtype.name == 'category':
            feature_ranges[col] = df_subset[col].unique().tolist()
        else:
            feature_ranges[col] = {
                'min': df_subset[col].min(),
                'max': df_subset[col].max()
            }


tab1, tab2 = st.tabs(['🔘 Individual prediction', '📂 File prediction (CSV)'])

# ====================
# TAB 1 – Individual prediction
with tab1:
    # Inputs
    st.subheader('✍️ Input values for prediction')
    user_inputs = {}

    for feature, values in feature_ranges.items():
        if isinstance(values, dict):  # numerical
            min_val = float(values['min'])
            max_val = float(values['max'])
            default_val = round((min_val + max_val) / 2, 2)

            user_input = st.number_input(
                label=f'{feature}',
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=0.01,
                format='%.2f'
            )

            user_inputs[feature] = user_input

    # Start prediction
    if st.button('Submit'):
        predict_rain(user_inputs, model, encoder, selected_features)

# ====================
# TAB 2 – File prediction
with tab2:
    st.subheader('📂 Upload your CSV file with the following values')
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            # Check required columns
            missing_cols = []
            for col in selected_features:
                if col not in df_uploaded.columns:
                    missing_cols.append(col)
            if missing_cols:
                st.error(f'❌ File does not contain correct columns: {missing_cols}')
            else:
                # Ensure column orde
                df_uploaded = df_uploaded[selected_features]

                # Predict
                predictions = model.predict(df_uploaded)
                predicted_labels = encoder.inverse_transform(predictions)

                # Add predictions to dataframe
                df_uploaded['Prediction'] = predicted_labels

                # Show predictions
                st.success('✅ Predictions done')
                st.dataframe(df_uploaded)


                # ================================
                # 📊 1. Countplot - Distribution of predictions
                st.subheader('📊 Frequency distribution of predictions')

                fig_count, ax_count = plt.subplots()
                sns.countplot(x='Prediction', data=df_uploaded, ax=ax_count)
                ax_count.set_title('Prediction distribution')
                ax_count.set_xticklabels(encoder.classes_) #ax_count.set_xticklabels(['No rain', 'Rain'])
                st.pyplot(fig_count)

                # ================================
                # 🗺️ 2. Geographic map of predictions
                if 'Latitude' in df_uploaded.columns and 'Longitude' in df_uploaded.columns:
                    fig_map = px.scatter_geo(
                        df_uploaded,
                        lat='Latitude',
                        lon='Longitude',
                        color='Prediction',
                        # Considera un color_continuous_scale si Prediction es numérica (0/1)
                        # Si es categórica ('Yes'/'No'), usa un color_discrete_map o simplemente deja Plotly manejarlo
                        color_continuous_scale='Bluered' if df_uploaded['Prediction'].dtype != 'object' else None,
                        color_discrete_map={'No': 'blue', 'Yes': 'red'} if df_uploaded['Prediction'].dtype == 'object' else None, # Ejemplo si las etiquetas son 'Yes'/'No'
                        title='Geolocalized predictions',
                        # scope='asia', # Podrías necesitar ajustar el scope o permitir al usuario seleccionarlo
                        labels={'Prediction': 'Rain?'}
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Latitude and Longitude columns are required for the map and were not found in the uploaded file.")

                # ================================
                # Option to download results
                csv = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button('📥 Download results', data=csv, file_name='predictions.csv', mime='text/csv')


        except Exception as e:
            st.error(f'❌ Error when loading file: {e}')