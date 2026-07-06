import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff  
import sklearn.metrics as metrics
st.title("Overview of Data Preprocessing for Shear Wave and Compression Wave Travel Time Prediction")
#st.write("Hello! This is a demo on capability of Machine Learning models to predict shear wave and compression wave travel times.")
#uploaded_file = st.file_uploader("Upload your well log data in Excel format", type=["xlsx"])

FILE_ID = "16pEh2CX7mCW80TnAeP0WgmMIxgcbiadT"
DIRECT_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_data(ttl=3600)
def load_gdrive_data():
    return pd.read_excel(DIRECT_URL, engine='openpyxl')
    
uploaded_file = load_gdrive_data()
if uploaded_file is not None:
    df=uploaded_file
    if 'Resistivity' in df.columns and 'Gamma Ray' in df.columns:
        df_cleaned = df[((df['Resistivity'] > 0) & (df['Resistivity'] < 1000)) & 
                        ((df['Gamma Ray'] > 0) & (df['Gamma Ray'] < 400))]
    if 'Effective Porosity' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(['Effective Porosity'], axis=1)
    target_columns = ['Shear Wave Travel Time', 'Compression Wave Travel Time']
    if all(col in df_cleaned.columns for col in target_columns):
        X_raw = df_cleaned.drop(target_columns, axis=1)
        y_raw = df_cleaned[target_columns]
        from sklearn.preprocessing import MinMaxScaler
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        try:
            X_scaled_arr = feature_scaler.fit_transform(X_raw)
            y_scaled_arr = target_scaler.fit_transform(y_raw)
        
            X_scaled = pd.DataFrame(X_scaled_arr, columns=X_raw.columns)
            y_scaled = pd.DataFrame(y_scaled_arr, columns=y_raw.columns)
            #st.dataframe(pd.concat([X_scaled, y_scaled], axis=1).head())
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()
        from sklearn.model_selection import train_test_split
        #st.subheader("Model Training Setup")
        test_train_percent = 30
        test_size = test_train_percent / 100.0

        seed = 1000
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=seed)
        #st.success("Data preprocessing completed successfully. The dataset has been cleaned and scaled.")
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        SVR_model = MultiOutputRegressor(SVR(kernel='rbf', C=1, gamma=1))
        SVR_model.fit(X_train, y_train)
        y_pred_train = SVR_model.predict(X_train)
        y_pred_test = SVR_model.predict(X_test)
        st.subheader("Run predictions on Custom Data Input")
        st.write("Adjust the features below to view the predicted shear wave and compression wave travel times based on the trained model.")
        depth_min, depth_max = float(X_raw['Depth'].min()), float(X_raw['Depth'].max())
        res_min, res_max = float(X_raw['Resistivity'].min()), float(X_raw['Resistivity'].max())
        gr_min, gr_max = float(X_raw['Gamma Ray'].min()), float(X_raw['Gamma Ray'].max())
        tporo_min, tporo_max = float(X_raw['Total Porosity'].min()), float(X_raw['Total Porosity'].max())
        bude_min, bude_max = float(X_raw['Bulk Density'].min()), float(X_raw['Bulk Density'].max())
        depth_res = st.slider("Select Depth (m)", min_value=depth_min, max_value=depth_max, value=(depth_min + depth_max) / 2, step=100.00)
        user_res = st.slider("Select Resistivity (ohm-m)", min_value=res_min, max_value=res_max, value=(res_min + res_max) / 2, step=100.00)
        user_gr = st.slider("Select Gamma Ray (API)", min_value=gr_min, max_value=gr_max, value=(gr_min + gr_max) / 2, step=25.00)
        user_tporo = st.slider("Select Total Porosity", min_value=tporo_min, max_value=tporo_max, value=(tporo_min + tporo_max) / 2, step=0.01)
        user_bude = st.slider("Select Bulk Density (kg/m$^3$)", min_value=bude_min, max_value=bude_max, value=(bude_min + bude_max) / 2, step=0.1)

        if st.button("Predict Shear and Compression Wave Travel Times"):
            custom_input_df = pd.DataFrame({
                'Depth': [depth_res],
                'Resistivity': [user_res],
                'Gamma Ray': [user_gr],
                'Total Porosity': [user_tporo],
                'Bulk Density': [user_bude]
            })
            custom_input_scaled = feature_scaler.transform(custom_input_df)
            custom_pred_scaled = SVR_model.predict(custom_input_scaled)
            custom_pred_real = target_scaler.inverse_transform(custom_pred_scaled)

            out_col1, out_col2 = st.columns(2)
            out_col1.metric("Predicted Compression Wave Travel Time", f"{custom_pred_real[0][1]:.2f} µs/ft")
            out_col2.metric("Predicted Shear Wave Travel Time", f"{custom_pred_real[0][0]:.2f} µs/ft")

    else:
        st.error(f"Missing required target columns: {target_columns}")



else: 
    st.info("Please upload a well log data file in Excel format to proceed with the analysis.")