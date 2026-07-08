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

st.html(
    """
    <style>
    .stApp{
        background-color: #FFFFFF;
    }
    </style>    
"""
)
st.title("ML based Prediction of Shear Wave and Compression Wave Travel Times")
st.subheader("Introduction")
intro_text="""
<p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;">
    The application demonstrates the use of Machine Learning (ML) models to predict shear wave and compression wave travel times.
</p>
"""
st.markdown(intro_text, unsafe_allow_html=True)
#st.write("The application demonstrates the use of Machine Learning (ML) models to predict shear wave and compression wave travel times. ")
st.subheader("Background")
background_text="""
<p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;">Geologists, geophysicists, petrophysicists, and petroleum engineers rely heavily on compressional sonic logs for subsurface characterization.
These logs measure acoustic compressional (P-wave) and shear (S-wave) travel times through the subsurface, offering key data on formation lithology, porosity, and mechanical properties.
Petroleum engineers leverage compressional sonic logs to analyze wellbore stability, identify fractures, and optimize drilling operations in real time.
</p>
"""
st.markdown(background_text, unsafe_allow_html=True)

st.subheader("Challenge")
challenge_text="""
<p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;"> Wireline and logging-while-drilling (LWD) sonic tools offer direct measurements of P-wave travel times. 
    However, despite delivering the high-resolution data essential for formation evaluation, direct sonic logging incurs substantial financial costs and requires extra rig time. 
    Additionally, prolonged open-hole exposure in shaly formations during these operations can jeopardize wellbore stability, elevating the risks of borehole collapse and formation damage. 
    Machine learning addresses these challenges by accurately predicting compressional velocities from extensive datasets, enabling the reliable reconstruction of missing sonic logs through the correlation of existing petrophysical well data.
</p>
"""
st.markdown(challenge_text, unsafe_allow_html=True)

st.subheader("Objective")
objective_text="""
<p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;"> To provide an user flexibility to predict shear and compression wave travel times by 
    varying petrophysical well data. 
</p>
"""
st.markdown(objective_text, unsafe_allow_html=True)

st.subheader("Methodology")
methodology_text="""
<p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;">
    <ul style="margin-top: 5px; padding-left: 20px;">
        <li><strong>Data Preprocessing:</strong> Preprocess the data for implementation of ML model. Check for details on our <a href="/Preprocessing" target="_blank" style="color: #0066cc; font-weight: bold;"> Preprocessing Page</a>.</li>
        <li><strong>Model Selection and Validation:</strong> Select a ML model and calculate the following:
            <ul style="margin-top: 5px; margin-bottom: 5px; padding-left: 20px; list-style-type: circle;">
                <li>Model Performance by Coefficient of Determination (R<sup>2</sup>)</li>
                <li>Error Quantification by Mean Absolute Error (MAE), Mean Square Error (MSE) and Root Mean Square Error (RMSE)</li>
            </ul>
            You can check for performance of model: <strong>Support Vector Machine (SVM)</strong> on our <a href="/Validation" target="_blank" style="color: #0066cc; font-weight: bold;">Model Validation Page</a>
        </li>
    </ul>
</p>
"""
st.markdown(methodology_text, unsafe_allow_html=True)
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
        st.subheader("Results")
        #st.write("Adjust the features below to view the predicted shear wave and compression wave travel times based on the trained model.")
        results_text="""
        <p style="text-align: justify; font-size: 15px; line-height: 1.6; color: #333333;"> You can adjust the petrophysical well data by using the sliders below to predict 
        shear wave and compression wave travel times. 
        </p>
        """
        st.markdown(results_text, unsafe_allow_html=True)

        
        depth_min, depth_max = float(X_raw['Depth'].min()), float(X_raw['Depth'].max())
        res_min, res_max = float(X_raw['Resistivity'].min()), float(X_raw['Resistivity'].max())
        gr_min, gr_max = float(X_raw['Gamma Ray'].min()), float(X_raw['Gamma Ray'].max())
        tporo_min, tporo_max = float(X_raw['Total Porosity'].min()), float(X_raw['Total Porosity'].max())
        bude_min, bude_max = float(X_raw['Bulk Density'].min()), float(X_raw['Bulk Density'].max())
        
        col_left, col_right = st.columns(2)
        with col_left:
            depth_res = st.slider("Select Depth (ft)", min_value=depth_min, max_value=depth_max, value=(depth_min + depth_max) / 2, step=100.00)
            user_res = st.slider("Select Resistivity (ohm-m)", min_value=res_min, max_value=res_max, value=(res_min + res_max) / 2, step=100.00)
            user_gr = st.slider("Select Gamma Ray (API)", min_value=gr_min, max_value=gr_max, value=(gr_min + gr_max) / 2, step=25.00)
        
        with col_right:
            user_tporo = st.slider("Select Total Porosity", min_value=tporo_min, max_value=tporo_max, value=(tporo_min + tporo_max) / 2, step=0.01)
            user_bude = st.slider("Select Bulk Density (g/cm³)", min_value=bude_min, max_value=bude_max, value=(bude_min + bude_max) / 2, step=0.1)

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

footer_html="""
<style>
.custom-footer {
    margin-top: 60px;
    width: 100%;
    background-color: transparent;
    padding-bottom: 20px;
    font-family: sans-serif;
}
.footer-main-line {
    margin: 0 0 20px 0;
    font-size: 14px;
    text-align: left;
    color: #333333;
}

.reference-box{
    max-width: 800px;
    margin: 0 auto;
    text-align: left;
    padding: 0 20px;
}

.reference-title{
    font-size: 14px;
    font-weight: bold;
    color: #333333;
    margin-bottom: 8px;

}

.reference-item,
.reference-item a,
.reference-item span{
    font-size: 10px !important;
    color: #555555;
    line-height: 1.5;
}
.reference-item{
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}
</style>
<div class="custom-footer">
    <div class="reference-box">
        <p class="footer-main-line">
            <a href="https://www.linkedin.com/in/prasanna-perumal-pp/" target="_blank" style="color: #0a66c2; text-decoration: none; font-weight: bold; display: inline-block">
            Prasanna Perumal</a>, 
            Research Scholar, Indian Institute of Technology Kharagpur | © 2026 
        </p>
        <div class="reference-title">References</div>
        <p class="reference-item"> 1. Belyadi, H., & Haghighat, A. (2021).
            <a href="https://books.google.co.in/books?id=MjoEEAAAQBAJ&lpg=PP1&ots=bgvs7mAGB3&dq=machine%20learning%20guide%20for%20oil%20and%20gas%20using%20python&lr&pg=PP1#v=onepage&q=machine%20learning%20guide%20for%20oil%20and%20gas%20using%20python&f=false" target="_blank" style="color: #555555; text-decoration: underline; display: inline; font-style: italic;">
                Machine learning guide for oil and gas using Python
            </a>. 
        Gulf Professional Publishing.
        </p>
        <p class="reference-item">2. Saleh, K., Mabrouk, W. M., & Metwally, A. (2025).
            <a href="https://doi.org/10.1038/s41598-025-97938-9" target="_blank" style="color: #555555; text-decoration: underline; display: inline; font-style: italic;">
                Machine learning model optimization for compressional sonic log prediction using well logs in Shahd SE field, Western Desert, Egypt
            </a>
             <span style="font-style: italic;">Scientific Reports</span>, <span style="font-style: italic;">15</span>(1), 14957
        </p>
    </div>       
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)