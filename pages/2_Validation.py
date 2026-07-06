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
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        df_scaled_array = scaler.fit_transform(df_cleaned)
        df_scaled = pd.DataFrame(df_scaled_array, columns=df_cleaned.columns)
        #st.dataframe(df_scaled)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
    from sklearn.model_selection import train_test_split
    target_columns = ['Shear Wave Travel Time', 'Compression Wave Travel Time']
    if all(col in df_scaled.columns for col in target_columns):
        X = df_scaled.drop(target_columns, axis=1)
        y = df_scaled[target_columns]
        st.subheader("Model Training Setup")
        test_train_percent = st.slider("Select the Test Dataset Size (%):", 
                                       min_value=10, 
                                       max_value=50, 
                                       value=30, 
                                       step=5)
        test_size = test_train_percent / 100.0

        seed = 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        #st.success("Data preprocessing completed successfully. The dataset has been cleaned and scaled.")
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        SVR_model = MultiOutputRegressor(SVR(kernel='rbf', C=1, gamma=1))
        SVR_model.fit(X_train, y_train)

        y_pred_train = SVR_model.predict(X_train)
        y_pred_test = SVR_model.predict(X_test)

        corr_train1 = np.corrcoef(y_train['Compression Wave Travel Time'], y_pred_train[:, 0])[0,1]
        corr_train2 = np.corrcoef(y_train['Shear Wave Travel Time'], y_pred_train[:, 1])[0,1]

        corr_test1 = np.corrcoef(y_test['Compression Wave Travel Time'], y_pred_test[:, 0])[0,1]
        corr_test2 = np.corrcoef(y_test['Shear Wave Travel Time'], y_pred_test[:, 1])[0,1]

        mae_compression_train = metrics.mean_absolute_error(y_train['Compression Wave Travel Time'], y_pred_train[:, 0])
        mse_compression_train = metrics.mean_squared_error(y_train['Compression Wave Travel Time'], y_pred_train[:, 0])
        rmse_compression_train = metrics.root_mean_squared_error(y_train['Compression Wave Travel Time'], y_pred_train[:, 0])
        
        mae_shear_train = metrics.mean_absolute_error(y_train['Shear Wave Travel Time'], y_pred_train[:, 1])
        mse_shear_train = metrics.root_mean_squared_error(y_train['Shear Wave Travel Time'], y_pred_train[:, 1])
        rmse_shear_train = metrics.root_mean_squared_error(y_train['Shear Wave Travel Time'], y_pred_train[:, 1])

        mae_compression_test = metrics.mean_absolute_error(y_test['Compression Wave Travel Time'], y_pred_test[:, 0])
        mse_compression_test = metrics.root_mean_squared_error(y_test['Compression Wave Travel Time'], y_pred_test[:, 0])
        rmse_compression_test = metrics.root_mean_squared_error(y_test['Compression Wave Travel Time'], y_pred_test[:, 0])
        
        mae_shear_test = metrics.mean_absolute_error(y_test['Shear Wave Travel Time'], y_pred_test[:, 1])
        mse_shear_test = metrics.root_mean_squared_error(y_test['Shear Wave Travel Time'], y_pred_test[:, 1])
        rmse_shear_test = metrics.root_mean_squared_error(y_test['Shear Wave Travel Time'], y_pred_test[:, 1])

        tab1, tab2 = st.tabs(["Compression Wave Metrics", "Shear Wave Metrics"])

        with tab1:
            st.markdown("### Compression Wave Travel Time Performance")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Training $R^2$ Score", value=f"{corr_train1**2:.4f}", help="Coefficient of Determination (Train)")
                st.metric("Training Correlation", f"{corr_train1:.4f}")

            with m_col2:
                st.metric(label="Testing $R^2$ Score", value=f"{corr_test1**2:.4f}", help="Coefficient of Determination (Test)")
                st.metric("Testing Correlation", f"{corr_test1:.4f}")
            
            st.markdown("#### Training Error Quantifications")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("MAE (Train)", f"{mae_compression_train:.4f}")
            ec2.metric("MSE (Train)", f"{mse_compression_train:.4f}")
            ec3.metric("RMSE (Train)", f"{rmse_compression_train:.4f}")


            st.markdown("#### Test Error Quantifications")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("MAE (Test)", f"{mae_compression_test:.4f}")
            ec2.metric("MSE (Test)", f"{mse_compression_test:.4f}")
            ec3.metric("RMSE (Test)", f"{rmse_compression_test:.4f}")

            fig_compression_test = go.Figure()
            fig_compression_test.add_trace(go.Scatter(x=y_test['Compression Wave Travel Time'], 
                                                      y=y_pred_test[:, 0], 
                                                      mode='markers',
                                                      name='Compression Wave Test Data', 
                                                      marker=dict(color='blue', size=5, opacity=0.7)),
                                                      )
            min_val = min(y_test['Compression Wave Travel Time'].min(), y_pred_test[:, 0].min())
            max_val = max(y_test['Compression Wave Travel Time'].max(), y_pred_test[:, 0].max())
            fig_compression_test.add_trace(go.Scatter(x=[min_val, max_val], 
                                                      y=[min_val, max_val], 
                                                      mode='lines',
                                                      name='Ideal Fit',
                                                      line=dict(color='gray', dash='dash')))
            fig_compression_test.update_layout(
                title='Compression Wave Travel Time: Actual vs Predicted (Test Set)',
                xaxis_title='Actual Compression Wave Travel Time',
                yaxis_title='Predicted Compression Wave Travel Time',
                template='plotly_white',
                width=700, height=500
                )
            st.plotly_chart(fig_compression_test, use_container_width=True)
        
        with tab2:
            st.markdown("### Shear Wave Travel Time Performance")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Training $R^2$ Score", value=f"{corr_train2**2:.4f}", help="Coefficient of Determination (Train)")
                st.metric("Training Correlation", f"{corr_train2:.4f}")

            with m_col2:
                st.metric(label="Testing $R^2$ Score", value=f"{corr_test2**2:.4f}", help="Coefficient of Determination (Test)")
                st.metric("Testing Correlation", f"{corr_test2:.4f}")
            
            st.markdown("#### Training Error Quantifications")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("MAE (Train)", f"{mae_shear_train:.4f}")
            ec2.metric("MSE (Train)", f"{mse_shear_train:.4f}")
            ec3.metric("RMSE (Train)", f"{rmse_shear_train:.4f}")


            st.markdown("#### Test Error Quantifications")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("MAE (Test)", f"{mae_shear_test:.4f}")
            ec2.metric("MSE (Test)", f"{mse_shear_test:.4f}")
            ec3.metric("RMSE (Test)", f"{rmse_shear_test:.4f}")

            fig_shear_test = go.Figure()
            fig_shear_test.add_trace(go.Scatter(x=y_test['Shear Wave Travel Time'], 
                                                      y=y_pred_test[:, 1], 
                                                      mode='markers',
                                                      name='Shear Wave Test Data', 
                                                      marker=dict(color='red', size=5, opacity=0.7)),
                                                      )
            min_val = min(y_test['Shear Wave Travel Time'].min(), y_pred_test[:, 1].min())
            max_val = max(y_test['Shear Wave Travel Time'].max(), y_pred_test[:, 1].max())
            fig_shear_test.add_trace(go.Scatter(x=[min_val, max_val], 
                                                      y=[min_val, max_val], 
                                                      mode='lines',
                                                      name='Ideal Fit',
                                                      line=dict(color='gray', dash='dash')))
            fig_shear_test.update_layout(
                title='Shear Wave Travel Time: Actual vs Predicted (Test Set)',
                xaxis_title='Actual Shear Wave Travel Time',
                yaxis_title='Predicted Shear Wave Travel Time',
                template='plotly_white',
                width=700, height=500
                )
            st.plotly_chart(fig_shear_test, use_container_width=True)
else: 
    st.info("Please upload a well log data file in Excel format to proceed with the analysis.")