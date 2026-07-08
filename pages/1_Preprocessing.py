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
st.title("Data Preprocessing")
#st.write("Hello! This is a demo on capability of Machine Learning models to predict shear wave and compression wave travel times.")
#uploaded_file = st.file_uploader("Upload your well log data in Excel format", type=["xlsx"])

FILE_ID = "16pEh2CX7mCW80TnAeP0WgmMIxgcbiadT"
DIRECT_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_data(ttl=3600)
def load_gdrive_data():
    return pd.read_excel(DIRECT_URL, engine='openpyxl')
    
uploaded_file = load_gdrive_data()
if uploaded_file is not None:
    df = uploaded_file
    st.dataframe(df)
    #st.subheader("To find out if we can apply Machine Learning, we need all the parameters in normal distribution.")
    st.subheader("Step 1: Prepare distribution plots for each feature.")
    st.markdown("*Tip: Hover over data points to see the values, use the top-right corner to zoom and pan, or double click to reset view*")
    features = [('Depth','red'), ('Gamma Ray', 'olive'), 
                ('Total Porosity', 'blue'), ('Effective Porosity', 'orange'), 
                ('Resistivity', 'black'), ('Bulk Density', 'green'), 
                ('Compression Wave Travel Time', 'cyan'), 
                ('Shear Wave Travel Time', 'brown')
                ]
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)

        with col1:
            feat1, color1 = features[i]
            #Drop NaN values so the KDE line math doesn't fail
            clean_data1 = df[feat1].dropna().values

            #fig1 = px.histogram(df, x=feat1, title=f"Distribution of {feat1}",
            #                    color_discrete_sequence=[color1], 
            #                    marginal="box", opacity=0.7, nbins=30)
            if (len(clean_data1) > 1):
                #Create subplot frame with a secondary y-axis
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                
                #1. Add Histogram on Primary Y-Axis (Left)
                fig1.add_trace(
                    go.Histogram(x=clean_data1, name="Count",  
                                 marker_color=color1, opacity=0.6),
                    secondary_y=False,
                )

                #2. Calculate KDE Math and Add on Secondary Y-Axis (Right)

                kde1 = gaussian_kde(clean_data1)
                x_range1 = np.linspace(clean_data1.min(), clean_data1.max(), 1000)
                y_kde1 = kde1(x_range1)
                fig1.add_trace(
                    go.Scatter(x=x_range1, y=y_kde1, name="KDE", 
                               line=dict(color=color1, width=2.5)),
                    secondary_y=True,
                )

                #Layout adjustments
                fig1.update_layout(
                    title=f"{feat1} Distribution",
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)  # Center the title
                )
                fig1.update_yaxes(title_text="Frequency Count", secondary_y=False)
                fig1.update_yaxes(title_text="Density (KDE)", secondary_y=True)
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if i+1 < len(features):
                feat2, color2 = features[i+1]
                clean_data2 = df[feat2].dropna().values

                if (len(clean_data2) > 1):
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    #1. Add Histogram on Primary Y-Axis (Left)
                    fig2.add_trace(
                        go.Histogram(x=clean_data2, name="Count",  
                                     marker_color=color2, opacity=0.6),
                        secondary_y=False,
                    )
                   
                    #2. Calculate KDE Math and Add on Secondary Y-Axis (Right)
                    kde2 = gaussian_kde(clean_data2)
                    x_range2 = np.linspace(clean_data2.min(), clean_data2.max(), 1000)
                    y_kde2 = kde2(x_range2)
                    fig2.add_trace(
                        go.Scatter(x=x_range2, y=y_kde2, name="KDE", 
                                   line=dict(color=color2, width=2.5)),
                        secondary_y=True,
                    )
                    #Layout adjustments
                    fig2.update_layout(
                        title=f"{feat2} Distribution",
                        showlegend=False,
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20)  # Center the title
                    )
                    fig2.update_yaxes(title_text="Frequency Count", secondary_y=False)
                    fig2.update_yaxes(title_text="Density (KDE)", secondary_y=True)
                    st.plotly_chart(fig2, use_container_width=True)
    st.write("Finding 1: Distribution plots reveal that Resistivity and Gamma Ray curves have log normal distributions as opposed to normal distributions.")
    st.subheader("Step 2: Visualize box plots for each feature to find the anomalous points.")
    boxplot_features = [('Depth', 'red'), ('Gamma Ray', 'olive'), 
                        ('Total Porosity', 'blue'), ('Effective Porosity', 'orange'), 
                        ('Resistivity', 'black'), ('Bulk Density', 'green'), 
                        ('Compression Wave Travel Time', 'cyan'), ('Shear Wave Travel Time', 'brown')
                        ]
    for i in range(0, len(boxplot_features), 2):
        col1, col2 = st.columns(2)

        with col1:
            feat1, color1 = boxplot_features[i]
            if feat1 in df.columns and df[feat1].notna().any():
                fig1 = px.box(df, x=feat1, 
                              title=f"{feat1} Box Plot", 
                              color_discrete_sequence=[color1],
                              orientation='h')
                fig1.update_layout( 
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)  # Center the title
                )
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            if i+1 < len(boxplot_features):
                feat2, color2 = boxplot_features[i+1]
                fig2 = px.box(df, x=feat2, 
                              title=f"{feat2} Box Plot", 
                              color_discrete_sequence=[color2],
                              orientation='h')
                fig2.update_layout( 
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)  # Center the title
                )
                st.plotly_chart(fig2, use_container_width=True)
    st.write("Finding 2: Multiple outliers were identified in Gamma Ray and Resistivity.")
    st.subheader("Step 3: Remove outliers (Resistivity >1000 and Gamma Ray >400) and plot heat map of the Pearson correlation coefficient.")
    if 'Resistivity' in df.columns and 'Gamma Ray' in df.columns:
        df_cleaned = df[((df['Resistivity'] > 0) & (df['Resistivity'] < 1000)) & 
                        ((df['Gamma Ray'] > 0) & (df['Gamma Ray'] < 400))]
        correlation_matrix = df_cleaned.corr(method='pearson')
        fig_corr = px.imshow(correlation_matrix, 
                             text_auto='.2f', 
                             color_continuous_scale='Viridis', 
                             title="Pearson Correlation Coefficient Heatmap",
                             labels=dict(color="Correlation"))
        fig_corr.update_layout(
            height=600,
            margin=dict(l=40, r=40, t=40, b=40)  # Center the title
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    st.write("Finding 3: Pearson correlation coefficient between effective porosity and total porosity is 0.9. Therefore, we drop effective porosity.")

               
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
    margin-bottom: 8px;

}

.reference-item,
.reference-item a,
.reference-item span{
    font-size: 10px !important;
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
            <a href="https://books.google.co.in/books?id=MjoEEAAAQBAJ&lpg=PP1&ots=bgvs7mAGB3&dq=machine%20learning%20guide%20for%20oil%20and%20gas%20using%20python&lr&pg=PP1#v=onepage&q=machine%20learning%20guide%20for%20oil%20and%20gas%20using%20python&f=false" target="_blank" style="text-decoration: underline; display: inline; font-style: italic;">
                Machine learning guide for oil and gas using Python
            </a>. 
        Gulf Professional Publishing.
        </p>
        <p class="reference-item">2. Saleh, K., Mabrouk, W. M., & Metwally, A. (2025).
            <a href="https://doi.org/10.1038/s41598-025-97938-9" target="_blank" style="text-decoration: underline; display: inline; font-style: italic;">
                Machine learning model optimization for compressional sonic log prediction using well logs in Shahd SE field, Western Desert, Egypt
            </a>
             <span style="font-style: italic;">Scientific Reports</span>, <span style="font-style: italic;">15</span>(1), 14957
        </p>
    </div>       
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)