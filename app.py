import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit

# 1. ENHANCED PAGE CONFIG
st.set_page_config(
    page_title="DiaTone AI", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. HIDE SIDEBAR & FOOTER (CSS Trick for Mobile)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

# 3. MOBILE FRIENDLY TITLE
st.markdown("### 🧪 DiaTone - Breath AI Analyzer")
st.write("Upload your Excel/CSV file below:")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded!")
        
        # Using a container for the data preview to save space
        with st.expander("View Raw Data"):
            st.write(df.head())

        expected_columns = ["Time (s)", "WE(1).Resistance (Ω)"]
        if not all(col in df.columns for col in expected_columns):
            st.error("Missing: 'Time (s)' and 'WE(1).Resistance (Ω)'")
        else:
            x = df["Time (s)"].to_numpy()
            y = df["WE(1).Resistance (Ω)"].to_numpy()

            # 📈 FULL WIDTH PLOT
            st.subheader("📈 Resistance Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Resistance', line=dict(color='#1976D2')))
            
            fig.update_layout(
                xaxis_title='Time (s)',
                yaxis_title='Res (Ω)',
                # Reduce margins for mobile screens
                margin=dict(l=10, r=10, t=30, b=10),
                height=350, # Fixed height works better in WebViews
                hovermode='x unified'
            )
            # This 'use_container_width' is key!
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # --- Feature Extraction (Your existing logic stays the same) ---
            peak_val = np.max(y)
            peak_index = np.argmax(y)
            
            # ... [Keep your calculation logic exactly as you have it] ...
            
            # 📊 IMPROVED FEATURE TABLE FOR MOBILE
            st.subheader("📊 Features")
            features = {
                "Peak Height": f"{peak_val - y[0]:.2f} Ω", # Example simplification
                "Rise Time": f"{x[peak_index] - x[0]:.2f} s",
                "AUC": f"{np.trapz(y, x):.2f}"
            }
            # Use columns to make it look like a dashboard
            col1, col2 = st.columns(2)
            for i, (k, v) in enumerate(features.items()):
                if i % 2 == 0: col1.metric(k, v)
                else: col2.metric(k, v)

    except Exception as e:
        st.error(f"Error: {e}")
