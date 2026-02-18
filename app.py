import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid  # NEW: Modern replacement for np.trapz

st.set_page_config(layout="wide", page_title="DiaTone AI Analyzer")

# Custom CSS to make metric cards look even better
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #1976D2; }
    </style>
    """, unsafe_allow_html=True)

st.title("DiaTone")
st.write("Upload your breath sensor data to see the AI Feature Extraction in action.")

uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        
        # Check for required columns
        # Note: Your Android app saves "Time_ms" and "Raw_Value" 
        # I have added a check to handle both your app format and your desktop format
        if "Time_ms" in df.columns:
            df["Time_ms"] = df["Time_ms"] / 1000.0
            df["Raw_Value"] = df["Raw_Value"]

        expected_columns = ["Time_ms", "Raw_Value"]
        
        if not all(col in df.columns for col in expected_columns):
            st.error(f"Missing columns! Need: {expected_columns}")
        else:
            x = df["Time_ms"].to_numpy()
            y = df["Raw_Value"].to_numpy()

            # --- VISUALIZATION ---
            st.subheader("📈 Breath Sensor Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Resistance', line=dict(color='#1976D2', width=3)))
            fig.update_layout(
                xaxis_title='Time (s)',
                yaxis_title='Intensity',
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- FEATURE EXTRACTION ---

            # 1. Peak & Baseline
            peak_val = np.max(y)
            peak_index = np.argmax(y)
            
            # Robust Baseline Detection
            N = 5
            baseline_index = 0
            for i in range(len(y) - N):
                if all(y[i + j] < y[i + j + 1] for j in range(N)) and y[i + N] < peak_val:
                    baseline_index = i
                    break
            
            baseline_val = y[baseline_index]
            peak_height = peak_val - baseline_val
            rise_time = x[peak_index] - x[baseline_index]

            # 2. Decay Constant
            def exp_decay(t, A, k, C):
                return A * np.exp(-k * (t - x[peak_index])) + C

            try:
                decay_x = x[peak_index:]
                decay_y = y[peak_index:]
                popt, _ = curve_fit(exp_decay, decay_x, decay_y, p0=(peak_val, 1, 0))
                decay_constant = popt[1]
            except:
                decay_constant = 0.0

            # 3. AUC & FWHM
            # FIXED: Using scipy.integrate.trapezoid instead of np.trapz
            auc = trapezoid(y, x) 
            
            half_max = peak_val / 2
            indices_above_half = np.where(y >= half_max)[0]
            fwhm = x[indices_above_half[-1]] - x[indices_above_half[0]] if len(indices_above_half) > 1 else 0

            # 4. Slopes
            rise_slope = peak_height / rise_time if rise_time != 0 else 0
            decay_time = x[-1] - x[peak_index]
            decay_slope = (y[-1] - peak_val) / decay_time if decay_time != 0 else 0

            # --- UI: METRIC CARDS ---
            st.subheader("📊 Key Health Indicators")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            m_col1.metric("Peak Height", f"{peak_height:.1f} ")
            m_col2.metric("Rise Time", f"{rise_time:.2f} s")
            m_col3.metric("Breath Volume (AUC)", f"{auc:.0f}")
            m_col4.metric("FWHM", f"{fwhm:.2f} s")

            # --- DATA TABLE ---
            st.subheader("📝 Detailed Analysis")
            features = {
                "Feature": ["Peak Height", "Rise Time", "Decay Constant (k)", "AUC", "FWHM", "Rise Slope", "Decay Slope"],
                "Value": [f"{peak_height:.2f} ", f"{rise_time:.2f} s", f"{decay_constant:.4f}", f"{auc:.2f}", f"{fwhm:.2f} s", f"{rise_slope:.2f} Ω/s", f"{decay_slope:.2f} Ω/s"]
            }
            st.table(pd.DataFrame(features))

    except Exception as e:
        st.error(f"Error processing file: {e}")
