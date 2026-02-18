import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("🧪 DiaTone - Breath Sensor AI Analyzer (Interactive)")
st.write("Upload your breath sensor Excel/CSV file (with Time (s) and WE(1).Resistance (Ω))")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        st.write(df.head())

        # Check for required columns
        expected_columns = ["Time (s)", "WE(1).Resistance (Ω)"]
        if not all(col in df.columns for col in expected_columns):
            st.error("The file must have columns: 'Time (s)' and 'WE(1).Resistance (Ω)'")
        else:
            x = df["Time (s)"].to_numpy()
            y = df["WE(1).Resistance (Ω)"].to_numpy()

            # 📈 Plot with Plotly
            st.subheader("📈 Breath Sensor Curve (Resistance vs Time)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Resistance'))
            fig.update_layout(
                xaxis_title='Time (s)',
                yaxis_title='Resistance (Ω)',
                title='Interactive Sensor Resistance Curve',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Feature Extraction ---

            # 1. Peak Value & Index
            peak_val = np.max(y)
            peak_index = np.argmax(y)

            # 2. Robust Baseline Detection using 5-point sustained rise
            N = 5
            baseline_index = None
            for i in range(len(y) - N):
                if all(y[i + j] < y[i + j + 1] for j in range(N)) and y[i + N] < peak_val:
                    baseline_index = i
                    break
            if baseline_index is None:
                baseline_index = 0

            baseline_val = y[baseline_index]

            # 3. Peak Height
            peak_height = peak_val - baseline_val

            # 4. Rise Time
            rise_time = x[peak_index] - x[baseline_index]

            # 5. Decay Constant (exponential fit after peak)
            def exp_decay(t, A, k, C):
                return A * np.exp(-k * (t - x[peak_index])) + C

            try:
                decay_x = x[peak_index:]
                decay_y = y[peak_index:]
                popt, _ = curve_fit(exp_decay, decay_x, decay_y, p0=(peak_val, 1, 0))
                decay_constant = popt[1]
            except:
                decay_constant = None

            # 6. Area Under Curve (AUC)
            auc = np.trapz(y, x)

            # 7. FWHM
            half_max = peak_val / 2
            indices_above_half = np.where(y >= half_max)[0]
            if len(indices_above_half) > 1:
                fwhm = x[indices_above_half[-1]] - x[indices_above_half[0]]
            else:
                fwhm = 0

            # 8. Slopes
            rise_slope = peak_height / rise_time if rise_time != 0 else 0
            try:
                end_index = len(y) - 1
                decay_time = x[end_index] - x[peak_index]
                decay_slope = (y[end_index] - peak_val) / decay_time if decay_time != 0 else 0
            except:
                decay_slope = 0

            # 📊 Show extracted features
            st.subheader("📊 Extracted Features (Resistance Curve)")
            features = {
                "Peak Height (Ω)": peak_height,
                "Rise Time (s)": rise_time,
                "Decay Constant (k)": decay_constant if decay_constant is not None else "Fit Failed",
                "Area Under Curve (AUC)": auc,
                "FWHM (s)": fwhm,
                "Rise Slope (Ω/s)": rise_slope,
                "Decay Slope (Ω/s)": decay_slope
            }

            st.write(pd.DataFrame(features.items(), columns=["Feature", "Value"]))

    except Exception as e:
        st.error(f"Error processing file: {e}")
