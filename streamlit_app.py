import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import psutil
import os

# --- MONITORING UTILITY ---
def display_performance_monitor():
    """Tracks CPU and RAM usage of the current Streamlit process."""
    process = psutil.Process(os.getpid())
    # Resident Set Size (Physical Memory) in MB
    mem_mb = process.memory_info().rss / (1024 * 1024)
    # CPU usage over a 0.1s window
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Health")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("Memory", f"{mem_mb:.1f} MB")

# --- APP CONFIG ---
st.set_page_config(page_title="Cardio Data Preprocessing", layout="wide")

# --- SIDEBAR & MONITOR ---
st.sidebar.header("App Controls")
st.sidebar.info("This monitor tracks the impact of statistical transformations and plotting.")
display_performance_monitor()

# --- DATA GENERATION ---
# Simulated biomedical dataset
np.random.seed(42)
data = {
    'Patient_ID': np.arange(1, 101),
    'Age': np.random.randint(20, 80, 100),
    'Blood_Pressure': np.append(np.random.randint(90, 140, 95), [300, 310, 320, 330, 340]),  # Outliers
    'Cholesterol': np.append(np.random.randint(150, 250, 95), [500, 510, 520, 530, 540]),  # Outliers
    'Glucose': np.append(np.random.randint(70, 150, 95), [300, 310, 320, 330, 340]),  # Outliers
    'BMI': np.append(np.random.normal(25, 5, 95), [50, 52, 55, 60, 65]),  # Outliers
    'Missing_Feature': [np.nan if i % 10 == 0 else np.random.randint(50, 100) for i in range(100)]
}
df = pd.DataFrame(data)

# --- PREPROCESSING PIPELINE ---

# 1. Detecting outliers using Z-score method
df_zscores = df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']].apply(zscore)
outliers = (df_zscores > 3) | (df_zscores < -3)

# 2. Handling outliers by capping extreme values (Winsorization)
def cap_outliers(series, lower_percentile=5, upper_percentile=95):
    lower, upper = np.percentile(series, [lower_percentile, upper_percentile])
    return np.clip(series, lower, upper)

df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']] = df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']].apply(cap_outliers)

# 3. Handling missing data using mean imputation
imputer = SimpleImputer(strategy='mean')
df['Missing_Feature'] = imputer.fit_transform(df[['Missing_Feature']])

# 4. Standardization (Z-score scaling)
scaler = StandardScaler()
df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']] = scaler.fit_transform(df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']])

# 5. Normalization (Min-Max scaling)
minmax_scaler = MinMaxScaler()
df[['Age', 'Missing_Feature']] = minmax_scaler.fit_transform(df[['Age', 'Missing_Feature']])

# --- STREAMLIT UI ---

st.title("Cardiovascular Risk Prediction - Data Preprocessing")
st.write("This app demonstrates data preprocessing techniques for a cardiovascular risk prediction system.")

# Layout Columns
col_data, col_viz = st.columns([1, 1])

with col_data:
    st.subheader("✅ Preprocessed Data (Head)")
    st.dataframe(df.head(10))
    
    st.subheader("🚩 Outlier Detection")
    st.write(f"Total outliers detected: **{outliers.sum().sum()}**")
    st.write("The Z-score method identified extreme values before Winsorization was applied.")

with col_viz:
    st.subheader("📈 Feature Distribution (Standardized)")
    # Plotting is often the most memory-intensive part of small Streamlit apps
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']], ax=ax, palette="Set2")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Informational Section
st.markdown("""
---
### Preprocessing Steps Applied:
1. **Z-Score Detection:** Identified values significantly far from the mean.
2. **Winsorization:** Capped extreme values to the 5th and 95th percentiles to reduce outlier influence.

3. **Mean Imputation:** Filled missing values in `Missing_Feature`.
4. **Standardization:** Rescaled clinical features to have $\mu=0$ and $\sigma=1$.
5. **Min-Max Scaling:** Normalized age and missing features to a [0, 1] range.
""")