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
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Health")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("Memory", f"{mem_mb:.1f} MB")

# --- APP CONFIG ---
st.set_page_config(page_title="Genomic Data Preprocessing", layout="wide")

# --- SIDEBAR & MONITOR ---
st.sidebar.header("App Controls")
st.sidebar.info("This monitor tracks the impact of statistical transformations on genomic datasets.")
display_performance_monitor()

# --- DATA GENERATION (Aligned with Notebook Logic) ---
# Simulated genomic dataset mirroring the structure in source 19
np.random.seed(42)
data = {
    'Gene_ID': np.arange(1, 101),
    'Sequencing_Depth': np.random.randint(20, 80, 100),
    'Expression_Level_A': np.append(np.random.randint(90, 140, 95), [300, 310, 320, 330, 340]),  # Outliers
    'Expression_Level_B': np.append(np.random.randint(150, 250, 95), [500, 510, 520, 530, 540]),  # Outliers
    'Expression_Level_C': np.append(np.random.randint(70, 150, 95), [300, 310, 320, 330, 340]),  # Outliers
    'Replicate_Variance': np.append(np.random.normal(25, 5, 95), [50, 52, 55, 60, 65]),  # Outliers
    'Missing_Expression_Profile': [np.nan if i % 10 == 0 else np.random.randint(50, 100) for i in range(100)]
}
df = pd.DataFrame(data)

# --- PREPROCESSING PIPELINE (Direct Alignment with Notebook) ---

# 1. Detecting outliers using Z-score method
