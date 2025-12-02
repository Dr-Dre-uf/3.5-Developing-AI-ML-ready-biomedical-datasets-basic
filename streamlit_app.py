import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Load and Preprocess Data ---

# Simulated biomedical dataset (same as in the notebook)
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

# Detecting outliers using Z-score method
df_zscores = df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']].apply(zscore)
outliers = (df_zscores > 3) | (df_zscores < -3)

# Handling outliers by capping extreme values (Winsorization)
def cap_outliers(series, lower_percentile=5, upper_percentile=95):
    lower, upper = np.percentile(series, [lower_percentile, upper_percentile])
    return np.clip(series, lower, upper)
df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']] = df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']].apply(cap_outliers)

# Handling missing data using mean imputation
imputer = SimpleImputer(strategy='mean')
df['Missing_Feature'] = imputer.fit_transform(df[['Missing_Feature']])

# Standardization (Z-score scaling)
scaler = StandardScaler()
df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']] = scaler.fit_transform(df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']])

# Normalization (Min-Max scaling)
minmax_scaler = MinMaxScaler()
df[['Age', 'Missing_Feature']] = minmax_scaler.fit_transform(df[['Age', 'Missing_Feature']])

# --- Streamlit App ---

st.title("Cardiovascular Risk Prediction - Data Preprocessing")
st.write("This app demonstrates data preprocessing techniques for a cardiovascular risk prediction system.")

# Display the preprocessed data
st.subheader("Preprocessed Data")
st.dataframe(df.head())

# Display outlier detection results
st.subheader("Outlier Detection")
st.write(f"Total outliers detected: {outliers.sum().sum()}")

# Display boxplot
st.subheader("Box Plot of Biomedical Features")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df[['Blood_Pressure', 'Cholesterol', 'Glucose', 'BMI']], ax=ax)
st.pyplot(fig)