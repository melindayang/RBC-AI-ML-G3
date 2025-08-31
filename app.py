import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# Load pre-trained model + scaler + sample data
# ------------------------
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("isolation_forest.pkl")
features_df = pd.read_csv("features_df.csv")

# Title
st.title("Fraud Detection Dashboard")

# KPI Metrics
total_txns = len(features_df)
suspicious_txns = features_df['IsAnomaly'].sum()
perc_suspicious = suspicious_txns / total_txns * 100

st.metric("Total Transactions", total_txns)
st.metric("Suspicious Transactions", suspicious_txns)
st.metric("% Suspicious Transactions", f"{perc_suspicious:.2f}%")

# ------------------------
# Anomaly Score Distribution
# ------------------------
st.subheader("Anomaly Score Distribution")
plt.figure(figsize=(10,5))
sns.histplot(features_df['AnomalyScore'], bins=50, kde=True)
st.pyplot(plt)

# ------------------------
# Transactions Over Time
# ------------------------
st.subheader("Transactions Over Time")
plt.figure(figsize=(12,5))
sns.scatterplot(
    x='TransactionDate', 
    y='AnomalyScore', 
    hue='IsAnomaly',
    data=features_df,
    palette={True:'red', False:'blue'},
    alpha=0.6
)
plt.xlabel("Transaction Date")
plt.ylabel("Anomaly Score")
st.pyplot(plt)

# ------------------------
# Top Suspicious Accounts
# ------------------------
st.subheader("Top 10 Accounts by Anomalies")
top_accounts = features_df[features_df['IsAnomaly']].groupby('AccountID').size().sort_values(ascending=False).head(10)
st.bar_chart(top_accounts)

# ------------------------
# High-Risk Transactions Table
# ------------------------
st.subheader("Sample Suspicious Transactions")
st.dataframe(features_df[features_df['IsAnomaly']].sort_values('AnomalyScore').head(20))
