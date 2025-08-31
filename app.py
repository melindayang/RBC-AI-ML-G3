import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# Load pre-trained model and scaler
# ------------------------
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("isolation_forest.pkl")

# File to store auditor feedback
feedback_file = "auditor_feedback.csv"
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
else:
    feedback_df = pd.DataFrame(columns=['TransactionID', 'IsFraud'])

# ------------------------
# Sidebar: upload new transactions
# ------------------------
st.sidebar.title("Upload New Transactions")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)

    # ------------------------
    # Feature engineering
    # ------------------------
    st.sidebar.write("Computing features...")
    df_new['TransactionDate'] = pd.to_datetime(df_new['TransactionDate'])
    df_new = df_new.sort_values(['AccountID','TransactionDate'])
    df_new['PrevTransactionDate'] = df_new.groupby('AccountID')['TransactionDate'].shift(1)
    df_new['TimeSinceLastTxn'] = (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)
    
    # Transaction frequency features
    df_new['TxnCount_24h'] = df_new.groupby('AccountID').rolling('24h', on='TransactionDate')['TransactionDate'].count().reset_index(level=0, drop=True)
    df_new['TxnCount_1h'] = df_new.groupby('AccountID').rolling('1h', on='TransactionDate')['TransactionDate'].count().reset_index(level=0, drop=True)

    # Time of day features
    df_new['Hour'] = df_new['TransactionDate'].dt.hour
    df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    # Merchant features
    df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].apply(lambda x: x.duplicated().map({True:0, False:1}))
    df_new['MerchantFrequency'] = df_new.groupby(['AccountID','MerchantID'])['TransactionID'].transform('count')

    # Select features for model
    feature_cols = ['TransactionAmount','AccountBalance','TransactionDuration',
                    'LoginAttempts','TimeSinceLastTxn','TxnCount_24h','TxnCount_1h',
                    'IsOddHour','MerchantNovelty','MerchantFrequency']

    # Fill missing numeric values
    X_new = df_new[feature_cols].fillna(0)
    X_scaled = scaler.transform(X_new)

    # ------------------------
    # Run Isolation Forest
    # ------------------------
    df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df_new['IsAnomaly'] = iso_forest.predict(X_scaled) == -1

    # ------------------------
    # Display KPIs
    # ------------------------
    st.subheader("KPIs")
    total_txns = len(df_new)
    suspicious_txns = df_new['IsAnomaly'].sum()
    st.metric("Total Transactions", total_txns)
    st.metric("Suspicious Transactions", suspicious_txns)
    st.metric("% Suspicious", f"{suspicious_txns/total_txns:.2%}")

    # ------------------------
    # Anomaly Distribution
    # ------------------------
    st.subheader("Anomaly Score Distribution")
    plt.figure(figsize=(10,5))
    sns.histplot(df_new['AnomalyScore'], bins=50, kde=True)
    st.pyplot(plt)

    # ------------------------
    # Suspicious Transactions Table
    # ------------------------
    st.subheader("Suspicious Transactions")
    anomalies = df_new[df_new['IsAnomaly']]
    st.dataframe(anomalies[['TransactionID','AccountID','MerchantID','TransactionAmount','AnomalyScore','IsAnomaly']].sort_values('AnomalyScore'))

    # ------------------------
    # Auditor labeling
    # ------------------------
    st.subheader("Mark Confirmed Fraudulent Transactions")
    for idx, row in anomalies.iterrows():
        flag = st.checkbox(f"{row['TransactionID']} | {row['AccountID']} | ${row['TransactionAmount']}", key=row['TransactionID'])
        if flag:
            feedback_df = pd.concat([feedback_df, pd.DataFrame({'TransactionID':[row['TransactionID']],'IsFraud':[1]})])
    
    # Save feedback
    feedback_df.drop_duplicates(subset='TransactionID', keep='last', inplace=True)
    feedback_df.to_csv(feedback_file, index=False)
    st.success("Auditor feedback saved! Use this for supervised retraining later.")
