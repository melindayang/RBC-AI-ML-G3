import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Financial Fraud Detection Dashboard")

# ------------------------
# 1️⃣ Upload raw transaction CSV
# ------------------------
uploaded_file = st.file_uploader("Upload your raw transactions CSV", type="csv")

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded successfully! Shape: {df_new.shape}")

    # ------------------------
    # 2️⃣ Feature Engineering
    # ------------------------
    # Ensure TransactionDate is datetime
    df_new['TransactionDate'] = pd.to_datetime(df_new['TransactionDate'], format='%Y-%m-%d %H:%M:%S')

    # Sort by AccountID and TransactionDate
    df_new = df_new.sort_values(['AccountID', 'TransactionDate'])
    df_new['PrevTransactionDate'] = df_new.groupby('AccountID')['TransactionDate'].shift(1)

    # Time-based features
    df_new['TimeSinceLastTxn'] = (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)
    df_new['TxnCount_24h'] = df_new.groupby('AccountID').rolling('24h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
    df_new['TxnCount_1h'] = df_new.groupby('AccountID').rolling('1h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
    df_new['Hour'] = df_new['TransactionDate'].dt.hour
    df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    # Merchant features
    df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(lambda x: (~x.duplicated()).astype(int))
    df_new['MerchantFrequency'] = df_new.groupby(['AccountID','MerchantID'])['TransactionID'].transform('count')

    # Optional numeric features
    numeric_features = [
        'TransactionAmount','AccountBalance','TransactionDuration','LoginAttempts',
        'TimeSinceLastTxn','TxnCount_24h','TxnCount_1h','IsOddHour','MerchantNovelty','MerchantFrequency'
    ]

    features_df = df_new[numeric_features].fillna(0)

    # ------------------------
    # 3️⃣ Load pre-trained model and scaler
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")

    # Scale features
    X_scaled = scaler.transform(features_df)

    # Predict anomalies
    df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df_new['IsAnomaly'] = iso_forest.predict(X_scaled) == -1

    st.subheader("Suspicious Transactions")
    st.dataframe(df_new[df_new['IsAnomaly']])

    # ------------------------
    # 4️⃣ Plot anomaly scores
    # ------------------------
    st.subheader("Anomaly Score Distribution")
    plt.figure(figsize=(10,5))
    sns.histplot(df_new['AnomalyScore'], bins=50, kde=True)
    st.pyplot(plt)

    # ------------------------
    # 5️⃣ Top accounts by number of anomalies
    # ------------------------
    top_accounts = df_new[df_new['IsAnomaly']].groupby('AccountID').size().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Accounts by Number of Anomalies")
    st.bar_chart(top_accounts)
