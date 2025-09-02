import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
    df_new['TransactionDate'] = pd.to_datetime(df_new['TransactionDate'], format='%Y-%m-%d %H:%M:%S')
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

    # ------------------------
    # 3️⃣ Align features with trained scaler
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")

    if hasattr(scaler, 'feature_names_in_'):
        for col in scaler.feature_names_in_:
            if col not in df_new.columns:
                df_new[col] = 0
        features_df = df_new[scaler.feature_names_in_].fillna(0)
    else:
        st.error("Scaler does not have feature_names_in_ attribute. Make sure it was trained on a DataFrame.")
        st.stop()

    # ------------------------
    # 4️⃣ Scale features and predict anomalies
    # ------------------------
    X_scaled = scaler.transform(features_df)
    df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df_new['IsAnomaly'] = iso_forest.predict(X_scaled) == -1

    # ------------------------
    # 4️⃣b Compute SHAP values for feature explanation
    # ------------------------
    explainer = shap.Explainer(iso_forest, X_scaled)
    shap_values = explainer(X_scaled)  # shape: (n_samples, n_features)

    # Get top feature contributing to anomaly for each transaction
    top_features = []
    top_feature_values = []
    for i in range(shap_values.values.shape[0]):
        # Absolute SHAP value for importance
        feature_idx = np.argmax(np.abs(shap_values.values[i]))
        top_features.append(features_df.columns[feature_idx])
        top_feature_values.append(shap_values.values[i][feature_idx])

    df_new['TopAnomalyFeature'] = top_features
    df_new['TopFeatureImpact'] = top_feature_values

    # ------------------------
    # 5️⃣ Display results
    # ------------------------
    st.subheader("Suspicious Transactions with Feature Explanation")
    st.dataframe(
        df_new[df_new['IsAnomaly']][
            ['TransactionID','AccountID','AnomalyScore','TopAnomalyFeature','TopFeatureImpact']
        ]
    )

    st.subheader("Anomaly Score Distribution")
    plt.figure(figsize=(10,5))
    sns.histplot(df_new['AnomalyScore'], bins=50, kde=True)
    st.pyplot(plt)

    top_accounts = df_new[df_new['IsAnomaly']].groupby('AccountID').size().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Accounts by Number of Anomalies")
    st.bar_chart(top_accounts)

    # Optional: SHAP summary plot for all anomalies
    st.subheader("SHAP Summary Plot for Anomalies")
    anomaly_idx = np.where(df_new['IsAnomaly'])[0]
    shap_values_anomalies = shap_values.values[anomaly_idx]
    plt.figure(figsize=(12,6))
    shap.summary_plot(shap_values_anomalies, features_df.iloc[anomaly_idx], plot_type="bar", show=False)
    st.pyplot(plt)
