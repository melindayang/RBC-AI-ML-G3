import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
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
    df_new['TransactionDate'] = pd.to_datetime(
        df_new['TransactionDate'], format='%Y-%m-%d %H:%M:%S'
    )
    df_new = df_new.sort_values(['AccountID', 'TransactionDate'])
    df_new['PrevTransactionDate'] = df_new.groupby('AccountID')['TransactionDate'].shift(1)

    # Time-based features
    df_new['TimeSinceLastTxn'] = (
        df_new['TransactionDate'] - df_new['PrevTransactionDate']
    ).dt.total_seconds().fillna(0)
    df_new['TxnCount_24h'] = (
        df_new.groupby('AccountID')
        .rolling('24h', on='TransactionDate')['TransactionID']
        .count()
        .reset_index(level=0, drop=True)
    )
    df_new['TxnCount_1h'] = (
        df_new.groupby('AccountID')
        .rolling('1h', on='TransactionDate')['TransactionID']
        .count()
        .reset_index(level=0, drop=True)
    )
    df_new['Hour'] = df_new['TransactionDate'].dt.hour
    df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    # Merchant features
    df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(
        lambda x: (~x.duplicated()).astype(int)
    )
    df_new['MerchantFrequency'] = df_new.groupby(['AccountID', 'MerchantID'])[
        'TransactionID'
    ].transform('count')

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
        st.error(
            "Scaler does not have feature_names_in_ attribute. "
            "Make sure it was trained on a DataFrame."
        )
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
    shap_values = explainer(X_scaled)

    top_features = []
    top_feature_values = []
    for i in range(shap_values.values.shape[0]):
        feature_idx = np.argmax(np.abs(shap_values.values[i]))
        top_features.append(features_df.columns[feature_idx])
        top_feature_values.append(shap_values.values[i][feature_idx])

    df_new['TopAnomalyFeature'] = top_features
    df_new['TopFeatureImpact'] = top_feature_values

    # ------------------------
    # 5️⃣ Suspicious transactions (paginated view)
    # ------------------------
    st.subheader("Suspicious Transactions")

    anomalies = df_new[df_new['IsAnomaly']].reset_index(drop=True)
    page_size = st.slider("Transactions per page", 5, 20, 10)
    total_pages = int(np.ceil(len(anomalies) / page_size))
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    anomalies_page = anomalies.iloc[start_idx:end_idx]

    for idx, row in anomalies_page.iterrows():
        with st.expander(f"Transaction {row['TransactionID']} (Account {row['AccountID']})"):
            # Default transaction details
            st.write("**Transaction Details:**")
            st.json({
                "TransactionID": row['TransactionID'],
                "AccountID": row['AccountID'],
                "MerchantID": row['MerchantID'],
                "TransactionDate": str(row['TransactionDate']),
                "TransactionAmount": row.get("TransactionAmount", "N/A"),
            })

            # Additional details
            with st.expander("Additional Details"):
                extra_cols = [
                    "TransactionType", "Location", "DeviceID", "IP Address", "Channel",
                    "CustomerAge", "CustomerOccupation", "TransactionDuration",
                    "LoginAttempts", "AccountBalance", "PrevTransactionDate"
                ]
                extra_data = {col: row[col] for col in extra_cols if col in row}
                st.json(extra_data)

            # Reason flagged
            st.write("**Reason flagged as suspicious:**")
            flagged_feature = row['TopAnomalyFeature']
            st.write(f"- Feature: `{flagged_feature}`")
            st.write(f"- Impact score: {row['TopFeatureImpact']:.4f}")
            st.write(f"- Anomaly score: {row['AnomalyScore']:.4f}")

            # Contextual raw values for the flagged feature
            if flagged_feature == "TimeSinceLastTxn":
                st.write(f"- Transaction time: {row['TransactionDate']}")
                st.write(f"- Previous transaction time: {row['PrevTransactionDate']}")
            elif flagged_feature in ["TxnCount_24h", "TxnCount_1h"]:
                st.write(f"- Value: {row[flagged_feature]}")
            elif flagged_feature == "IsOddHour":
                st.write(f"- Hour of day: {row['Hour']}")
            elif flagged_feature in ["MerchantNovelty", "MerchantFrequency"]:
                st.write(f"- Merchant ID: {row['MerchantID']}")
                st.write(f"- Merchant frequency: {row['MerchantFrequency']}")

    # ------------------------
    # 6️⃣ Top suspicious merchants
    # ------------------------
    st.subheader("Top 10 Suspicious Merchants")

    top_merchants = anomalies.groupby('MerchantID').size().sort_values(ascending=False).head(10)

    for merchant_id, count in top_merchants.items():
        with st.expander(f"Merchant {merchant_id} — {count} suspicious transactions"):
            merchant_anomalies = anomalies[anomalies['MerchantID'] == merchant_id]

            for idx, row in merchant_anomalies.iterrows():
                with st.expander(f"Transaction {row['TransactionID']} (Account {row['AccountID']})"):
                    st.write("**Transaction Details:**")
                    st.json({
                        "TransactionID": row['TransactionID'],
                        "AccountID": row['AccountID'],
                        "TransactionDate": str(row['TransactionDate']),
                        "TransactionAmount": row.get("TransactionAmount", "N/A"),
                    })

                    with st.expander("Additional Details"):
                        extra_cols = [
                            "TransactionType", "Location", "DeviceID", "IP Address", "Channel",
                            "CustomerAge", "CustomerOccupation", "TransactionDuration",
                            "LoginAttempts", "AccountBalance", "PrevTransactionDate"
                        ]
                        extra_data = {col: row[col] for col in extra_cols if col in row}
                        st.json(extra_data)

                    st.write("**Reason flagged as suspicious:**")
                    flagged_feature = row['TopAnomalyFeature']
                    st.write(f"- Feature: `{flagged_feature}`")
                    st.write(f"- Impact score: {row['TopFeatureImpact']:.4f}")
                    st.write(f"- Anomaly score: {row['AnomalyScore']:.4f}")

                    if flagged_feature == "TimeSinceLastTxn":
                        st.write(f"- Transaction time: {row['TransactionDate']}")
                        st.write(f"- Previous transaction time: {row['PrevTransactionDate']}")
                    elif flagged_feature in ["TxnCount_24h", "TxnCount_1h"]:
                        st.write(f"- Value: {row[flagged_feature]}")
                    elif flagged_feature == "IsOddHour":
                        st.write(f"- Hour of day: {row['Hour']}")
                    elif flagged_feature in ["MerchantNovelty", "MerchantFrequency"]:
                        st.write(f"- Merchant ID: {row['MerchantID']}")
                        st.write(f"- Merchant frequency: {row['MerchantFrequency']}")
