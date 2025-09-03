import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    # 3️⃣ Load scaler, model, and feature order
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")
    feature_names = joblib.load("feature_names.pkl")  # saved during training

    # Enforce feature order
    for col in feature_names:
        if col not in df_new.columns:
            df_new[col] = 0
    features_df = df_new[feature_names].fillna(0)

    # ------------------------
    # 4️⃣ Scale + predict anomalies
    # ------------------------
    X_scaled = scaler.transform(features_df)
    df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df_new['IsAnomaly'] = iso_forest.predict(X_scaled) == -1

    # ------------------------
    # 5️⃣ SHAP values
    # ------------------------
    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X_scaled)  # shape (n_samples, n_features)

    abs_shap = np.abs(shap_values)
    feature_list = []
    shap_list = []
    for i in range(len(df_new)):
        nonzero_idxs = [j for j in range(len(feature_names)) if abs_shap[i][j] > 0.0]
        feature_list.append([feature_names[j] for j in nonzero_idxs])
        shap_list.append([shap_values[i][j] for j in nonzero_idxs])

    df_new['FeaturesFlagged'] = feature_list
    df_new['SHAPValuesFlagged'] = shap_list

    # ------------------------
    # 6️⃣ UI Tabs
    # ------------------------
    tab1, tab2 = st.tabs(["Suspicious Transactions", "Suspicious Merchants"])

    # Suspicious Transactions Tab
    with tab1:
        suspicious_df = df_new[df_new['IsAnomaly']]
        st.subheader(f"Suspicious Transactions ({len(suspicious_df)})")

        # Pagination: 10 per page
        page_size = 10
        total_pages = (len(suspicious_df) - 1) // page_size + 1
        page = st.number_input("Page", 1, total_pages, 1)

        start = (page - 1) * page_size
        end = start + page_size
        page_df = suspicious_df.iloc[start:end]

        for idx, row in page_df.iterrows():
            with st.expander(f"Transaction {row['TransactionID']} | Account {row['AccountID']} | Merchant {row['MerchantID']}"):
                st.write(f"**Date:** {row['TransactionDate']}")
                st.write(f"**Amount:** {row['TransactionAmount']}")

                with st.expander("Additional details"):
                    st.write({
                        "TransactionType": row.get("TransactionType"),
                        "Location": row.get("Location"),
                        "DeviceID": row.get("DeviceID"),
                        "IP Address": row.get("IP Address"),
                        "Channel": row.get("Channel"),
                        "CustomerAge": row.get("CustomerAge"),
                        "CustomerOccupation": row.get("CustomerOccupation"),
                        "TransactionDuration": row.get("TransactionDuration"),
                        "LoginAttempts": row.get("LoginAttempts"),
                        "AccountBalance": row.get("AccountBalance"),
                        "PreviousTransactionDate": row.get("PrevTransactionDate"),
                    })

                with st.expander("Reason(s) flagged"):
                    if len(row['FeaturesFlagged']) == 0:
                        st.write("No dominant feature; flagged by anomaly model")
                    else:
                        for feat, val in zip(row['FeaturesFlagged'], row['SHAPValuesFlagged']):
                            reason = f"{feat} (impact={val:.4f})"
                            if feat == "TimeSinceLastTxn":
                                reason += f" → TxnDate={row['TransactionDate']}, PrevTxnDate={row['PrevTransactionDate']}"
                            if feat == "TxnCount_24h":
                                reason += f" → {row['TxnCount_24h']} txns in 24h"
                            if feat == "TxnCount_1h":
                                reason += f" → {row['TxnCount_1h']} txns in 1h"
                            if feat == "IsOddHour":
                                reason += f" → Hour={row['Hour']}"
                            if feat in ["MerchantNovelty", "MerchantFrequency"]:
                                reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                            st.write("- " + reason)

    # Suspicious Merchants Tab
    with tab2:
        suspicious_merchants = df_new[df_new['IsAnomaly']].groupby('MerchantID').size().sort_values(ascending=False).head(10)
        st.subheader("Top 10 Suspicious Merchants")

        for merchant_id, count in suspicious_merchants.items():
            st.markdown(f"### Merchant {merchant_id} ({count} suspicious txns)")
            merchant_txns = df_new[(df_new['MerchantID'] == merchant_id) & (df_new['IsAnomaly'])]

            for idx, row in merchant_txns.iterrows():
                with st.expander(f"Transaction {row['TransactionID']} | Account {row['AccountID']} | Amount {row['TransactionAmount']}"):
                    st.write(f"**Date:** {row['TransactionDate']}")
                    with st.expander("Reason(s) flagged"):
                        if len(row['FeaturesFlagged']) == 0:
                            st.write("No dominant feature; flagged by anomaly model")
                        else:
                            for feat, val in zip(row['FeaturesFlagged'], row['SHAPValuesFlagged']):
                                reason = f"{feat} (impact={val:.4f})"
                                if feat == "TimeSinceLastTxn":
                                    reason += f" → TxnDate={row['TransactionDate']}, PrevTxnDate={row['PrevTransactionDate']}"
                                if feat == "TxnCount_24h":
                                    reason += f" → {row['TxnCount_24h']} txns in 24h"
                                if feat == "TxnCount_1h":
                                    reason += f" → {row['TxnCount_1h']} txns in 1h"
                                if feat == "IsOddHour":
                                    reason += f" → Hour={row['Hour']}"
                                if feat in ["MerchantNovelty", "MerchantFrequency"]:
                                    reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                                st.write("- " + reason)
