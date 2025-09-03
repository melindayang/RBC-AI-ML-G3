import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Financial Fraud Detection Dashboard")

# ------------------------
# 1️⃣ Upload raw transaction CSV
# ------------------------
uploaded_file = st.file_uploader("Upload your raw transactions CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded successfully! Shape: {df.shape}")

    # ------------------------
    # 2️⃣ Feature Engineering (same as Colab)
    # ------------------------
    features_df = pd.DataFrame()
    features_df['Amt_to_Balance'] = df['TransactionAmount'] / df['AccountBalance']
    avg_amount = df.groupby('AccountID')['TransactionAmount'].transform('mean')
    features_df['Amt_to_AvgAmt'] = df['TransactionAmount'] / avg_amount
    features_df['Duration_to_Amt'] = df['TransactionDuration'] / df['TransactionAmount']
    
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%Y-%m-%d %H:%M:%S')
    df_sorted = df.sort_values(['AccountID', 'TransactionDate'])
    df_sorted['PrevTransactionDate'] = df_sorted.groupby('AccountID')['TransactionDate'].shift(1)
    features_df['TimeSinceLastTxn'] = (df_sorted['TransactionDate'] - df_sorted['PrevTransactionDate']).dt.total_seconds()
    
    features_df['TxnCount_24h'] = (
        df_sorted.groupby('AccountID', group_keys=False)
                 .apply(lambda g: g.sort_values('TransactionDate')
                                  .rolling('24h', on='TransactionDate')
                                  .TransactionDate.count())
    )
    features_df['TxnCount_1h'] = (
        df_sorted.groupby('AccountID', group_keys=False)
                 .apply(lambda g: g.sort_values('TransactionDate')
                                  .rolling('1h', on='TransactionDate')
                                  .TransactionDate.count())
    )
    
    # Location anomaly
    location_counts = df['Location'].value_counts()
    df['location_frequency'] = df['Location'].map(location_counts) / len(df)
    user_normal_locations = {account: df[df['AccountID']==account]['Location'].mode().iloc[0]
                             for account in df['AccountID'].unique()}
    df['is_unusual_location'] = df.apply(lambda row: 0 if row['Location'] == user_normal_locations.get(row['AccountID'], row['Location']) else 1, axis=1)
    user_location_counts = df.groupby('AccountID')['Location'].nunique()
    df['user_location_diversity'] = df['AccountID'].map(user_location_counts)
    
    # KMeans location anomaly
    from sklearn.cluster import KMeans
    loc_features = df[['location_frequency','is_unusual_location','user_location_diversity']].values
    loc_features_scaled = StandardScaler().fit_transform(loc_features)
    n_clusters = min(5, len(df['Location'].unique()))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(loc_features_scaled)
    cluster_centers = kmeans.cluster_centers_
    distances = [np.linalg.norm(loc_features_scaled[i]-cluster_centers[cluster_labels[i]]) for i in range(len(df))]
    max_distance = max(distances) if distances else 1
    features_df['LocationAnomalyScore'] = [d/max_distance for d in distances]
    
    # Merchant features
    df_sorted['newMerchant'] = (~df_sorted.groupby('AccountID')['MerchantID'].transform(lambda x: x.duplicated())).astype(int)
    merchant_counts = df_sorted.groupby(['AccountID','MerchantID']).size().rename('MerchantCount')
    account_counts = df_sorted.groupby('AccountID').size().rename('AccountTxnCount')
    df_sorted = df_sorted.join(merchant_counts,on=['AccountID','MerchantID']).join(account_counts,on='AccountID')
    df_sorted['MerchantFrequencyScore'] = df_sorted['MerchantCount']/df_sorted['AccountTxnCount']
    
    features_df['TxnAmt'] = df['TransactionAmount']
    features_df['LoginAttempts_log'] = np.log1p(df['LoginAttempts'])
    features_df['TxnDuration'] = df['TransactionDuration']

    # ------------------------
    # 3️⃣ Load scaler, model, feature names
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")
    feature_names = joblib.load("feature_names.pkl")
    
    # Ensure same order & missing columns filled
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[feature_names]

    # ------------------------
    # 4️⃣ Scale + predict
    # ------------------------
    X_scaled = scaler.transform(features_df)
    df['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df['IsAnomaly'] = iso_forest.predict(X_scaled) == -1

    # ------------------------
    # 5️⃣ SHAP contributions
    # ------------------------
    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X_scaled)
    abs_shap = np.abs(shap_values)
    
    # Store non-zero contributions per transaction
    feature_flags = []
    shap_flags = []
    for i in range(len(df)):
        nonzero_idxs = [j for j in range(len(feature_names)) if abs_shap[i][j] != 0.0]
        feature_flags.append([feature_names[j] for j in nonzero_idxs])
        shap_flags.append([shap_values[i][j] for j in nonzero_idxs])
    df['FeaturesFlagged'] = feature_flags
    df['SHAPValuesFlagged'] = shap_flags

    # ------------------------
    # 6️⃣ UI: Tabs for Transactions and Merchants
    # ------------------------
    tab1, tab2 = st.tabs(["Suspicious Transactions", "Suspicious Merchants"])
    
    # ---------------- Transactions ----------------
    with tab1:
        suspicious_df = df[df['IsAnomaly']]
        st.subheader(f"Suspicious Transactions ({len(suspicious_df)})")

        # Pagination
        page_size = 10
        total_pages = (len(suspicious_df)-1)//page_size + 1
        page = st.number_input("Page", 1, total_pages, 1)
        start = (page-1)*page_size
        end = start + page_size
        page_df = suspicious_df.iloc[start:end]

        for idx, row in page_df.iterrows():
            with st.expander(f"Transaction {row['TransactionID']} | Account {row['AccountID']} | Merchant {row['MerchantID']} | Amount {row['TransactionAmount']}"):
                st.write(f"**Date:** {row['TransactionDate']}")
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
                        "PreviousTransactionDate": row.get("PrevTransactionDate")
                    })
                with st.expander("Reason(s) flagged"):
                    for feat, val in zip(row['FeaturesFlagged'], row['SHAPValuesFlagged']):
                        reason = f"{feat} (impact={val:.4f})"
                        if feat == "TimeSinceLastTxn":
                            reason += f" → TxnDate={row['TransactionDate']}, PrevTxnDate={row['PrevTransactionDate']}"
                        if feat == "TxnCount_24h":
                            reason += f" → {row['TxnCount_24h']} txns in 24h"
                        if feat == "TxnCount_1h":
                            reason += f" → {row['TxnCount_1h']} txns in 1h"
                        if feat in ["newMerchant","MerchantFrequencyScore"]:
                            reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequencyScore']}"
                        st.write("- "+reason)
                # Similar transactions (same AccountID, same MerchantID)
                similar = df[(df['AccountID']==row['AccountID']) & (df['MerchantID']==row['MerchantID']) & (df['TransactionID']!=row['TransactionID'])]
                st.write("**Similar transactions:**")
                for sidx, srow in similar.head(10).iterrows():
                    st.write(f"Transaction {srow['TransactionID']} | Amount {srow['TransactionAmount']} | Date {srow['TransactionDate']}")

    # ---------------- Merchants ----------------
    with tab2:
        suspicious_merchants = df[df['IsAnomaly']].groupby('MerchantID').size().sort_values(ascending=False).head(10)
        st.subheader("Top 10 Suspicious Merchants")
        for merchant_id, count in suspicious_merchants.items():
            st.markdown(f"### Merchant {merchant_id} ({count} suspicious txns)")
            merchant_txns = df[(df['MerchantID']==merchant_id) & (df['IsAnomaly'])]
            for idx, row in merchant_txns.iterrows():
                with st.expander(f"Transaction {row['TransactionID']} | Account {row['AccountID']} | Amount {row['TransactionAmount']}"):
                    with st.expander("Reason(s) flagged"):
                        for feat, val in zip(row['FeaturesFlagged'], row['SHAPValuesFlagged']):
                            reason = f"{feat} (impact={val:.4f})"
                            if feat == "TimeSinceLastTxn":
                                reason += f" → TxnDate={row['TransactionDate']}, PrevTxnDate={row['PrevTransactionDate']}"
                            if feat == "TxnCount_24h":
                                reason += f" → {row['TxnCount_24h']} txns in 24h"
                            if feat == "TxnCount_1h":
                                reason += f" → {row['TxnCount_1h']} txns in 1h"
                            if feat in ["newMerchant","MerchantFrequencyScore"]:
                                reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequencyScore']}"
                            st.write("- "+reason)
