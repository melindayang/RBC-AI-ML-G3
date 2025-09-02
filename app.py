import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

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

    df_new['TimeSinceLastTxn'] = (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)
    df_new['TxnCount_24h'] = df_new.groupby('AccountID').rolling('24h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
    df_new['TxnCount_1h'] = df_new.groupby('AccountID').rolling('1h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
    df_new['Hour'] = df_new['TransactionDate'].dt.hour
    df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(lambda x: (~x.duplicated()).astype(int))
    df_new['MerchantFrequency'] = df_new.groupby(['AccountID','MerchantID'])['TransactionID'].transform('count')

    # ------------------------
    # 3️⃣ Load models and align features
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")

    if hasattr(scaler, 'feature_names_in_'):
        for col in scaler.feature_names_in_:
            if col not in df_new.columns:
                df_new[col] = 0
        features_df = df_new[scaler.feature_names_in_].fillna(0)
    else:
        st.error("Scaler missing feature names. Retrain scaler with a DataFrame.")
        st.stop()

    # ------------------------
    # 4️⃣ Scale features and predict anomalies
    # ------------------------
    X_scaled = scaler.transform(features_df)
    df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)

    # Force ~5% anomalies
    threshold = np.percentile(df_new['AnomalyScore'], 5)
    df_new['IsAnomaly'] = df_new['AnomalyScore'] <= threshold

    # ------------------------
    # 5️⃣ Display suspicious transactions
    # ------------------------
    st.subheader("Suspicious Transactions")

    flagged = df_new[df_new['IsAnomaly']].copy()

    if flagged.empty:
        st.info("No suspicious transactions detected at the 5% threshold.")
    else:
        items_per_page = 10
        total_pages = int(np.ceil(len(flagged) / items_per_page))
        page = st.number_input("Page", 1, total_pages, 1)

        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = flagged.iloc[start_idx:end_idx]

        # Compute distances once for efficiency
        distances = euclidean_distances(X_scaled, X_scaled)

        for _, row in page_data.iterrows():
            with st.expander(f"Transaction {row['TransactionID']} — Account {row['AccountID']}"):
                st.markdown(f"""
                    **TransactionID:** {row['TransactionID']}  
                    **AccountID:** {row['AccountID']}  
                    **MerchantID:** {row['MerchantID']}  
                    **Date:** {row['TransactionDate']}  
                    **TransactionAmount:** {row['TransactionAmount']}
                """)

                # Additional Details
                with st.expander("Additional Details"):
                    st.write({
                        "TransactionType": row.get("TransactionType", None),
                        "Location": row.get("Location", None),
                        "DeviceID": row.get("DeviceID", None),
                        "IP Address": row.get("IPAddress", None),
                        "Channel": row.get("Channel", None),
                        "CustomerAge": row.get("CustomerAge", None),
                        "CustomerOccupation": row.get("CustomerOccupation", None),
                        "TransactionDuration": row.get("TransactionDuration", None),
                        "LoginAttempts": row.get("LoginAttempts", None),
                        "AccountBalance": row.get("AccountBalance", None),
                        "PreviousTransactionDate": row.get("PrevTransactionDate", None),
                    })

                # Reason flagged
                with st.expander("Reason Flagged"):
                    reasons = []
                    if row['TimeSinceLastTxn'] > df_new['TimeSinceLastTxn'].quantile(0.95):
                        reasons.append(f"Unusual gap since last transaction ({row['PrevTransactionDate']} → {row['TransactionDate']})")
                    if row['TxnCount_24h'] > df_new['TxnCount_24h'].quantile(0.95):
                        reasons.append(f"High number of transactions in 24h ({row['TxnCount_24h']})")
                    if row['TxnCount_1h'] > df_new['TxnCount_1h'].quantile(0.95):
                        reasons.append(f"High number of transactions in 1h ({row['TxnCount_1h']})")
                    if row['IsOddHour'] == 1:
                        reasons.append(f"Transaction at odd hour ({row['Hour']}h)")
                    if row['MerchantNovelty'] == 1:
                        reasons.append("New/rare merchant for this account")

                    if reasons:
                        for r in reasons:
                            st.write("- " + r)
                    else:
                        st.write("No single dominant feature; flagged by anomaly model.")

                # Similar Transactions (non-flagged, nearest in feature space)
                with st.expander("Similar Transactions (for reference)"):
                    txn_idx = df_new.index.get_loc(row.name)
                    dists = distances[txn_idx]
                    similar_idxs = np.argsort(dists)[1:21]  # top 20 closest
                    similar = df_new.iloc[similar_idxs]
                    similar = similar[~similar['IsAnomaly']].head(10)

                    for _, srow in similar.iterrows():
                        with st.expander(f"Similar Txn {srow['TransactionID']} — Account {srow['AccountID']}"):
                            st.markdown(f"""
                                **TransactionID:** {srow['TransactionID']}  
                                **AccountID:** {srow['AccountID']}  
                                **MerchantID:** {srow['MerchantID']}  
                                **Date:** {srow['TransactionDate']}  
                                **TransactionAmount:** {srow['TransactionAmount']}
                            """)
                            with st.expander("Additional Details"):
                                st.write({
                                    "TransactionType": srow.get("TransactionType", None),
                                    "Location": srow.get("Location", None),
                                    "DeviceID": srow.get("DeviceID", None),
                                    "IP Address": srow.get("IPAddress", None),
                                    "Channel": srow.get("Channel", None),
                                    "CustomerAge": srow.get("CustomerAge", None),
                                    "CustomerOccupation": srow.get("CustomerOccupation", None),
                                    "TransactionDuration": srow.get("TransactionDuration", None),
                                    "LoginAttempts": srow.get("LoginAttempts", None),
                                    "AccountBalance": srow.get("AccountBalance", None),
                                    "PreviousTransactionDate": srow.get("PrevTransactionDate", None),
                                })

        # ------------------------
        # 6️⃣ Suspicious Merchants
        # ------------------------
        st.subheader("Top 10 Suspicious Merchants")
        top_merchants = flagged.groupby('MerchantID').size().sort_values(ascending=False).head(10).index

        for merchant in top_merchants:
            st.markdown(f"### Merchant {merchant}")
            merchant_txns = flagged[flagged['MerchantID'] == merchant]

            for _, row in merchant_txns.iterrows():
                with st.expander(f"Transaction {row['TransactionID']} — Account {row['AccountID']}"):
                    st.markdown(f"""
                        **TransactionID:** {row['TransactionID']}  
                        **AccountID:** {row['AccountID']}  
                        **Date:** {row['TransactionDate']}  
                        **TransactionAmount:** {row['TransactionAmount']}
                    """)

                    with st.expander("Reason Flagged"):
                        reasons = []
                        if row['TimeSinceLastTxn'] > df_new['TimeSinceLastTxn'].quantile(0.95):
                            reasons.append(f"Unusual gap since last transaction ({row['PrevTransactionDate']} → {row['TransactionDate']})")
                        if row['TxnCount_24h'] > df_new['TxnCount_24h'].quantile(0.95):
                            reasons.append(f"High number of transactions in 24h ({row['TxnCount_24h']})")
                        if row['TxnCount_1h'] > df_new['TxnCount_1h'].quantile(0.95):
                            reasons.append(f"High number of transactions in 1h ({row['TxnCount_1h']})")
                        if row['IsOddHour'] == 1:
                            reasons.append(f"Transaction at odd hour ({row['Hour']}h)")
                        if row['MerchantNovelty'] == 1:
                            reasons.append("New/rare merchant for this account")

                        if reasons:
                            for r in reasons:
                                st.write("- " + r)
                        else:
                            st.write("No single dominant feature; flagged by anomaly model.")
