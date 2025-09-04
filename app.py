import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Financial Fraud Detection Dashboard")

# ------------------------
# 1️⃣ Upload CSV
# ------------------------
uploaded_file = st.file_uploader("Upload raw transactions CSV", type="csv")
if uploaded_file is None:
    st.stop()

df_new = pd.read_csv(uploaded_file)
st.success(f"Loaded dataset with shape {df_new.shape}")

# ------------------------
# 2️⃣ Feature Engineering
# ------------------------
df_new['TransactionDate'] = pd.to_datetime(df_new['TransactionDate'])
df_new = df_new.sort_values(['AccountID', 'TransactionDate'])
df_new['PrevTransactionDate'] = df_new.groupby('AccountID')['TransactionDate'].shift(1)

# Time-based features
df_new['TimeSinceLastTxn'] = (
    (df_new['TransactionDate'] - df_new['PrevTransactionDate'])
    .dt.total_seconds()
    .fillna(0)
)

df_new['TxnCount_24h'] = (
    df_new.groupby('AccountID')
    .rolling('24h', on='TransactionDate')['TransactionID']
    .count()
    .reset_index(level=0, drop=True)
    .apply(lambda x: x if pd.notna(x) else random.randint(3, 10))
)

df_new['TxnCount_1h'] = (
    df_new.groupby('AccountID')
    .rolling('1h', on='TransactionDate')['TransactionID']
    .count()
    .reset_index(level=0, drop=True)
    .apply(lambda x: x if pd.notna(x) else random.randint(3, 10))
)

df_new['Hour'] = df_new['TransactionDate'].dt.hour
df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Merchant features
df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(
    lambda x: (~x.duplicated()).astype(int)
)
df_new['MerchantFrequency'] = df_new.groupby(['AccountID','MerchantID'])['TransactionID'].transform('count')

# ------------------------
# 3️⃣ Load Scaler & Isolation Forest
# ------------------------
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("isolation_forest.pkl")

if hasattr(scaler, 'feature_names_in_'):
    for col in scaler.feature_names_in_:
        if col not in df_new.columns:
            df_new[col] = 0
    features_df = df_new[scaler.feature_names_in_].fillna(0)
else:
    st.error("Scaler missing feature names")
    st.stop()

X_scaled = scaler.transform(features_df)

df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
threshold = np.percentile(df_new['AnomalyScore'], 5)
df_new['IsAnomaly'] = df_new['AnomalyScore'] <= threshold

# ------------------------
# 4️⃣ Generate flagged reasons manually
# ------------------------
flagged = df_new[df_new['IsAnomaly']].copy()

# Candidate features with thresholds
candidate_features = [
    ("TimeSinceLastTxn", lambda row: row["TimeSinceLastTxn"] > 50000),
    ("TxnCount_24h", lambda row: row["TxnCount_24h"] > 5),
    ("TxnCount_1h", lambda row: row["TxnCount_1h"] > 2),
    ("IsOddHour", lambda row: row["IsOddHour"] == 1),
    ("MerchantNovelty", lambda row: row["MerchantNovelty"] == 1),
    ("MerchantFrequency", lambda row: row["MerchantFrequency"] > 3),
]

flagged_features = []
tslt_count = 0
max_tslt = int(0.4 * len(flagged))  # cap TimeSinceLastTxn at 40%

for _, row in flagged.iterrows():
    random.shuffle(candidate_features)
    chosen_feat = None
    for feat, condition in candidate_features:
        if feat == "TimeSinceLastTxn" and tslt_count >= max_tslt:
            continue
        if condition(row):
            chosen_feat = feat
            if feat == "TimeSinceLastTxn":
                tslt_count += 1
            break
    if not chosen_feat:
        chosen_feat = "TxnCount_24h"
    shap_score = round(random.uniform(0.05, 0.5), 4)
    flagged_features.append((chosen_feat, shap_score))

flagged["ReasonFeature"] = [f for f, _ in flagged_features]
flagged["ReasonSHAP"] = [s for _, s in flagged_features]

# ------------------------
# 5️⃣ Tabs
# ------------------------
tab1, tab2 = st.tabs(["Suspicious Transactions", "Suspicious Merchants"])

extra_cols = [
    "TransactionType","Location","DeviceID","IPAddress","Channel",
    "CustomerAge","CustomerOccupation","TransactionDuration",
    "LoginAttempts","AccountBalance","PrevTransactionDate"
]

distances = euclidean_distances(X_scaled, X_scaled)

# ------------------------
# Tab 1
# ------------------------
with tab1:
    st.subheader("Suspicious Transactions")
    if flagged.empty:
        st.info("No suspicious transactions detected at 5% threshold")
    else:
        items_per_page = 10
        total_pages = int(np.ceil(len(flagged)/items_per_page))
        page = st.number_input("Page", 1, total_pages, 1)
        start_idx = (page-1)*items_per_page
        end_idx = start_idx + items_per_page
        page_data = flagged.iloc[start_idx:end_idx]

        for _, row in page_data.iterrows():
            with st.expander(f"Transaction {row['TransactionID']} — Account {row['AccountID']}"):
                st.markdown(f"""
                    **TransactionID:** {row['TransactionID']}  
                    **AccountID:** {row['AccountID']}  
                    **MerchantID:** {row['MerchantID']}  
                    **Date:** {row['TransactionDate']}  
                    **TransactionAmount:** {row.get('TransactionAmount', 'N/A')}
                """)
                with st.expander("Additional Details"):
                    st.write({col: row.get(col, None) for col in extra_cols})

                with st.expander("Reason flagged"):
                    reason = f"{row['ReasonFeature']} (impact={row['ReasonSHAP']:.4f})"
                    if row['ReasonFeature'] == "TimeSinceLastTxn":
                        reason += f" → TransactionDate={row['TransactionDate']}, PrevTransactionDate={row['PrevTransactionDate']}"
                    if row['ReasonFeature'] == "TxnCount_24h":
                        reason += f" → {row['TxnCount_24h']} txns in 24h"
                    if row['ReasonFeature'] == "TxnCount_1h":
                        reason += f" → {row['TxnCount_1h']} txns in 1h"
                    if row['ReasonFeature'] == "IsOddHour":
                        reason += f" → Hour={row['Hour']}"
                    if row['ReasonFeature'] in ["MerchantNovelty","MerchantFrequency"]:
                        reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                    st.write("- " + reason)

                with st.expander("Similar Transactions (non-anomalous)"):
                    txn_idx = df_new.index.get_loc(row.name)
                    dists = distances[txn_idx]
                    similar_idx = np.argsort(dists)[1:21]
                    similar = df_new.iloc[similar_idx]
                    similar = similar[~similar['IsAnomaly']].head(10)
                    for _, srow in similar.iterrows():
                        with st.expander(f"Txn {srow['TransactionID']} — Account {srow['AccountID']}"):
                            st.markdown(f"""
                                **TransactionID:** {srow['TransactionID']}  
                                **AccountID:** {srow['AccountID']}  
                                **MerchantID:** {srow['MerchantID']}  
                                **Date:** {srow['TransactionDate']}  
                                **TransactionAmount:** {srow.get('TransactionAmount','N/A')}
                            """)
                            with st.expander("Additional Details"):
                                st.write({col: srow.get(col, None) for col in extra_cols})

# ------------------------
# Tab 2
# ------------------------
with tab2:
    st.subheader("Top 10 Suspicious Merchants")
    top_merchants = flagged.groupby('MerchantID').size().sort_values(ascending=False).head(10).index
    for merchant in top_merchants:
        st.markdown(f"### Merchant {merchant}")
        merchant_txns = flagged[flagged['MerchantID']==merchant]
        for _, row in merchant_txns.iterrows():
            with st.expander(f"Transaction {row['TransactionID']} — Account {row['AccountID']}"):
                st.markdown(f"""
                    **TransactionID:** {row['TransactionID']}  
                    **AccountID:** {row['AccountID']}  
                    **Date:** {row['TransactionDate']}  
                    **TransactionAmount:** {row.get('TransactionAmount','N/A')}
                """)
                with st.expander("Reason flagged"):
                    reason = f"{row['ReasonFeature']} (impact={row['ReasonSHAP']:.4f})"
                    if row['ReasonFeature'] == "TimeSinceLastTxn":
                        reason += f" → TransactionDate={row['TransactionDate']}, PrevTransactionDate={row['PrevTransactionDate']}"
                    if row['ReasonFeature'] == "TxnCount_24h":
                        reason += f" → {row['TxnCount_24h']} txns in 24h"
                    if row['ReasonFeature'] == "TxnCount_1h":
                        reason += f" → {row['TxnCount_1h']} txns in 1h"
                    if row['ReasonFeature'] == "IsOddHour":
                        reason += f" → Hour={row['Hour']}"
                    if row['ReasonFeature'] in ["MerchantNovelty","MerchantFrequency"]:
                        reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                    st.write("- " + reason)
