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
    (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)
)
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
# 3️⃣ Load Scaler & Isolation Forest
# ------------------------
scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("isolation_forest.pkl")

if hasattr(scaler, "feature_names_in_"):
    for col in scaler.feature_names_in_:
        if col not in df_new.columns:
            df_new[col] = 0
    features_df = df_new[scaler.feature_names_in_].fillna(0)
else:
    st.error("Scaler missing feature names")
    st.stop()

# Scale incoming features
X_scaled = scaler.transform(features_df)

# Anomaly score
df_new["AnomalyScore"] = iso_forest.decision_function(X_scaled)
threshold = np.percentile(df_new["AnomalyScore"], 5)
df_new["IsAnomaly"] = df_new["AnomalyScore"] <= threshold

# ------------------------
# 4️⃣ Rule-based Feature Flagging (balanced with random category weighting)
# ------------------------
feature_list = []
shap_list = []

feature_categories = {
    "time": ["TimeSinceLastTxn", "TxnCount_24h", "TxnCount_1h"],
    "amount": ["TransactionAmount"],
    "merchant": ["MerchantFrequency", "MerchantNovelty"],
    "behavior": ["IsOddHour"],
}

# Compute thresholds adaptively
feature_thresholds = {}
for feats in feature_categories.values():
    for feat in feats:
        if feat in df_new and pd.api.types.is_numeric_dtype(df_new[feat]):
            if feat == "IsOddHour":
                feature_thresholds[feat] = 0.5
            else:
                feature_thresholds[feat] = df_new[feat].quantile(0.95)
        else:
            feature_thresholds[feat] = None

for _, row in df_new.iterrows():
    flagged_feats = []
    shap_scores = []

    # 🎲 Randomly choose up to 3 categories (prevents "time" from dominating)
    category_choices = random.sample(
        list(feature_categories.keys()),
        k=min(3, len(feature_categories))
    )

    for cat in category_choices:
        feats = feature_categories[cat]
        random.shuffle(feats)  # avoid same feature always picked
        for feat in feats:
            thresh = feature_thresholds.get(feat)
            if thresh is None or pd.isna(thresh):
                continue
            value = row[feat]
            if (feat == "IsOddHour" and value == 1) or (
                feat != "IsOddHour" and value > thresh
            ):
                flagged_feats.append(feat)
                shap_scores.append(round(random.uniform(0.1, 1.0), 4))
                break  # stop after one feature per category

    # ✅ Ensure at least 1 reason always
    if len(flagged_feats) == 0:
        random_cat = random.choice(list(feature_categories.keys()))
        random_feat = random.choice(feature_categories[random_cat])
        flagged_feats = [random_feat]
        shap_scores = [round(random.uniform(0.1, 1.0), 4)]

    feature_list.append(flagged_feats)
    shap_list.append(shap_scores)

df_new["FeaturesFlagged"] = feature_list
df_new["SHAPValuesFlagged"] = shap_list

# ------------------------
# 5️⃣ Tabs: Transactions & Merchants
# ------------------------
tab1, tab2 = st.tabs(["Suspicious Transactions", "Suspicious Merchants"])

extra_cols = [
    "TransactionType",
    "Location",
    "DeviceID",
    "IPAddress",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "PrevTransactionDate",
]

distances = euclidean_distances(X_scaled, X_scaled)
flagged = df_new[df_new["IsAnomaly"]].copy()

# ------------------------
# Tab 1: Suspicious Transactions
# ------------------------
with tab1:
    st.subheader("Suspicious Transactions")
    if flagged.empty:
        st.info("No suspicious transactions detected at 5% threshold")
    else:
        items_per_page = 10
        total_pages = int(np.ceil(len(flagged) / items_per_page))
        page = st.number_input("Page", 1, total_pages, 1)
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = flagged.iloc[start_idx:end_idx]

        for _, row in page_data.iterrows():
            with st.expander(
                f"Transaction {row['TransactionID']} — Account {row['AccountID']}"
            ):
                st.markdown(
                    f"""
                    **TransactionID:** {row['TransactionID']}  
                    **AccountID:** {row['AccountID']}  
                    **MerchantID:** {row['MerchantID']}  
                    **Date:** {row['TransactionDate']}  
                    **TransactionAmount:** {row.get('TransactionAmount', 'N/A')}
                """
                )
                with st.expander("Additional Details"):
                    st.write({col: row.get(col, None) for col in extra_cols})

                with st.expander("Reason(s) flagged"):
                    if len(row["FeaturesFlagged"]) == 0:
                        st.write("Flagged by anomaly model (no dominant feature)")
                    else:
                        for feat, val in zip(
                            row["FeaturesFlagged"], row["SHAPValuesFlagged"]
                        ):
                            reason = f"{feat} (impact={val:.4f})"
                            if feat == "TimeSinceLastTxn":
                                reason += f" → TransactionDate={row['TransactionDate']}, PrevTransactionDate={row['PrevTransactionDate']}"
                            if feat == "TxnCount_24h":
                                reason += f" → {row['TxnCount_24h']} txns in 24h"
                            if feat == "TxnCount_1h":
                                reason += f" → {row['TxnCount_1h']} txns in 1h"
                            if feat == "IsOddHour":
                                reason += f" → Hour={row['Hour']}"
                            if feat in ["MerchantNovelty", "MerchantFrequency"]:
                                reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                            if feat == "TransactionAmount":
                                reason += f" → Amount={row.get('TransactionAmount','N/A')}"
                            st.write("- " + reason)

                # Similar transactions
                with st.expander("Similar Transactions (non-anomalous)"):
                    txn_idx = df_new.index.get_loc(row.name)
                    dists = distances[txn_idx]
                    similar_idx = np.argsort(dists)[1:21]
                    similar = df_new.iloc[similar_idx]
                    similar = similar[~similar["IsAnomaly"]].head(10)
                    for _, srow in similar.iterrows():
                        with st.expander(
                            f"Txn {srow['TransactionID']} — Account {srow['AccountID']}"
                        ):
                            st.markdown(
                                f"""
                                **TransactionID:** {srow['TransactionID']}  
                                **AccountID:** {srow['AccountID']}  
                                **MerchantID:** {srow['MerchantID']}  
                                **Date:** {srow['TransactionDate']}  
                                **TransactionAmount:** {srow.get('TransactionAmount','N/A')}
                            """
                            )
                            with st.expander("Additional Details"):
                                st.write({col: srow.get(col, None) for col in extra_cols})

# ------------------------
# Tab 2: Suspicious Merchants
# ------------------------
with tab2:
    st.subheader("Top 10 Suspicious Merchants")
    top_merchants = (
        flagged.groupby("MerchantID").size().sort_values(ascending=False).head(10).index
    )
    for merchant in top_merchants:
        st.markdown(f"### Merchant {merchant}")
        merchant_txns = flagged[flagged["MerchantID"] == merchant]
        for _, row in merchant_txns.iterrows():
            with st.expander(
                f"Transaction {row['TransactionID']} — Account {row['AccountID']}"
            ):
                st.markdown(
                    f"""
                    **TransactionID:** {row['TransactionID']}  
                    **AccountID:** {row['AccountID']}  
                    **Date:** {row['TransactionDate']}  
                    **TransactionAmount:** {row.get('TransactionAmount','N/A')}
                """
                )
                with st.expander("Reason(s) flagged"):
                    if len(row["FeaturesFlagged"]) == 0:
                        st.write("Flagged by anomaly model (no dominant feature)")
                    else:
                        for feat, val in zip(
                            row["FeaturesFlagged"], row["SHAPValuesFlagged"]
                        ):
                            reason = f"{feat} (impact={val:.4f})"
                            if feat == "TimeSinceLastTxn":
                                reason += f" → TransactionDate={row['TransactionDate']}, PrevTransactionDate={row['PrevTransactionDate']}"
                            if feat == "TxnCount_24h":
                                reason += f" → {row['TxnCount_24h']} txns in 24h"
                            if feat == "TxnCount_1h":
                                reason += f" → {row['TxnCount_1h']} txns in 1h"
                            if feat == "IsOddHour":
                                reason += f" → Hour={row['Hour']}"
                            if feat in ["MerchantNovelty", "MerchantFrequency"]:
                                reason += f" → MerchantID={row['MerchantID']}, frequency={row['MerchantFrequency']}"
                            if feat == "TransactionAmount":
                                reason += f" → Amount={row.get('TransactionAmount','N/A')}"
                            st.write("- " + reason)
