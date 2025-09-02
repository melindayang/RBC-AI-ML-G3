import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import shap
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
df_new['TimeSinceLastTxn'] = (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)
df_new['TxnCount_24h'] = df_new.groupby('AccountID').rolling('24h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
df_new['TxnCount_1h'] = df_new.groupby('AccountID').rolling('1h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)
df_new['Hour'] = df_new['TransactionDate'].dt.hour
df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Merchant features
df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(lambda x: (~x.duplicated()).astype(int))
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

# Scale incoming features
X_scaled = scaler.transform(features_df)

# Anomaly score
df_new['AnomalyScore'] = iso_forest.decision_function(X_scaled)
threshold = np.percentile(df_new['AnomalyScore'], 5)
df_new['IsAnomaly'] = df_new['AnomalyScore'] <= threshold

# ------------------------
# 4️⃣ SHAP values
# ------------------------
explainer = shap.Explainer(iso_forest, X_scaled)
shap_values = explainer(X_scaled)
abs_shap = np.abs(shap_values.values)

# Keep all non-zero contributing features
feature_list = []
shap_list = []
for i in range(len(df_new)):
    nonzero_idxs = [j for j in range(len(features_df.columns)) if abs_shap[i][j] > 0.0]
    feature_list.append(list(features_df.columns[nonzero_idxs]))
    shap_list.append(list(shap_values.values[i][nonzero_idxs]))

df_new['FeaturesFlagged'] = feature_list
df_new['SHAPValuesFlagged'] = shap_list

# ------------------------
# 5️⃣ Tabs: Transactions & Merchants
# ------------------------
tab1, tab2 = st.tabs(["Suspicious Transactions", "Suspicious Merchants"])

extra_cols = ["TransactionType","Location","DeviceID","IPAddress","Channel",
              "CustomerAge","CustomerOccupation","TransactionDuration",
              "LoginAttempts","AccountBalance","PrevTransactionDate"]

distances = euclidean_distances(X_scaled, X_scaled)
flagged = df_new[df_new['IsAnomaly']].copy()

# ------------------------
# Tab 1: Suspicious Transactions
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

                with st.expander("Reason(s) flagged"):
                    if len(row['FeaturesFlagged'])==0:
                        st.write("No dominant feature; flagged by anomaly model")
                    else:
                        for feat, val in zip(row['FeaturesFlagged'], row['SHAPValuesFlagged']):
                            reason = f"{feat} (impact={val:.4f}
