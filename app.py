import pandas as pd
import numpy as np

# Assume df_new is your uploaded raw transaction DataFrame
# Ensure TransactionDate is datetime
df_new['TransactionDate'] = pd.to_datetime(df_new['TransactionDate'], format='%Y-%m-%d %H:%M:%S')

# Sort transactions by AccountID and TransactionDate
df_new = df_new.sort_values(['AccountID', 'TransactionDate'])
df_new['PrevTransactionDate'] = df_new.groupby('AccountID')['TransactionDate'].shift(1)

# ------------------------
# Time-based features
# ------------------------
# Time since last transaction in seconds
df_new['TimeSinceLastTxn'] = (df_new['TransactionDate'] - df_new['PrevTransactionDate']).dt.total_seconds().fillna(0)

# Transactions per 24 hours
df_new['TxnCount_24h'] = df_new.groupby('AccountID').rolling('24h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)

# Transactions per 1 hour
df_new['TxnCount_1h'] = df_new.groupby('AccountID').rolling('1h', on='TransactionDate')['TransactionID'].count().reset_index(level=0, drop=True)

# Hour of day & odd hour flag
df_new['Hour'] = df_new['TransactionDate'].dt.hour
df_new['IsOddHour'] = df_new['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# ------------------------
# Merchant features
# ------------------------
# Merchant novelty: 1 if first time merchant for account, 0 otherwise
df_new['MerchantNovelty'] = df_new.groupby('AccountID')['MerchantID'].transform(lambda x: (~x.duplicated()).astype(int))

# Merchant frequency: how many times this merchant appears for this account
df_new['MerchantFrequency'] = df_new.groupby(['AccountID','MerchantID'])['TransactionID'].transform('count')

# ------------------------
# Optional: other numeric features you want to include
# ------------------------
# TransactionAmount, AccountBalance, TransactionDuration, LoginAttempts, etc.
numeric_features = ['TransactionAmount','AccountBalance','TransactionDuration','LoginAttempts',
                    'TimeSinceLastTxn','TxnCount_24h','TxnCount_1h','IsOddHour','MerchantNovelty','MerchantFrequency']

features_df = df_new[numeric_features].fillna(0)
