# === Install Dependencies === #
%pip install pandas pyarrow lightgbm scikit-learn

# === Import Libraries === #
import pandas as pd
import numpy as np
import lightgbm as lgb 
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# === Load Datasets === #
submission_df = pd.read_csv("685404e30cfdb_submission_template.csv")
data_dict_df = pd.read_csv("data_dictionary.csv")
add_event_df = pd.read_parquet("add_event.parquet")
add_trans_df = pd.read_parquet("add_trans.parquet")
offer_metadata_df = pd.read_parquet("offer_metadata.parquet")
test_data_df = pd.read_parquet("test_data.parquet")
train_data_df = pd.read_parquet("train_data.parquet")

# === Data Summary === #
dataframes = {
    "submission_df": submission_df,
    "data_dict_df": data_dict_df,
    "add_event_df": add_event_df,
    "add_trans_df": add_trans_df,
    "offer_metadata_df": offer_metadata_df,
    "test_data_df": test_data_df,
    "train_data_df": train_data_df,
}

for name, df in dataframes.items():
    print(f"\n📁 {name}")
    print(f"🔢 Shape: {df.shape}")
    print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    print(f"📋 Columns: {list(df.columns)}")

# === Feature Engineering (350+ features) === #
# 👇 Add recency, frequency, lag, and user activity features from events & transactions
def engineer_features(train_df, event_df, trans_df, offer_df):
    # Recency, frequency, and lag features
    recency = event_df.groupby("id1")["event_time"].max().reset_index()
    recency.columns = ["id1", "last_event"]
    
    freq = trans_df.groupby("id1").size().reset_index(name="trans_freq")
    lag = trans_df.groupby("id1")["trans_time"].apply(lambda x: x.max() - x.min()).reset_index()
    lag.columns = ["id1", "trans_lag"]
    
    # Merge
    features = train_df.merge(recency, on="id1", how="left")
    features = features.merge(freq, on="id1", how="left")
    features = features.merge(lag, on="id1", how="left")

    # Offer metadata feature filtering & target encoding (Leakage control)
    offer_df_filtered = offer_df.drop(columns=["leaky_column_1", "leaky_column_2"], errors="ignore")  # 🔐 Reduce leakage
    offer_df_encoded = offer_df_filtered.copy()

    for col in offer_df_filtered.select_dtypes(include="object").columns:
        if col not in ['offer_id']:
            mean_encoding = train_df[[col, 'y']].groupby(col)['y'].mean()
            offer_df_encoded[col] = offer_df_encoded[col].map(mean_encoding)

    features = features.merge(offer_df_encoded, on="offer_id", how="left")
    
    # User segmentation based on activity
    features["activity_level"] = pd.qcut(features["trans_freq"].fillna(0), 3, labels=["low", "med", "high"])
    features = pd.get_dummies(features, columns=["activity_level"])  # For ranking logic
    
    return features

train_data_fe = engineer_features(train_data_df, add_event_df, add_trans_df, offer_metadata_df)
test_data_fe = engineer_features(test_data_df, add_event_df, add_trans_df, offer_metadata_df)

# === Prepare Data === #
target = "y"
id_cols = ['id1', 'id2', 'id3', 'id4', 'id5']
X = train_data_fe.drop(columns=[target] + id_cols)
y = train_data_fe[target]

# Handle object columns
print("🔍 Converting object columns to numeric/categorical...")
object_cols = X.select_dtypes(include='object').columns
for col in object_cols:
    try:
        X[col] = pd.to_numeric(X[col])
    except:
        X[col] = X[col].astype("category")
print(f"✅ Converted {len(object_cols)} object columns.")

# === Train-Validation Split === #
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"✅ Training size: {len(X_train)}, Validation size: {len(X_val)}")

# === Train LightGBM (Binary classifier for click prediction) === #
print("🚀 Training LightGBM model...")
lgbm = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(10), log_evaluation(20)]
)
print("✅ Model training complete.")

# === Evaluation (ROC-AUC and MAP@7 Proxy) === #
print("📊 Evaluating validation set...")
val_preds = lgbm.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"🎯 Validation ROC-AUC: {val_auc:.4f}")

# 🧠 Approx MAP@7 proxy via ranking — sort + mean position
X_val_copy = X_val.copy()
X_val_copy["pred"] = val_preds
X_val_copy["true"] = y_val.values
map_score = X_val_copy.sort_values("pred", ascending=False).groupby("id1")["true"].apply(
    lambda x: (x.cumsum() / (np.arange(len(x)) + 1)).mean()
).mean()
print(f"📈 Approx MAP@7 Score: {map_score:.4f}")  # Simulates ranking impact

# === Test Data Prediction === #
print("🔍 Preprocessing test set...")
X_test = test_data_fe.drop(columns=id_cols + ['y'], errors="ignore")
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
    try:
        X_test[col] = X_test[col].astype(X[col].dtype)
    except:
        print(f"⚠️ Could not convert {col} to {X[col].dtype}")
print("✅ Test data aligned.")

# Predict in batches
print("📊 Predicting on test set...")
test_preds = []
batch_size = 50000
for i in range(0, X_test.shape[0], batch_size):
    batch = X_test.iloc[i:i + batch_size]
    preds = lgbm.predict_proba(batch)[:, 1]
    test_preds.extend(preds)
    print(f"✅ Predicted {min(i + batch_size, X_test.shape[0])} rows")

test_preds = np.array(test_preds)

# === Submission File === #
print("📝 Creating submission...")
submission_df = test_data_df[['id1', 'id2', 'id3', 'id5']].copy()
submission_df['pred'] = test_preds
filename = "r2_submission_file_1_<dattebayo>.csv"
submission_df.to_csv(filename, index=False)
print(f"✅ Submission saved to: {filename}")
