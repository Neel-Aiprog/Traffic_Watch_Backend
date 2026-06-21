import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from feature_engineering import build_features, CATEGORICAL_FEATURES, ALL_FEATURE_COLS

MODEL_DIR = "./"
_lgb_model = lgb.Booster(model_file=f"{MODEL_DIR}/model_lgb_binary.txt")
_xgb_model = xgb.XGBClassifier()
_xgb_model.load_model(f"{MODEL_DIR}/model_xgb_binary.json")
_cb_model = CatBoostClassifier()
_cb_model.load_model(f"{MODEL_DIR}/model_catboost_binary.cbm")

def _build_single_row_features(event: dict) -> pd.DataFrame:
    df = pd.DataFrame([event])
    df["severity"] = "routine"
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    features = build_features(df)
    features = features.drop(columns=["severity"])
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")
    return features[list(_xgb_model.get_booster().feature_names)]

def _ensemble_proba_high_raw(X: pd.DataFrame) -> float:
    lgb_proba = _lgb_model.predict(X)
    xgb_proba = _xgb_model.predict_proba(X)[:, 1]
    cb_proba = _cb_model.predict_proba(X)[:, 1]
    return float(np.mean([lgb_proba[0], xgb_proba[0], cb_proba[0]]))

# Load dataset
df = pd.read_csv("dataset.csv")
# Take a random sample of 200 rows
sample = df.sample(n=200, random_state=42).reset_index(drop=True)

# We need the true severity to compute actual high? We have severity column? Let's check.
# The dataset likely has a severity column (maybe string). We'll see.
print("Columns in dataset:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Assuming there is a column 'severity' with values like 'routine', 'high', maybe 'medium'?
# Actually from earlier we know they binary classification: high vs not high.
# Let's check unique values in severity if exists.
if 'severity' in df.columns:
    print("Severity unique values:", df['severity'].unique())
    # Convert to binary: high vs not high
    df['high_severity'] = (df['severity'] == 'high').astype(int)
else:
    print("No severity column; cannot compute actual rates.")

# Build features for each row (this may be slow for 200 rows, but okay)
print("\nProcessing sample...")
raw_probs = []
for idx, row in sample.iterrows():
    event = {
        "event_cause": row["event_cause"],
        "corridor": row["corridor"],
        "priority": row["priority"],
        "requires_road_closure": int(row["requires_road_closure"]),
        "veh_type": row["veh_type"],
        "start_datetime": row["start_datetime"]
    }
    X = _build_single_row_features(event)
    p_raw = _ensemble_proba_high_raw(X)
    raw_probs.append(p_raw)
    
    if idx % 50 == 0:
        print(f"  Processed {idx+1}/200")

raw_probs = np.array(raw_probs)
print(f"\nRaw probability stats: min={raw_probs.min():.3f}, max={raw_probs.max():.3f}, mean={raw_probs.mean():.3f}")

# Apply thresholds from optimization
high_thresh = 0.935
med_thresh = 0.199
tiers = []
for p in raw_probs:
    if p >= high_thresh:
        tiers.append("HIGH")
    elif p >= med_thresh:
        tiers.append("MEDIUM")
    else:
        tiers.append("ROUTINE")
tiers = np.array(tiers)

print("\nTier distribution:")
unique, counts = np.unique(tiers, return_counts=True)
for t, c in zip(unique, counts):
    print(f"{t}: {c} ({c/len(tiers)*100:.1f}%)")

# If we have actual high_severity, compute accuracy
if 'high_severity' in df.columns:
    # Map tiers to binary prediction: HIGH -> 1, MEDIUM/ROUTINE -> 0? Actually MEDIUM is also considered high? 
    # In the original tier system, MEDIUM is still considered elevated but not HIGH.
    # For binary classification of high severity, we only consider HIGH as positive.
    # Let's compute precision/recall for HIGH vs actual high.
    y_pred_high = (tiers == "HIGH").astype(int)
    y_true = sample['high_severity'].values
    from sklearn.metrics import confusion_matrix, classification_report
    print("\nClassification report for HIGH severity (vs others):")
    print(classification_report(y_true, y_pred_high, target_names=['NOT_HIGH', 'HIGH']))
