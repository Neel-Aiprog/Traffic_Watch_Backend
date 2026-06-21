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

with open(f"{MODEL_DIR}/label_encoder_classes_binary.json") as f:
    _label_classes = json.load(f)
with open(f"{MODEL_DIR}/thresholds.json") as f:
    _thresholds = json.load(f)

HIGH_THRESHOLD = _thresholds["high_threshold"]
MEDIUM_THRESHOLD = _thresholds["medium_threshold"]

print(f"Using thresholds: high={HIGH_THRESHOLD}, medium={MEDIUM_THRESHOLD}")

def _build_single_row_features(event: dict) -> pd.DataFrame:
    df = pd.DataFrame([event])
    df["severity"] = "routine"
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    features = build_features(df)
    features = features.drop(columns=["severity"])
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")
    # No remapping - we assume events are from training distribution
    return features[list(_xgb_model.get_booster().feature_names)]

def _ensemble_proba_high_raw(X: pd.DataFrame) -> float:
    lgb_proba = _lgb_model.predict(X)
    xgb_proba = _xgb_model.predict_proba(X)[:, 1]
    cb_proba = _cb_model.predict_proba(X)[:, 1]
    return float(np.mean([lgb_proba[0], xgb_proba[0], cb_proba[0]]))

# Test causes from debug_calibrated.py
causes_to_test = [
    "vehicle_breakdown", "accident", "construction", "water_logging",
    "tree_fall", "pot_holes", "congestion", "road_conditions",
    "public_event", "others",
]

actual_rates = {
    "vehicle_breakdown": 0.016,
    "accident": 0.164,
    "construction": 0.662,
    "water_logging": 0.860,
    "tree_fall": 0.800,
    "pot_holes": 0.860,
    "congestion": 0.050,
    "road_conditions": 0.812,
    "public_event": 0.444,
    "others": 0.528,
}

print("\nCause          Raw P(high)  Actual Rate  Diff")
print("-" * 50)
for cause in causes_to_test:
    payload = {
        "event_cause": cause,
        "corridor": "Mysore Road",
        "priority": "Low",
        "requires_road_closure": False,
        "veh_type": "unknown",
        "start_datetime": "2026-06-18T14:00:00Z",
    }
    X = _build_single_row_features(payload)
    p_high = _ensemble_proba_high_raw(X)
    actual = actual_rates[cause]
    diff = p_high - actual
    print(f"{cause:<14} {p_high:<12.4f} {actual:<12.3f} {diff:+.3f}")

# Also compute what thresholds would give if we wanted to map to tiers
print("\nTier assignment based on current thresholds:")
for cause in causes_to_test:
    payload = {
        "event_cause": cause,
        "corridor": "Mysore Road",
        "priority": "Low",
        "requires_road_closure": False,
        "veh_type": "unknown",
        "start_datetime": "2026-06-18T14:00:00Z",
    }
    X = _build_single_row_features(payload)
    p_high = _ensemble_proba_high_raw(X)
    if p_high >= HIGH_THRESHOLD:
        tier = "HIGH"
    elif p_high >= MEDIUM_THRESHOLD:
        tier = "MEDIUM"
    else:
        tier = "ROUTINE"
    print(f"{cause:<14} p={p_high:.4f} -> {tier}")
