from inference import _ensemble_proba_high, _build_single_row_features
import pandas as pd
import numpy as np
from inference import _lgb_model, _xgb_model, _cb_model

print("BEFORE/AFTER CALIBRATION COMPARISON")
print("=" * 50)

# Test events that showed severe miscalibration in the original diagnosis
test_events = [
    {
        "name": "Vehicle Breakdown",
        "event": {
            "event_cause": "vehicle_breakdown",
            "corridor": "Tumkur Road",
            "priority": "High",
            "requires_road_closure": False,
            "veh_type": "lcv",
            "start_datetime": "2026-06-18T10:30:00Z",
        },
        "actual_rate": 0.016,
        "original_prediction": 0.2481  # From diagnosis
    },
    {
        "name": "Accident",
        "event": {
            "event_cause": "accident",
            "corridor": "ORR East 1",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "heavy_vehicle",
            "start_datetime": "2026-06-18T14:45:00Z",
        },
        "actual_rate": 0.164,
        "original_prediction": 0.1136  # From diagnosis
    },
    {
        "name": "Construction",
        "event": {
            "event_cause": "construction",
            "corridor": "Mysore Road",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-17T19:32:00Z",
        },
        "actual_rate": 0.662,
        "original_prediction": 0.9337  # From diagnosis
    },
    {
        "name": "Public Event",
        "event": {
            "event_cause": "public_event",
            "corridor": "Hosur Road",
            "priority": "Low",  # Using Low to stay in training data
            "requires_road_closure": False,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T20:00:00Z",
        },
        "actual_rate": 0.444,
        "original_prediction": 0.8892  # From diagnosis
    },
    {
        "name": "Pot Holes",
        "event": {
            "event_cause": "pot_holes",
            "corridor": "Electronic City",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T08:00:00Z",
        },
        "actual_rate": 0.860,
        "original_prediction": 0.9615  # From diagnosis
    }
]

print(f"{'Event':<15} {'Original':<10} {'Calibrated':<12} {'Actual':<8} {'Improvement'}")
print("-" * 70)

total_improvement = 0
count = 0

for test in test_events:
    # Get calibrated prediction
    X = _build_single_row_features(test["event"])
    p_high_calibrated, individual_scores = _ensemble_proba_high(X)
    
    # Calculate improvement (reduction in absolute error)
    original_error = abs(test["original_prediction"] - test["actual_rate"])
    calibrated_error = abs(p_high_calibrated - test["actual_rate"])
    improvement = original_error - calibrated_error
    total_improvement += improvement
    count += 1
    
    # Determine if it's an improvement
    imp_status = "IMPROVED" if improvement > 0 else "WORSENED" if improvement < 0 else "SAME"
    
    print(f"{test['name']:<15} {test['original_prediction']:<10.4f} {p_high_calibrated:<12.4f} {test['actual_rate']:<8.3f} {imp_status} ({improvement:+.3f})")

print("-" * 70)
print(f"Average improvement per event: {total_improvement/count:.3f}")
print(f"Total error reduction: {total_improvement:.3f}")

print("\n" + "=" * 50)
print("INDIVIDUAL MODEL SCORES (UNCHANGED)")
print("=" * 50)

# Show that individual model scores are preserved
sample_event = test_events[2]["event"]  # Construction event
X = _build_single_row_features(sample_event)
_, individual_scores = _ensemble_proba_high(X)

# Get raw scores directly from models for comparison
lgb_raw = _lgb_model.predict(X)[0]
xgb_raw = _xgb_model.predict_proba(X)[0][1]
cb_raw = _cb_model.predict_proba(X)[0][1]

print(f"For '{sample_event['event_cause']}' event:")
print(f"  LightGBM: {lgb_raw:.4f} (raw) -> {individual_scores['lightgbm']:.4f} (returned)")
print(f"  XGBoost:  {xgb_raw:.4f} (raw) -> {individual_scores['xgboost']:.4f} (returned)")
print(f"  CatBoost: {cb_raw:.4f} (raw) -> {individual_scores['catboost']:.4f} (returned)")
print("(Individual scores are unchanged - only ensemble probability is calibrated)")
