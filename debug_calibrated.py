from inference import _ensemble_proba_high, _build_single_row_features, _calibrate_probability
import pandas as pd

# Test the same events as in Diagnose_Public_Event.py
causes_to_test = [
    "vehicle_breakdown", "accident", "construction", "water_logging",
    "tree_fall", "pot_holes", "congestion", "road_conditions",
    "public_event", "others",
]

results = []
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
    p_high_raw, individual_scores = _ensemble_proba_high(X)
    # _ensemble_proba_high now returns (calibrated_p_high, individual_scores)
    p_high_calibrated = p_high_raw  # This is already calibrated
    results.append({
        "cause": cause, 
        "raw_p_high": None,  # We don't have access to raw anymore easily
        "calibrated_p_high": p_high_calibrated,
        "lgb": individual_scores["lightgbm"], 
        "xgb": individual_scores["xgboost"], 
        "cb": individual_scores["catboost"]
    })

# Let's also compute raw probabilities separately for comparison
raw_results = []
for cause in causes_to_test:
    payload = {
        "event_cause": cause,
        "corridor": "Mysore Road",
        "priority": "Low",
        "requires_road_closure": False,
        "veh_type": "unknown",
        "start_datetime": "2026-06-18T14:00:00Z",
    }
    # Temporarily access raw probability by copying the ensemble logic
    from inference import _lgb_model, _xgb_model, _cb_model
    import numpy as np
    X = _build_single_row_features(payload)
    lgb_proba = _lgb_model.predict(X)
    xgb_proba = _xgb_model.predict_proba(X)[:, 1]
    cb_proba = _cb_model.predict_proba(X)[:, 1]
    ensemble_raw = float(np.mean([lgb_proba[0], xgb_proba[0], cb_proba[0]]))
    raw_results.append({"cause": cause, "raw_p_high": ensemble_raw})

# Combine results
for i, cause in enumerate(causes_to_test):
    results[i]["raw_p_high"] = raw_results[i]["raw_p_high"]

df_results = pd.DataFrame(results)
print("Cause          Raw P(high)  Calibrated P(high)  LGBM      XGBoost   CatBoost")
print("-" * 70)
for _, row in df_results.iterrows():
    print(f"{row['cause']:<14} {row['raw_p_high']:<12.4f} {row['calibrated_p_high']:<18.4f} {row['lgb']:<9.4f} {row['xgb']:<9.4f} {row['cb']:<9.4f}")

print()
print("Comparison with actual rates from diagnosis:")
print("Cause          Calibrated  Actual  Error")
print("-" * 40)
actual_rates = {
    "vehicle_breakdown": 0.016,
    "accident": 0.164,
    "construction": 0.662,
    "water_logging": 0.860,  # approximation
    "tree_fall": 0.800,      # approximation
    "pot_holes": 0.860,
    "congestion": 0.050,     # approximation
    "road_conditions": 0.812,
    "public_event": 0.444,
    "others": 0.528,
}
for _, row in df_results.iterrows():
    cause = row['cause']
    if cause in actual_rates:
        actual = actual_rates[cause]
        calibrated = row['calibrated_p_high']
        error = abs(calibrated - actual)
        print(f"{cause:<14} {calibrated:<10.4f} {actual:<8.3f} {error:.3f}")
