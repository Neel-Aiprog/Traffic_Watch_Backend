from inference import _ensemble_proba_high, _build_single_row_features
import pandas as pd

# Test a few events to make sure individual scores are unchanged
test_events = [
    {"event_cause": "construction", "corridor": "Mysore Road", "priority": "High", "requires_road_closure": True, "veh_type": "unknown", "start_datetime": "2026-06-17T19:32:00Z"},
    {"event_cause": "accident", "corridor": "ORR East 1", "priority": "High", "requires_road_closure": True, "veh_type": "heavy_vehicle", "start_datetime": "2026-06-18T14:45:00Z"},
    {"event_cause": "vehicle_breakdown", "corridor": "Tumkur Road", "priority": "High", "requires_road_closure": False, "veh_type": "lcv", "start_datetime": "2026-06-18T10:30:00Z"}
]

print("Verifying individual model scores are preserved (should match raw values)")
print("=" * 80)
print(f"{'Event':<20} {'Model':<12} {'Score':<8} {'Source'}")
print("-" * 80)

for i, event in enumerate(test_events):
    cause = event['event_cause']
    X = _build_single_row_features(event)
    
    # Get calibrated ensemble and individual scores from our modified function
    p_high_calibrated, individual_scores = _ensemble_proba_high(X)
    
    # Calculate raw ensemble probability manually for comparison
    from inference import _lgb_model, _xgb_model, _cb_model
    import numpy as np
    lgb_proba = _lgb_model.predict(X)
    xgb_proba = _xgb_model.predict_proba(X)[:, 1]
    cb_proba = _cb_model.predict_proba(X)[:, 1]
    ensemble_raw = float(np.mean([lgb_proba[0], xgb_proba[0], cb_proba[0]]))
    
    print(f"{cause:<20} {'LGBM':<12} {individual_scores['lightgbm']:<8.4f} {'from model'}")
    print(f"{'':<20} {'XGB':<12} {individual_scores['xgboost']:<8.4f} {'from model'}")
    print(f"{'':<20} {'CAT':<12} {individual_scores['catboost']:<8.4f} {'from model'}")
    print(f"{cause:<20} {'ENSEMBLE (raw)':<12} {ensemble_raw:<8.4f} {'calculated'}")
    print(f"{cause:<20} {'ENSEMBLE (cal)':<12} {p_high_calibrated:<8.4f} {'after calibration'}")
    print()
