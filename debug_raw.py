from inference import _ensemble_proba_high, _build_single_row_features
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
    p_high, individual_scores = _ensemble_proba_high(X)
    results.append({"cause": cause, "raw_p_high": p_high, "lgb": individual_scores["lightgbm"], "xgb": individual_scores["xgboost"], "cb": individual_scores["catboost"]})

df_results = pd.DataFrame(results)
print("Raw ensemble probabilities:")
print(df_results.to_string(index=False))
print()
print("Individual model scores:")
print(df_results.to_string(index=False))
