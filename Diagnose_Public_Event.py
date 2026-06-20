"""
check_calibration.py — run in your backend folder.

Compares the model's predicted P(high) for a fixed, neutral set of
conditions across multiple event_causes against each cause's ACTUAL
high-rate in the training data. Large, consistent gaps would indicate
a calibration problem; gaps specific to low-sample causes would point
to overfitting on sparse categories.
"""

import pandas as pd
from inference import predict_severity

# Use Mysore Road + a neutral daytime timestamp + Low priority + no closure
# for every cause, to isolate the effect of event_cause alone.
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
    pred = predict_severity(payload)
    results.append({"cause": cause, "predicted_p_high": pred["probability_high"]})

df_results = pd.DataFrame(results).sort_values("predicted_p_high")
print(df_results.to_string(index=False))
print()
print("Compare these predicted_p_high values against the actual high-rate")
print("table from earlier (vehicle_breakdown=0.016, accident=0.164,")
print("public_event=0.444, others=0.528, construction=0.662,")
print("road_conditions=0.812, pot_holes=0.860). Large mismatches,")
print("especially for low-sample causes, indicate overfitting.")