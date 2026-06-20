from inference import predict_severity
from recommendation import get_recommendation

test_events = [
    {
        "name": "Vehicle Breakdown (Low Actual Rate)",
        "event": {
            "event_cause": "vehicle_breakdown",
            "corridor": "Tumkur Road",
            "priority": "High",  # Only High/Low in training data
            "requires_road_closure": False,
            "veh_type": "lcv",
            "start_datetime": "2026-06-18T10:30:00Z",
        },
        "actual_rate": 0.016
    },
    {
        "name": "Accident (Moderate Actual Rate)",
        "event": {
            "event_cause": "accident",
            "corridor": "ORR East 1",
            "priority": "High",  # Only High/Low in training data
            "requires_road_closure": True,
            "veh_type": "heavy_vehicle",
            "start_datetime": "2026-06-18T14:45:00Z",
        },
        "actual_rate": 0.164
    },
    {
        "name": "Construction (High Actual Rate)",
        "event": {
            "event_cause": "construction",
            "corridor": "Mysore Road",
            "priority": "High",  # Only High/Low in training data
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-17T19:32:00Z",
        },
        "actual_rate": 0.662
    },
    {
        "name": "Public Event (Very High Actual Rate)",
        "event": {
            "event_cause": "public_event",
            "corridor": "Hosur Road",
            "priority": "Low",  # Using Low to stay in training distribution
            "requires_road_closure": False,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T20:00:00Z",
        },
        "actual_rate": 0.444
    },
    {
        "name": "Congestion (Low-Medium Actual Rate)",
        "event": {
            "event_cause": "congestion",
            "corridor": "Electronic City",
            "priority": "Low",  # Only High/Low in training data
            "requires_road_closure": False,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T08:00:00Z",
        },
        "actual_rate": 0.050  # estimated
    }
]

print("Calibrated Model Predictions vs Actual Rates")
print("=" * 60)
for test in test_events:
    try:
        prediction = predict_severity(test["event"])
        recommendation = get_recommendation(prediction["tier"], test["event"]["corridor"])
        
        print(f"\n{test['name']}:")
        print(f"  Actual high-rate: {test['actual_rate']:.3f}")
        print(f"  Predicted P(high): {prediction['probability_high']:.4f}")
        print(f"  Tier: {prediction['tier']}")
        print(f"  Action: {recommendation['action']}")
        print(f"  Message: {recommendation['message']}")
        
        # Calculate error
        error = abs(prediction['probability_high'] - test['actual_rate'])
        print(f"  Prediction Error: {error:.3f}")
        
        # Show if tier is reasonable
        actual_tier = "HIGH" if test['actual_rate'] >= 0.817 else ("MEDIUM" if test['actual_rate'] >= 0.3046 else "ROUTINE")
        tier_correct = prediction['tier'] == actual_tier
        status = "PASS" if tier_correct else "FAIL"
        print(f"  Expected Tier (based on actual rate): {actual_tier} [{status}]")
    except Exception as e:
        print(f"\n{test['name']}: ERROR - {e}")
