from inference import predict_severity
from recommendation import get_recommendation

# Test scenarios
scenarios = [
    {
        "name": "1. Minor vehicle breakdown on small road",
        "event": {
            "event_cause": "vehicle_breakdown",
            "corridor": "Small Local Road",
            "priority": "Low",
            "requires_road_closure": False,
            "veh_type": "two_wheeler",
            "start_datetime": "2026-06-18T14:30:00Z",
        },
        "expected": "ROUTINE",
        "notes": "Minor issue, low priority, no closure expected"
    },
    {
        "name": "2. Construction during peak hours",
        "event": {
            "event_cause": "construction",
            "corridor": "Main Highway",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T08:30:00Z",
        },
        "expected": "MEDIUM",
        "notes": "Construction with road closure during peak"
    },
    {
        "name": "3. Large public event with road closure",
        "event": {
            "event_cause": "public_event",
            "corridor": "City Center Boulevard",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T19:00:00Z",
        },
        "expected": "MEDIUM",
        "notes": "Large public event with closure"
    },
    {
        "name": "4. Severe water logging/flooding",
        "event": {
            "event_cause": "water_logging",
            "corridor": "Main Arterial Road",
            "priority": "High",
            "requires_road_closure": True,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T16:00:00Z",
        },
        "expected": "MEDIUM",
        "notes": "Severe water logging with closure during peak"
    },
    {
        "name": "5. Pothole on major road",
        "event": {
            "event_cause": "pot_holes",
            "corridor": "National Highway",
            "priority": "High",
            "requires_road_closure": False,
            "veh_type": "unknown",
            "start_datetime": "2026-06-18T12:00:00Z",
        },
        "expected": "MEDIUM",
        "notes": "Pothole repair on major road"
    }
]

print("ADDITIONAL TEST SCENARIOS FOR CALIBRATED MODEL")
print("=" * 50)

for scenario in scenarios:
    try:
        prediction = predict_severity(scenario["event"])
        recommendation = get_recommendation(prediction["tier"], scenario["event"]["corridor"])
        
        print(f"\n{scenario['name']}")
        print("-" * len(scenario['name']))
        print(f"Event: {scenario['event']['event_cause']} on {scenario['event']['corridor']}")
        print(f"Predicted P(high): {prediction['probability_high']:.4f}")
        print(f"Tier: {prediction['tier']} (Expected: {scenario['expected']}) {'PASS' if prediction['tier'] == scenario['expected'] else 'FAIL'}")
        print(f"Action: {recommendation['action']}")
        print(f"Message: {recommendation['message']}")
        if scenario['notes']:
            print(f"Notes: {scenario['notes']}")
        print(f"Model Scores - LGBM: {prediction['individual_scores']['lightgbm']:.3f}, "
              f"XGB: {prediction['individual_scores']['xgboost']:.3f}, "
              f"CAT: {prediction['individual_scores']['catboost']:.3f}")
    except Exception as e:
        print(f"\n{scenario['name']}")
        print("-" * len(scenario['name']))
        print(f"ERROR: {e}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
