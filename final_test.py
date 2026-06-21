import inference
import json

# Test 1: unseen category two_wheeler
event1 = {
    "event_cause": "accident",
    "corridor": "Bannerghata Road",
    "priority": "High",
    "requires_road_closure": 0,
    "veh_type": "two_wheeler",
    "start_datetime": "2026-06-21T10:00:00Z",
}
print("Test 1: unseen veh_type")
result1 = inference.predict_severity(event1)
print(json.dumps(result1, indent=2))
print()

# Test 2: known category private_car
event2 = {
    "event_cause": "accident",
    "corridor": "Bannerghata Road",
    "priority": "High",
    "requires_road_closure": 0,
    "veh_type": "private_car",
    "start_datetime": "2026-06-21T10:00:00Z",
}
print("Test 2: known veh_type private_car")
result2 = inference.predict_severity(event2)
print(json.dumps(result2, indent=2))
print()

# Test 3: high actual cause construction (should be MEDIUM or HIGH?)
event3 = {
    "event_cause": "construction",
    "corridor": "Mysore Road",
    "priority": "Low",
    "requires_road_closure": 0,
    "veh_type": "unknown",
    "start_datetime": "2026-06-18T14:00:00Z",
}
print("Test 3: construction")
result3 = inference.predict_severity(event3)
print(json.dumps(result3, indent=2))
print()

# Test 4: low actual cause vehicle_breakdown (should be ROUTINE)
event4 = {
    "event_cause": "vehicle_breakdown",
    "corridor": "Mysore Road",
    "priority": "Low",
    "requires_road_closure": 0,
    "veh_type": "unknown",
    "start_datetime": "2026-06-18T14:00:00Z",
}
print("Test 4: vehicle_breakdown")
result4 = inference.predict_severity(event4)
print(json.dumps(result4, indent=2))
print()

# Show thresholds used
with open("./thresholds.json") as f:
    th = json.load(f)
print("Thresholds in use:")
print(json.dumps(th, indent=2))
