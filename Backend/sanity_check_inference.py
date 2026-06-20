from inference import predict_severity
from recommendation import get_recommendation

# construction was shown in earlier EDA to have a much higher proportion
# of long-duration (high severity) events than vehicle_breakdown -- this
# event should score noticeably higher than the 0.033 we saw before.
sample_event = {
    "event_cause": "construction",
    "corridor": "Mysore Road",
    "priority": "High",
    "requires_road_closure": True,
    "veh_type": "unknown",
    "start_datetime": "2026-06-17T19:32:00Z",
}

print("Running prediction on construction event...")
prediction = predict_severity(sample_event)
print(prediction)
print()
recommendation = get_recommendation(prediction["tier"], sample_event["corridor"])
print(recommendation)