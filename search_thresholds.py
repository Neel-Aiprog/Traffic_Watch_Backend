import numpy as np

# Data: cause, raw_score, actual_rate
data = [
    ("vehicle_breakdown", 0.1985, 0.016),
    ("accident", 0.1358, 0.164),
    ("congestion", 0.0615, 0.050),
    ("public_event", 0.8227, 0.444),
    ("others", 0.9043, 0.528),
    ("construction", 0.9135, 0.662),
    ("road_conditions", 0.9462, 0.812),
    ("water_logging", 0.9367, 0.860),
    ("tree_fall", 0.9344, 0.800),
    ("pot_holes", 0.9654, 0.860),
]

# Desired tier based on actual rate and thresholds 0.817/0.3046
def desired_tier(actual):
    if actual >= 0.817:
        return "HIGH"
    elif actual >= 0.3046:
        return "MEDIUM"
    else:
        return "ROUTINE"

best_acc = -1
best_thresh = None
# Search over possible thresholds
for high in np.arange(0.5, 0.99, 0.001):
    for med in np.arange(0.1, high-0.001, 0.001):
        correct = 0
        for _, raw, actual in data:
            if raw >= high:
                pred = "HIGH"
            elif raw >= med:
                pred = "MEDIUM"
            else:
                pred = "ROUTINE"
            if pred == desired_tier(actual):
                correct += 1
        acc = correct / len(data)
        if acc > best_acc:
            best_acc = acc
            best_thresh = (high, med)
            if acc == 1.0:
                break
    if best_acc == 1.0:
        break

print(f"Best accuracy: {best_acc:.2%} with thresholds high={best_thresh[0]:.3f}, medium={best_thresh[1]:.3f}")
print("\nPer-cause results:")
for cause, raw, actual in data:
    if raw >= best_thresh[0]:
        pred = "HIGH"
    elif raw >= best_thresh[1]:
        pred = "MEDIUM"
    else:
        pred = "ROUTINE"
    desired = desired_tier(actual)
    ok = (pred == desired)
    print(f"{cause:<15} raw={raw:.4f} actual={actual:.3f} desired={desired:<6} pred={pred:<6} {'OK' if ok else 'FAIL'}")

# Also show what thresholds would be if we want to minimize MAE between raw and actual after applying thresholds?
# Not needed.
