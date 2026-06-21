import numpy as np

# diagnosis data from test_calibration.py
diagnosis_data = [
    (0.2481, 0.016),   # vehicle_breakdown
    (0.1136, 0.164),   # accident
    (0.8892, 0.444),   # public_event
    (0.9255, 0.528),   # others
    (0.9337, 0.662),   # construction
    (0.9550, 0.812),   # road_conditions
    (0.9615, 0.860),   # pot_holes
]

# shrinkage parameters
lam = 0.3
def calibrate(raw):
    return raw * (1 - lam) + lam * 0.5

# thresholds on calibrated
high_thresh = 0.817
med_thresh = 0.3046

print("Raw -> Calibrated -> Tier (calibrated thresholds)")
for raw, actual in diagnosis_data:
    cal = calibrate(raw)
    if cal >= high_thresh:
        tier = "HIGH"
    elif cal >= med_thresh:
        tier = "MEDIUM"
    else:
        tier = "ROUTINE"
    print(f"{raw:.4f} -> {cal:.4f} -> {tier} (actual {actual:.3f})")

# compute equivalent raw thresholds
high_raw_thresh = (high_thresh - lam*0.5) / (1 - lam)
med_raw_thresh = (med_thresh - lam*0.5) / (1 - lam)
print(f"\nEquivalent raw thresholds: high={high_raw_thresh:.4f}, medium={med_raw_thresh:.4f}")

# Now test using raw thresholds directly
print("\nUsing raw thresholds directly:")
for raw, actual in diagnosis_data:
    if raw >= high_raw_thresh:
        tier = "HIGH"
    elif raw >= med_raw_thresh:
        tier = "MEDIUM"
    else:
        tier = "ROUTINE"
    print(f"{raw:.4f} -> {tier} (actual {actual:.3f})")
