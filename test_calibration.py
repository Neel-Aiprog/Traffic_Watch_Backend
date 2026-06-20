import numpy as np
from inference import _ensemble_proba_high, _build_single_row_features

# Diagnosis data: (predicted, actual) pairs from comments
diagnosis_data = [
    (0.2481, 0.016),   # vehicle_breakdown
    (0.1136, 0.164),   # accident
    (0.8892, 0.444),   # public_event
    (0.9255, 0.528),   # others
    (0.9337, 0.662),   # construction
    (0.9550, 0.812),   # road_conditions
    (0.9615, 0.860),   # pot_holes
]

def apply_shrinkage(proba, lambda_val, base_rate=0.5):
    return proba * (1 - lambda_val) + lambda_val * base_rate

def apply_logit_shift(proba, lambda_val):
    # lambda_val < 1 shrinks toward 0.5, lambda_val > 1 pushes away from 0.5
    # Avoid log(0) or log(1) 
    eps = 1e-7
    proba = np.clip(proba, eps, 1.0 - eps)
    logit = np.log(proba / (1 - proba))
    shifted_logit = logit * lambda_val  
    return 1.0 / (1.0 + np.exp(-shifted_logit))

def piecewise_linear_calibration(proba, points):
    # points: list of (predicted, actual) tuples, sorted by predicted
    if not points:
        return proba
        
    # Handle edge cases
    if proba <= points[0][0]:
        # Below first point - extrapolate or use first point
        if len(points) >= 2:
            # Linear extrapolation using first two points
            x1, y1 = points[0]
            x2, y2 = points[1]
            return y1 + (proba - x1) * (y2 - y1) / (x2 - x1)
        else:
            return points[0][1]
            
    if proba >= points[-1][0]:
        # Above last point - extrapolate or use last point
        if len(points) >= 2:
            # Linear extrapolation using last two points
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            return y1 + (proba - x1) * (y2 - y1) / (x2 - x1)
        else:
            return points[-1][1]
    
    # Find interval containing proba
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= proba <= x2:
            # Linear interpolation
            return y1 + (proba - x1) * (y2 - y1) / (x2 - x1)
    
    # Should not reach here
    return points[-1][1]

def evaluate_calibration(calibrate_func, param_name, param_values, points=None):
    best_param = None
    best_mae = float('inf')
    
    print(f"Evaluating {param_name}:")
    for param_val in param_values:
        errors = []
        for pred, actual in diagnosis_data:
            if points is not None:
                calibrated = calibrate_func(pred, points)
            elif param_name == "lambda":
                calibrated = apply_shrinkage(pred, param_val)
            else:  # lambda_val for logit
                calibrated = apply_logit_shift(pred, param_val)
            error = abs(calibrated - actual)
            errors.append(error)
        mae = np.mean(errors)
        print(f"  {param_name}={param_val:.3f}: MAE={mae:.4f}")
        if mae < best_mae:
            best_mae = mae
            best_param = param_val
    
    print(f"  Best {param_name}: {best_param:.3f} (MAE={best_mae:.4f})")
    return best_param, best_mae

# Test shrinkage approach
print("=== SHRINKAGE APPROACH ===")
lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_lambda, best_mae = evaluate_calibration(apply_shrinkage, "lambda", lambda_values)

print()
print("Results with best lambda (0.3):")
print("Pred -> Calibrated -> Actual")
for pred, actual in diagnosis_data:
    calibrated = apply_shrinkage(pred, 0.3)
    print(f"{pred:.4f} -> {calibrated:.4f} -> {actual:.3f} (error: {abs(calibrated-actual):.3f})")

print()
print("=== LOGIT SHIFT APPROACH ===")
lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
best_lambda_logit, best_mae_logit = evaluate_calibration(apply_logit_shift, "lambda", lambda_values)

print()
print("Results with best logit lambda:")
print("Pred -> Calibrated -> Actual")
for pred, actual in diagnosis_data:
    calibrated = apply_logit_shift(pred, best_lambda_logit)
    print(f"{pred:.4f} -> {calibrated:.4f} -> {actual:.3f} (error: {abs(calibrated-actual):.3f})")

print()
print("=== PIECEWISE LINEAR APPROACH ===")
# Sort diagnosis points by predicted value
sorted_points = sorted(diagnosis_data, key=lambda x: x[0])
print("Calibration points (predicted -> actual):")
for p, a in sorted_points:
    print(f"  {p:.4f} -> {a:.3f}")

# Test piecewise linear
errors = []
for pred, actual in diagnosis_data:
    calibrated = piecewise_linear_calibration(pred, sorted_points)
    error = abs(calibrated - actual)
    errors.append(error)
    print(f"{pred:.4f} -> {calibrated:.4f} -> {actual:.3f} (error: {error:.3f})")
mae = np.mean(errors)
print(f"Piecewise linear MAE: {mae:.4f}")
