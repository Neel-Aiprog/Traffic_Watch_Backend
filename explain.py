"""
explain.py — per-prediction SHAP attribution for Traffic Watch.

Computes which specific feature VALUES pushed THIS incident's probability
toward HIGH or toward ROUTINE, averaged across the three ensemble models.
Kept separate from inference.py so the core prediction path is unaffected
if explanation ever needs to be disabled for latency reasons.

Depends on: shap  (pip install shap)
"""

import shap
import numpy as np

# Import the already-loaded model objects and feature order directly from
# inference.py so we never load models twice and always use the exact same
# objects the prediction itself used.
from inference import (
    _lgb_model,
    _xgb_model,
    _cb_model,          # your actual variable name (not _catboost_model)
    TRAINED_FEATURE_ORDER,
)

# Human-readable labels for the feature names that appear in TRAINED_FEATURE_ORDER.
# Anything not in this dict is an internal/engineered column that we skip.
FEATURE_LABELS = {
    "event_cause":             "Incident cause",
    "corridor_grouped":        "Corridor",
    "priority":                "Priority level",
    "requires_road_closure":   "Road closure required",
    "veh_type":                "Vehicle type",
    "time_of_day":             "Time of day",
    "day_of_week":             "Day of week",
    "is_weekend":              "Weekend",
}

# Build explainers once at import time — TreeExplainer setup has a
# one-time cost that we don't want to pay per-request.
_lgb_explainer = shap.TreeExplainer(_lgb_model)
_xgb_explainer = shap.TreeExplainer(_xgb_model)

# CatBoost sometimes needs model_output="raw" depending on shap version.
# We try the default first and fall back gracefully if it errors.
try:
    _cb_explainer = shap.TreeExplainer(_cb_model)
    _CB_EXPLAINER_AVAILABLE = True
except Exception as e:
    print(f"[explain] CatBoost explainer failed to init: {e}. "
          f"Explanation will average LightGBM + XGBoost only.")
    _CB_EXPLAINER_AVAILABLE = False


def _format_value(feature_name, raw_value):
    """Turn a raw feature value into a readable string for display."""
    if feature_name == "requires_road_closure":
        return "Yes" if raw_value else "No"
    if feature_name == "is_weekend":
        return "Yes" if raw_value else "No"
    if isinstance(raw_value, str):
        return raw_value.replace("_", " ").capitalize()
    return str(raw_value)


def _get_display_value(feature_name, event_dict, feature_row):
    """
    Pull the display value for a feature from the original event dict
    where possible (more readable), falling back to the encoded feature row.
    corridor_grouped maps back to the original corridor field.
    """
    if feature_name == "corridor_grouped":
        return _format_value(feature_name, event_dict.get("corridor", "Unknown"))
    if feature_name in event_dict:
        return _format_value(feature_name, event_dict[feature_name])
    # Fall back to the encoded value
    try:
        val = feature_row[feature_name].iloc[0]
        return _format_value(feature_name, val)
    except Exception:
        return "Unknown"


def explain_prediction(event_dict: dict, feature_row) -> list:
    """
    Compute the top contributing factors for a single prediction.

    Args:
        event_dict:   The raw event dict from the API request (for readable values).
        feature_row:  The encoded, column-ordered DataFrame row that was passed
                      to the models — must already have TRAINED_FEATURE_ORDER
                      columns in the right order (i.e. the output of
                      _build_single_row_features from inference.py).

    Returns:
        List of up to 4 dicts, sorted by impact descending:
        [
          {
            "feature":           "Incident cause",
            "value":             "Construction",
            "direction":         "HIGH",        # pushed toward HIGH or ROUTINE
            "relative_strength": 85,            # 0–100, where 100 = strongest factor
          },
          ...
        ]
    """
    # Collect SHAP values from each available model.
    shap_arrays = []

    try:
        lgb_shap = _lgb_explainer.shap_values(feature_row)
        # LightGBM binary returns shape (n_samples, n_features) directly
        shap_arrays.append(np.array(lgb_shap).reshape(-1))
    except Exception as e:
        print(f"[explain] LightGBM SHAP failed: {e}")

    try:
        xgb_shap = _xgb_explainer.shap_values(feature_row)
        shap_arrays.append(np.array(xgb_shap).reshape(-1))
    except Exception as e:
        print(f"[explain] XGBoost SHAP failed: {e}")

    if _CB_EXPLAINER_AVAILABLE:
        try:
            cb_shap = _cb_explainer.shap_values(feature_row)
            shap_arrays.append(np.array(cb_shap).reshape(-1))
        except Exception as e:
            print(f"[explain] CatBoost SHAP failed: {e}")

    if not shap_arrays:
        return []  # All explainers failed — return empty rather than crash

    # Average across whichever models succeeded. This way the explanation
    # reflects the ensemble's actual behavior even if one explainer errored.
    avg_shap = np.mean(shap_arrays, axis=0)

    # Build the contributions list, skipping features with no friendly label.
    contributions = []
    for feat_name, shap_val in zip(TRAINED_FEATURE_ORDER, avg_shap):
        if feat_name not in FEATURE_LABELS:
            continue
        contributions.append({
            "feature":   FEATURE_LABELS[feat_name],
            "value":     _get_display_value(feat_name, event_dict, feature_row),
            "shap_value": float(shap_val),
            "direction": "HIGH" if shap_val > 0 else "ROUTINE",
        })

    # Sort by absolute impact, take top 4.
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    top4 = contributions[:4]

    # Normalise to 0–100 relative strength so the frontend can render
    # proportional bars without exposing raw SHAP units to the UI.
    if top4:
        max_abs = max(abs(c["shap_value"]) for c in top4)
        for c in top4:
            c["relative_strength"] = (
                round(abs(c["shap_value"]) / max_abs * 100) if max_abs > 0 else 0
            )
            del c["shap_value"]  # don't expose raw SHAP values in the API response

    return top4