"""
inference.py — loads the trained ensemble ONCE and exposes a single
predict_severity() function for the API layer to call.

Models are loaded at module import time (i.e. once, when the FastAPI app
starts), not per-request. This matters: loading a CatBoost/LightGBM/XGBoost
model from disk takes tens of milliseconds, which is fine once at startup
but would add real latency if done on every request.

v3.0 changes (adapted for native models):
  - known_categories.json loaded at startup to remap unseen category values
    → NaN before XGBoost sees them (XGBoost hard-crashes on unknown
    categories; LightGBM/CatBoost handle them gracefully via missing-value
    branch).
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from feature_engineering import build_features, CATEGORICAL_FEATURES, ALL_FEATURE_COLS

MODEL_DIR = "./"  # Model files are in the same directory

# ── Load models once at import time (native formats) ─────────────────────
_lgb_model = lgb.Booster(model_file=f"{MODEL_DIR}/model_lgb_binary.txt")

_xgb_model = xgb.XGBClassifier()
_xgb_model.load_model(f"{MODEL_DIR}/model_xgb_binary.json")

_cb_model = CatBoostClassifier()
_cb_model.load_model(f"{MODEL_DIR}/model_catboost_binary.cbm")

# XGBoost is strict about column ORDER matching training exactly (unlike
# LightGBM/CatBoost, which matched by name and didn't error). Rather than
# maintaining a second hardcoded column-order list that can silently drift
# out of sync with feature_engineering_v2.ALL_FEATURE_COLS (which is what
# caused this bug), read the true training-time order directly off the
# saved model and use THAT as the single source of truth at inference time.
TRAINED_FEATURE_ORDER = list(_xgb_model.get_booster().feature_names)

with open(f"{MODEL_DIR}/label_encoder_classes_binary.json") as f:
    _label_classes = json.load(f)  # e.g. ["routine", "high"] -- index 1 should be "high"

with open(f"{MODEL_DIR}/thresholds.json") as f:
    _thresholds = json.load(f)

HIGH_THRESHOLD = _thresholds["high_threshold"]
MEDIUM_THRESHOLD = _thresholds["medium_threshold"]

assert _label_classes[1] == "high", (
    f"Expected label_classes[1] == 'high', got {_label_classes}. "
    f"Check label_encoder_classes_binary.json -- predict_proba()[:, 1] "
    f"below assumes index 1 is the 'high' class."
)

# ── Load known training categories once at startup ───────────────────────
# Generated at training time and saved as known_categories.json.
# XGBoost hard-crashes on any category value it didn't see during training.
# LightGBM and CatBoost handle this gracefully (NaN → missing-value branch),
# so we normalise before prediction: unseen values → NaN.
_KNOWN_CATEGORIES: dict = {}
_known_cat_path = os.path.join(MODEL_DIR, "known_categories.json")
if os.path.exists(_known_cat_path):
    with open(_known_cat_path) as f:
        _KNOWN_CATEGORIES = json.load(f)
    print(f"[inference] Loaded known_categories.json ({len(_KNOWN_CATEGORIES)} columns).")
else:
    # ── Fallback hardcoded vocabulary ────────────────────────────────────────
    # Used until known_categories.json is deployed alongside the model files.
    # Extend this with every value that appeared in your training data.
    _KNOWN_CATEGORIES = {
        "veh_type": [
            "car", "bus", "truck", "auto", "emergency",
            "heavy_vehicle", "light_vehicle", "other",
        ],
        "event_cause": [
            "accident", "vehicle_breakdown", "public_event",
            "construction", "road_conditions", "pot_holes", "others",
        ],
    }
    print(
        "[inference] WARNING: known_categories.json not found. "
        "Using hardcoded fallback vocabulary. "
        "Re-run training and deploy known_categories.json to fix this properly."
    )


# ── Category safety ──────────────────────────────────────────────────────
def _remap_unseen_categories(features: pd.DataFrame) -> pd.DataFrame:
    """
    Remap any category value not seen during training → NaN.

    XGBoost hard-crashes on unseen category codes; LightGBM and CatBoost
    both route NaN to their missing-value/default branch, so NaN is the
    correct neutral value for all three models.

    Must be called AFTER astype("category") (so the dtype is set) and
    BEFORE column reordering (order doesn't affect the remap).
    """
    for col, known_vals in _KNOWN_CATEGORIES.items():
        if col not in features.columns:
            continue
        known_set = set(known_vals)
        mask = features[col].notna() & ~features[col].isin(known_set)
        if mask.any():
            unseen = features.loc[mask, col].unique().tolist()
            print(
                f"[inference] Unseen category in '{col}': {unseen!r}. "
                f"Remapping to NaN (models will use missing-value branch)."
            )
            # Cast to object first — pandas rejects NaN assignment into a
            # Categorical series that doesn't list NaN as a valid level.
            features[col] = features[col].astype(object).where(~mask, other=np.nan)
            # If the entire column is NaN (unlikely but possible), fill with a known category
            # to avoid XGBoost error on empty categories.
            if features[col].isna().all():
                # Prefer 'unknown' if it exists in known_vals, otherwise use the first known category.
                if 'unknown' in known_vals:
                    default_val = 'unknown'
                else:
                    default_val = known_vals[0] if known_vals else None
                if default_val is not None:
                    features[col] = features[col].fillna(default_val)
                else:
                    # Fallback if no known values (should not happen)
                    features[col] = features[col].fillna("unknown")
            features[col] = features[col].astype("category")
    return features


# ── Feature construction ─────────────────────────────────────────────────
def _build_single_row_features(event: dict) -> pd.DataFrame:
    """
    Build a 1-row feature DataFrame from a raw event dict using the SAME
    build_features() pipeline used during training, so feature handling
    (corridor grouping, time bucketing, missing-value fills) stays
    identical between training and inference.

    `event` is expected to have raw fields matching the original dataset:
    event_cause, corridor, priority, requires_road_closure, veh_type,
    start_datetime (ISO string) — NOT pre-engineered features. This
    avoids duplicating feature logic in two places that could drift apart.
    """
    df = pd.DataFrame([event])

    # build_features() expects 'severity' to exist (added by
    # add_severity_target_binary at training time). Inject a placeholder
    # so the shared pipeline runs unmodified, then drop it.
    df["severity"] = "routine"  # placeholder — value is irrelevant
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)

    features = build_features(df)
    features = features.drop(columns=["severity"])

    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")

    # Remap unseen categories → NaN before XGBoost sees them.
    features = _remap_unseen_categories(features)

    # Reorder columns to EXACTLY match training order (XGBoost requirement).
    return features[TRAINED_FEATURE_ORDER]


# ── Ensemble prediction ──────────────────────────────────────────────────
def _ensemble_proba_high(X: pd.DataFrame) -> tuple[float, dict]:
    """
    Average P(high) across the 3 calibrated models.
    Returns (ensemble_probability, individual_scores).

    All three models are wrappers — predict_proba() returns probabilities.
    For LightGBM Booster, predict returns P(class=1) directly for binary.
    For XGBoost and CatBoost, predict_proba()[:, 1] gives P(class=1).
    """
    lgb_proba = _lgb_model.predict(X)  # lgb.Booster.predict returns P(class=1) directly for binary
    xgb_proba = _xgb_model.predict_proba(X)[:, 1]
    cb_proba  = _cb_model.predict_proba(X)[:, 1]

    individual = {
        "lightgbm": float(lgb_proba[0]),
        "xgboost":  float(xgb_proba[0]),
        "catboost": float(cb_proba[0]),
    }
    ensemble = float(np.mean([lgb_proba[0], xgb_proba[0], cb_proba[0]]))
    return ensemble, individual


# ── Tier assignment ──────────────────────────────────────────────────────
def apply_tier(p_high: float) -> str:
    if p_high >= HIGH_THRESHOLD:
        return "HIGH"
    elif p_high >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "ROUTINE"


# ── Public API ───────────────────────────────────────────────────────────
def predict_severity(event: dict) -> dict:
    """
    Main entry point. Takes a raw event dict, returns the prediction block
    (tier, probability, individual model scores) ready to be merged into
    the API response.
    """
    X = _build_single_row_features(event)
    p_high, individual_scores = _ensemble_proba_high(X)
    tier = apply_tier(p_high)

    return {
        "tier": tier,
        "probability_high": round(p_high, 4),
        "thresholds_used": {
            "high_threshold":   HIGH_THRESHOLD,
            "medium_threshold": MEDIUM_THRESHOLD,
        },
        "individual_scores": {k: round(v, 4) for k, v in individual_scores.items()},
        "_feature_row": X,  # passed to explainer, stripped before API response
    }