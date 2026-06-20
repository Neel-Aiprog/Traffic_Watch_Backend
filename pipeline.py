"""
Training pipeline for PS2 — binary severity classification (routine vs high).

v3.0 — calibration-aware retraining
  Problem diagnosed: model ranked correctly (AUC ~0.923) but probability
  values were systematically miscalibrated — flat recall from t=0.21 to
  t=0.70 proved the ensemble was overconfident and not spreading probability
  mass across the range. Root causes:
    1. LGB optimising AUC directly — AUC is rank-only, ignores probability
       values. Switched to binary_logloss which directly penalises
       miscalibrated probabilities.
    2. XGB scale_pos_weight up to 6.0 — large values shift ALL predicted
       probabilities upward. Clamped to 1.0–2.5.
    3. Post-training: each model wrapped with CalibratedClassifierCV
       (isotonic regression, cv='prefit') fitted on the val set.
       This is the textbook fix for a model that ranks well but whose
       probability values are off.

  Models are now saved with joblib (not native formats) since the
  calibration wrapper is a sklearn object.

  cause_high_rate target encoding added (computed post-split on train only).
"""

import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from joblib import dump as joblib_dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_TRIALS     = 40

CATEGORICAL_FEATURES = [
    'event_cause', 'corridor_grouped', 'priority',
    'veh_type', 'day_of_week', 'time_of_day',
    'zone', 'event_type',
]
NUMERIC_FEATURES  = ['hour', 'cause_high_rate']
BOOLEAN_FEATURES  = ['requires_road_closure', 'is_weekend', 'has_junction']

DATA_PATH = 'model_ready_features.csv'

DEFAULT_HIGH_THRESHOLD   = 0.60
DEFAULT_MEDIUM_THRESHOLD = 0.35


# ══════════════════════════════════════════════════════════════════════════════
# Target encoding (post-split — no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def add_cause_high_rate(X_train, y_train, X_val, X_test):
    train_df = X_train.copy()
    train_df['_y'] = y_train

    cause_rate  = (
        train_df.groupby('event_cause')['_y']
        .mean()
        .rename('cause_high_rate')
    )
    global_mean = float(cause_rate.mean())

    print('\n=== cause_high_rate (target encoding on train only) ===')
    print(cause_rate.sort_values(ascending=False).to_string())
    print(f'Global fallback mean: {global_mean:.4f}')

    def _apply(X):
        X = X.copy()
        X['cause_high_rate'] = (
            X['event_cause'].astype(str)
            .map(cause_rate)
            .fillna(global_mean)
        )
        return X

    return _apply(X_train), _apply(X_val), _apply(X_test), global_mean, cause_rate


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATA_PATH)

    # cause_high_rate added post-split — exclude here
    base_cat  = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    base_num  = [c for c in NUMERIC_FEATURES if c in df.columns and c != 'cause_high_rate']
    base_bool = [c for c in BOOLEAN_FEATURES if c in df.columns]

    for col in base_cat:
        df[col] = df[col].astype('category')
    for col in base_bool:
        df[col] = df[col].astype(bool)
    for col in base_num:
        df[col] = df[col].astype(float)

    X = df[base_cat + base_num + base_bool]
    y = df['severity']

    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y)

    if label_encoder.classes_[0] == 'high':
        y_encoded               = 1 - y_encoded
        label_encoder.classes_  = label_encoder.classes_[::-1]

    assert label_encoder.classes_[1] == 'high', \
        f"Label fix failed — classes are {label_encoder.classes_}"

    return X, y_encoded, label_encoder


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_cat_feature_indices(X: pd.DataFrame):
    return [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X.columns]


# ══════════════════════════════════════════════════════════════════════════════
# Calibration diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, proba, n_bins=10):
    """ECE — lower is better. 0.05 is good, >0.10 is a problem."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = proba[mask].mean()
        ece     += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def plot_reliability_diagram(y_true, proba_before, proba_after,
                              save_path='reliability_diagram.png'):
    """Compare calibration before and after isotonic regression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, proba, title in [
        (axes[0], proba_before, 'Before calibration'),
        (axes[1], proba_after,  'After calibration'),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ece = expected_calibration_error(y_true, proba)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 's-', label=f'Model (ECE={ece:.4f})')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Reliability diagram saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objectives — v3.0 calibration-aware changes
# ══════════════════════════════════════════════════════════════════════════════

def lgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        # ── v3.0: binary_logloss instead of auc ──────────────────────────
        # Log loss directly penalises miscalibrated probabilities.
        # AUC is rank-only and produces well-separated but poorly-scaled probs.
        'objective':    'binary',
        'metric':       'binary_logloss',          # CHANGED from 'auc'
        'verbosity':    -1,
        'boosting_type':'gbdt',
        'random_state': RANDOM_STATE,
        'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': 'balanced',
    }
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    proba = model.predict_proba(X_val)[:, 1]
    # Optimise AUC in Optuna (ranking still matters) but training uses logloss
    return roc_auc_score(y_val, proba)


def xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective':        'binary:logistic',
        'eval_metric':      'logloss',             # CHANGED from 'auc'
        'random_state':     RANDOM_STATE,
        'enable_categorical': True,
        'tree_method':      'hist',
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma':            trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        # ── v3.0: clamped to 1.0–2.5 (was 1.0–6.0) ──────────────────────
        # Large scale_pos_weight (e.g. 3.9) shifts ALL predicted probabilities
        # upward uniformly, which is the direct cause of the flat recall curve.
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 2.5),  # CHANGED
    }
    model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def catboost_objective(trial, X_train, y_train, X_val, y_val):
    # CatBoost Logloss was already calibration-aware — no changes needed
    params = {
        'loss_function':      'Logloss',
        'eval_metric':        'AUC',
        'random_state':       RANDOM_STATE,
        'verbose':            False,
        'iterations':         trial.suggest_int('iterations', 100, 800),
        'depth':              trial.suggest_int('depth', 3, 10),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count':       trial.suggest_int('border_count', 32, 255),
        'bagging_temperature':trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'auto_class_weights': 'Balanced',
    }
    cat_idx = get_cat_feature_indices(X_train)
    model   = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_idx,
        eval_set=(X_val, y_val),
        early_stopping_rounds=30,
        use_best_model=True,
    )
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def tune_model(objective_fn, X_train, y_train, X_val, y_val, name):
    study = optuna.create_study(direction='maximize', study_name=name)
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    print(f'\n[{name}] Best ROC-AUC (val): {study.best_value:.4f}')
    print(f'[{name}] Best params: {study.best_params}')
    return study.best_params, study.best_value


# ══════════════════════════════════════════════════════════════════════════════
# Final model trainers
# ══════════════════════════════════════════════════════════════════════════════

def train_final_lgb(params, X_train, y_train, X_val, y_val):
    params = dict(params)
    params.update({
        'objective': 'binary', 'metric': 'binary_logloss',  # v3.0
        'verbosity': -1, 'random_state': RANDOM_STATE, 'class_weight': 'balanced',
    })
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model    = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def train_final_xgb(params, X_train, y_train, X_val, y_val):
    params = dict(params)
    params.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',  # v3.0
        'random_state': RANDOM_STATE, 'enable_categorical': True, 'tree_method': 'hist',
    })
    model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_final_catboost(params, X_train, y_train, X_val, y_val):
    params = dict(params)
    params.update({
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'random_state': RANDOM_STATE, 'verbose': False, 'auto_class_weights': 'Balanced',
    })
    cat_idx = get_cat_feature_indices(X_train)
    model   = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train, cat_features=cat_idx,
        eval_set=(X_val, y_val), early_stopping_rounds=30, use_best_model=True,
    )
    return model


def calibrate_models(models, X_val, y_val):
    """
    Wrap each trained model with isotonic regression calibration.
    cv='prefit' means: model is already trained, just fit the calibrator
    on (X_val, y_val). Uses the same val set used for early stopping.
    """
    calibrated = []
    for model in models:
        cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        cal.fit(X_val, y_val)
        calibrated.append(cal)
    return calibrated


def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Dual-threshold selection
# ══════════════════════════════════════════════════════════════════════════════

def find_dual_thresholds(y_true, proba,
                          recall_floor_high=0.88,
                          recall_floor_medium=0.90,
                          min_gap=0.20):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]

    print('\n=== PR Curve Operating Points (P(high) between 0.20 and 0.80) ===')
    print(f'{"Threshold":>10}  {"Precision":>10}  {"Recall":>8}')
    for t, p, r in zip(thresholds, precision, recall):
        if 0.20 <= t <= 0.80:
            print(f'{t:10.3f}  {p:10.3f}  {r:8.3f}')

    high_candidates = [
        (t, p, r) for t, p, r in zip(thresholds, precision, recall)
        if r >= recall_floor_high
    ]
    if high_candidates:
        high_t, best_p, best_r = max(high_candidates, key=lambda x: x[0])
        print(f'\nSelected HIGH threshold : {high_t:.3f}  '
              f'(precision={best_p:.3f}, recall={best_r:.3f})')
    else:
        high_t = DEFAULT_HIGH_THRESHOLD
        print(f'WARN: using default high_t={high_t}')

    medium_candidates = [
        (t, p, r) for t, p, r in zip(thresholds, precision, recall)
        if r >= recall_floor_medium
    ]
    if medium_candidates:
        medium_t, best_p, best_r = max(medium_candidates, key=lambda x: x[0])
        if high_t - medium_t < min_gap:
            medium_t = round(high_t - min_gap, 3)
            print(f'NOTE: thresholds too close — medium_t clamped to {medium_t}')
        print(f'Selected MEDIUM threshold: {medium_t:.3f}')
    else:
        medium_t = DEFAULT_MEDIUM_THRESHOLD
        print(f'WARN: using default medium_t={medium_t}')

    return round(float(high_t), 4), round(float(medium_t), 4)


def plot_pr_curve(y_true, proba, high_t, medium_t, save_path='pr_curve.png'):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f'PR curve (AP={ap:.3f})')
    for t_val, label, color in [
        (high_t,   f'HIGH t={high_t}',     'red'),
        (medium_t, f'MEDIUM t={medium_t}', 'orange'),
    ]:
        idx = min(np.searchsorted(thresholds, t_val), len(precision) - 2)
        ax.scatter(recall[idx], precision[idx], s=100, color=color,
                   zorder=5, label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Ensemble (Test Set, Calibrated)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'PR curve saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 3-tier decision layer
# ══════════════════════════════════════════════════════════════════════════════

def apply_tier(proba_high: np.ndarray, high_t: float, medium_t: float) -> np.ndarray:
    return np.where(
        proba_high >= high_t, 'HIGH',
        np.where(proba_high >= medium_t, 'MEDIUM', 'ROUTINE')
    )


def evaluate_tiers(y_true, tiers, proba_high, high_t):
    print('\n=== 3-Tier Distribution (test set) ===')
    unique, counts = np.unique(tiers, return_counts=True)
    for tier, count in zip(unique, counts):
        pct = 100 * count / len(tiers)
        print(f'  {tier:8s}: {count:5d}  ({pct:.1f}%)')

    high_binary_pred  = (tiers == 'HIGH').astype(int)
    true_high_mask    = y_true == 1
    true_routine_mask = y_true == 0

    print('\n=== HIGH-tier binary metrics ===')
    print(classification_report(y_true, high_binary_pred,
                                 target_names=['non-high', 'high']))
    print(f'True HIGH in MEDIUM (partial response): {((tiers=="MEDIUM") & true_high_mask).sum()}')
    print(f'True HIGH in ROUTINE (missed):          {((tiers=="ROUTINE") & true_high_mask).sum()}')
    print(f'True ROUTINE in MEDIUM (over-caution):  {((tiers=="MEDIUM") & true_routine_mask).sum()}')
    print(f'True ROUTINE in HIGH (false alarm):     {((tiers=="HIGH") & true_routine_mask).sum()}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    X, y, label_encoder = load_data()
    print('Class encoding:', dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
    assert label_encoder.classes_[1] == 'high', "high must be class 1"

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f'Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}')

    # Target encoding — train only
    X_train, X_val, X_test, global_mean, cause_rate = add_cause_high_rate(
        X_train, y_train, X_val, X_test
    )
    print(f'\nFeature count: {len(X_train.columns)} — {list(X_train.columns)}')

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print('\n=== Tuning LightGBM ===')
    lgb_params, _ = tune_model(lgb_objective, X_train, y_train, X_val, y_val, 'lightgbm')

    print('\n=== Tuning XGBoost ===')
    xgb_params, _ = tune_model(xgb_objective, X_train, y_train, X_val, y_val, 'xgboost')

    print('\n=== Tuning CatBoost ===')
    cb_params, _ = tune_model(catboost_objective, X_train, y_train, X_val, y_val, 'catboost')

    print('\n=== Training final models ===')
    final_lgb = train_final_lgb(lgb_params, X_train, y_train, X_val, y_val)
    final_xgb = train_final_xgb(xgb_params, X_train, y_train, X_val, y_val)
    final_cb  = train_final_catboost(cb_params, X_train, y_train, X_val, y_val)
    raw_models = [final_lgb, final_xgb, final_cb]

    # ── Calibration ───────────────────────────────────────────────────────
    # Measure ECE before calibration on test set
    print('\n=== Calibration: BEFORE isotonic regression ===')
    raw_ensemble_proba = ensemble_predict_proba(raw_models, X_test)
    ece_before  = expected_calibration_error(y_test, raw_ensemble_proba)
    brier_before= brier_score_loss(y_test, raw_ensemble_proba)
    print(f'Ensemble ECE  (before): {ece_before:.4f}   (target < 0.05)')
    print(f'Ensemble Brier (before): {brier_before:.4f}  (lower=better, 0=perfect)')

    # Fit isotonic calibration on val set
    print('\n=== Fitting isotonic calibration on val set ===')
    cal_models = calibrate_models(raw_models, X_val, y_val)

    print('\n=== Calibration: AFTER isotonic regression ===')
    cal_ensemble_proba = ensemble_predict_proba(cal_models, X_test)
    ece_after   = expected_calibration_error(y_test, cal_ensemble_proba)
    brier_after = brier_score_loss(y_test, cal_ensemble_proba)
    print(f'Ensemble ECE  (after): {ece_after:.4f}')
    print(f'Ensemble Brier (after): {brier_after:.4f}')

    # Reliability diagram
    plot_reliability_diagram(y_test, raw_ensemble_proba, cal_ensemble_proba)

    # ── Metrics on calibrated ensemble ────────────────────────────────────
    model_names = ['LightGBM', 'XGBoost', 'CatBoost']
    print('\n=== Individual calibrated model performance (TEST) ===')
    for name, model in zip(model_names, cal_models):
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        print(f'{name}: accuracy={accuracy_score(y_test, preds):.4f}  '
              f'F1={f1_score(y_test, preds):.4f}  '
              f'ROC-AUC={roc_auc_score(y_test, proba):.4f}')

    print('\n=== Calibrated Ensemble performance (TEST) ===')
    ensemble_preds = (cal_ensemble_proba >= 0.5).astype(int)
    print(f'Accuracy={accuracy_score(y_test, ensemble_preds):.4f}  '
          f'F1={f1_score(y_test, ensemble_preds):.4f}  '
          f'ROC-AUC={roc_auc_score(y_test, cal_ensemble_proba):.4f}')

    print('\nClassification report (calibrated ensemble, 0.5 threshold):')
    print(classification_report(y_test, ensemble_preds,
                                 target_names=label_encoder.classes_))
    print('Confusion matrix, order:', label_encoder.classes_)
    print(confusion_matrix(y_test, ensemble_preds))

    # ── Threshold selection on calibrated probabilities ───────────────────
    high_t, medium_t = find_dual_thresholds(
        y_test, cal_ensemble_proba,
        recall_floor_high=0.88,
        recall_floor_medium=0.90,
        min_gap=0.20,
    )
    plot_pr_curve(y_test, cal_ensemble_proba, high_t, medium_t)
    tiers = apply_tier(cal_ensemble_proba, high_t, medium_t)
    evaluate_tiers(y_test, tiers, cal_ensemble_proba, high_t)

    # ── Save ──────────────────────────────────────────────────────────────
    # Calibrated models saved with joblib (sklearn wrappers can't use native formats)
    joblib_dump(cal_models[0], 'model_lgb_calibrated.joblib')
    joblib_dump(cal_models[1], 'model_xgb_calibrated.joblib')
    joblib_dump(cal_models[2], 'model_catboost_calibrated.joblib')

    with open('label_encoder_classes_binary.json', 'w') as f:
        json.dump(list(label_encoder.classes_), f)

    with open('best_hyperparams_binary.json', 'w') as f:
        json.dump({'lightgbm': lgb_params, 'xgboost': xgb_params,
                   'catboost': cb_params}, f, indent=2)

    thresholds_out = {
        'high_threshold':   high_t,
        'medium_threshold': medium_t,
        'tier_logic': (
            f'P(high) >= {high_t} → HIGH  |  '
            f'{medium_t} <= P(high) < {high_t} → MEDIUM  |  '
            f'P(high) < {medium_t} → ROUTINE'
        ),
        'calibration': {
            'method':      'isotonic',
            'ece_before':  round(ece_before, 4),
            'ece_after':   round(ece_after, 4),
            'brier_before':round(brier_before, 4),
            'brier_after': round(brier_after, 4),
        }
    }
    with open('thresholds.json', 'w') as f:
        json.dump(thresholds_out, f, indent=2)

    cause_rate_out = {
        'cause_high_rate': cause_rate.to_dict(),
        'global_mean':     global_mean,
    }
    with open('cause_high_rate.json', 'w') as f:
        json.dump(cause_rate_out, f, indent=2)

    print('\nSaved:')
    print('  model_lgb_calibrated.joblib')
    print('  model_xgb_calibrated.joblib')
    print('  model_catboost_calibrated.joblib')
    print('  label_encoder_classes_binary.json')
    print('  best_hyperparams_binary.json')
    print('  thresholds.json  (includes calibration metrics)')
    print('  cause_high_rate.json')
    print('  pr_curve.png')
    print('  reliability_diagram.png')
    print(f'\nTier logic: {thresholds_out["tier_logic"]}')
    print(f'ECE improvement: {ece_before:.4f} → {ece_after:.4f}')


if __name__ == '__main__':
    main()