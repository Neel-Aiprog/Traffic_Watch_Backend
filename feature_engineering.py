"""
Feature engineering v2 — addresses sparsity diagnosis:
  - hour_of_day (24 levels) -> time_of_day (5 coarser buckets)
  - corridor (24 levels, long tail) -> rare corridors (<100 samples)
    collapsed into 'other_corridor'

Diagnosis that drove this change: with the original feature set, 1219
unique (event_cause, corridor, hour_of_day) combinations existed across
~7000 rows -- 44.8% appeared exactly once, only 26.6% had >=5 samples.
Median within-group class purity was 1.0 (features ARE separable when
enough samples exist per group) -- the problem was granularity, not
inherent noise. Coarsening hour_of_day and corridor increases samples
per group so boosted trees can actually learn stable splits instead of
memorizing near-unique combinations.
"""

import pandas as pd
from target_engineering import add_severity_target_binary

TIER1_CATEGORICAL = ['event_cause', 'corridor_grouped', 'priority']
TIER1_BOOLEAN = ['requires_road_closure']
TIER2_CATEGORICAL = ['veh_type']

CATEGORICAL_FEATURES = TIER1_CATEGORICAL + TIER2_CATEGORICAL + ['day_of_week', 'time_of_day']
ALL_FEATURE_COLS = (
    TIER1_CATEGORICAL + TIER1_BOOLEAN + TIER2_CATEGORICAL
    + ['time_of_day', 'day_of_week', 'is_weekend']
)

MIN_CORRIDOR_COUNT = 100   # corridors with fewer rows than this get grouped into 'other_corridor'


def _normalize_event_cause(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower()


def _bucket_hour(hour: int) -> str:
    """Coarsen 24 hourly levels into 5 operationally meaningful buckets.
    Matches the diurnal pattern found in EDA: heavy clustering 19:00-23:00
    and 04:00-07:00, near-zero 12:00-17:00."""
    if 0 <= hour < 4:
        return 'late_night'        # 00:00-03:59
    elif 4 <= hour < 8:
        return 'early_morning'     # 04:00-07:59 (EDA spike)
    elif 8 <= hour < 17:
        return 'daytime'           # 08:00-16:59 (EDA low-activity window)
    elif 17 <= hour < 20:
        return 'evening'           # 17:00-19:59
    else:
        return 'night'             # 20:00-23:59 (EDA spike)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['time_of_day'] = df['hour_of_day'].apply(_bucket_hour)
    df['day_of_week'] = df['start_datetime'].dt.day_name()
    df['is_weekend'] = df['start_datetime'].dt.dayofweek.isin([5, 6])
    return df


def _group_rare_corridors(df: pd.DataFrame, min_count: int = MIN_CORRIDOR_COUNT) -> pd.Series:
      # DEBUG: Print what we're working with


      # Skip rare corridor grouping during inference (small dataframes)
      if len(df) < 30:  # Heuristic: <30 rows = likely inference/batch

          result = df['corridor']

          return result


      counts = df['corridor'].value_counts()

      rare = counts[counts < min_count].index

      result = df['corridor'].apply(lambda c: 'other_corridor' if c in rare else c)

      return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['event_cause'] = _normalize_event_cause(df['event_cause'])
    df = add_temporal_features(df)

    df['veh_type'] = df['veh_type'].fillna('unknown')
    df['corridor'] = df['corridor'].fillna('Non-corridor')
    df['corridor_grouped'] = _group_rare_corridors(df)

    df = df[df['priority'].notna()]

    model_df = df[ALL_FEATURE_COLS + ['severity']].copy()

    for col in CATEGORICAL_FEATURES:
        model_df[col] = model_df[col].astype('category')

    return model_df


if __name__ == '__main__':
    PATH = 'dataset.csv'
    raw = pd.read_csv(PATH)
    raw = add_severity_target_binary(raw)
    features = build_features(raw)

    print('Final model-ready shape:', features.shape)
    print()
    print('Dtypes:')
    print(features.dtypes)
    print()
    print('Null check (should be all zero):')
    print(features.isnull().sum())
    print()
    print('corridor_grouped value counts:')
    print(features['corridor_grouped'].value_counts())
    print()
    print('time_of_day value counts:')
    print(features['time_of_day'].value_counts())
    print()
    print('Severity distribution in final frame:')
    print(features['severity'].value_counts())

    # re-run the sparsity diagnostic on the NEW grouping to confirm improvement
    group_cols = ['event_cause', 'corridor_grouped', 'time_of_day']
    group_sizes = features.groupby(group_cols).size()
    print()
    print(f'New unique groups: {len(group_sizes)} (was 1219)')
    print(f'Groups with >=5 samples: {(group_sizes >= 5).sum()} ({(group_sizes>=5).mean()*100:.1f}%) (was 26.6%)')
    print(f'Groups with 1 sample only: {(group_sizes == 1).sum()} ({(group_sizes==1).mean()*100:.1f}%) (was 44.8%)')

    features.to_csv('model_ready_features.csv', index=False)
    print()
    print('Saved to model_ready_features.csv')