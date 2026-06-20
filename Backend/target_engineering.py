"""
Target engineering v2 — 2-class reframe.

Diagnosis from 3-class modeling: low vs medium severity (both dominated by
vehicle_breakdown events) showed near-identical feature distributions
across every available structured feature (corridor, cause, time-of-day) --
median within-group purity was high only when groups had enough samples,
but even with enough samples, low/medium ratio stayed ~70/25-30 regardless
of corridor or time bucket. This is very likely driven by operational
factors not present in this data (how fast a tow truck arrived, real-time
traffic at that moment) rather than a fixable feature gap.

The 'high' class, by contrast, IS separable -- the original 3-class model
got 0.91 recall / 0.68 precision on it with no extra effort. So we keep
that distinction and collapse the unreliable boundary:

  routine : <= 240 min (was low + medium)
  high    : > 240 min  (sustained disruption, needs active resource allocation)

This produces a binary classification problem with a real, demonstrable
signal, instead of a 3-class problem where 1 of 3 classes is close to
unlearnable noise.
"""

import pandas as pd

SEVERITY_THRESHOLD_MINUTES = 240


def add_severity_target_binary(df: pd.DataFrame, drop_active: bool = True) -> pd.DataFrame:
    df = df.copy()

    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce', utc=True)

    # Active events are right-censored -- see v1 rationale, same logic applies.
    if drop_active:
        df = df[df['status'] != 'active']

    df['duration_minutes'] = (
        (df['modified_datetime'] - df['start_datetime']).dt.total_seconds() / 60
    )

    df = df[df['duration_minutes'].notna()]
    df = df[df['duration_minutes'] >= 0]

    df['duration_minutes_clipped'] = df['duration_minutes'].clip(upper=1440)

    df['severity'] = df['duration_minutes_clipped'].apply(
        lambda m: 'high' if m > SEVERITY_THRESHOLD_MINUTES else 'routine'
    )

    return df


if __name__ == '__main__':
    PATH = 'dataset.csv'
    df = pd.read_csv(PATH)
    df = add_severity_target_binary(df)

    print('Rows after dropping unparseable/negative/active durations:', len(df))
    print()
    print('Severity distribution (binary):')
    print(df['severity'].value_counts())
    print((df['severity'].value_counts(normalize=True) * 100).round(1))