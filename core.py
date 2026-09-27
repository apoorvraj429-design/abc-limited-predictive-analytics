"""Shared model schema, validation and explanations for ABC Limited."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
NUM = ['watch_hours', 'last_login_days', 'number_of_profiles']
CAT = ['subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
CHURN_FEATURES = NUM + CAT
LINEAR_NUM = ['last_login_days', 'number_of_profiles']
LINEAR_FEATURES = LINEAR_NUM + CAT
LABELS = {'watch_hours': 'Recorded viewing hours', 'last_login_days': 'Days since last login',
          'number_of_profiles': 'Number of profiles', 'subscription_type': 'Subscription plan',
          'region': 'Region', 'device': 'Device', 'payment_method': 'Payment method', 'favorite_genre': 'Favourite genre'}

def load_artifacts():
    meta = json.loads((ROOT / 'results/metrics.json').read_text())
    return joblib.load(ROOT/'models/churn.joblib'), joblib.load(ROOT/'models/viewing.joblib'), meta

def validate_inputs(frame, schema, features=CHURN_FEATURES):
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = frame[features].copy()
    if out.empty or len(out) > 10000:
        raise ValueError('Please provide between 1 and 10,000 customer rows.')
    for c in features:
        if c in NUM:
            out[c] = pd.to_numeric(out[c], errors='coerce')
            lo, hi = schema['ranges'][c]
            if not np.isfinite(out[c]).all() or not out[c].between(lo, hi).all():
                raise ValueError(f'{LABELS[c]} must be a number between {lo} and {hi}.')
            if c != 'watch_hours' and not (out[c] == out[c].round()).all():
                raise ValueError(f'{LABELS[c]} must contain whole numbers.')
        else:
            if out[c].isna().any() or not out[c].isin(schema['categories'][c]).all():
                raise ValueError(f'{LABELS[c]} must use one of: ' + ', '.join(schema['categories'][c]))
    return out

def explanations(model, frame):
    """Exact additive log-odds contributions against training mean/reference categories."""
    prep = model.named_steps['prep']
    names = prep.get_feature_names_out()
    values = prep.transform(frame)
    coef = model.named_steps['model'].coef_[0]
    contributions = values[0] * coef
    result = []
    for feature in CHURN_FEATURES:
        indices = [i for i, n in enumerate(names) if n == 'num__'+feature or n.startswith('cat__'+feature+'_')]
        amount = float(sum(contributions[i] for i in indices))
        result.append({'Factor': LABELS[feature], 'Contribution to log-odds': amount,
                       'Direction': 'Raises model score' if amount > 0 else 'Lowers model score'})
    return pd.DataFrame(result).sort_values('Contribution to log-odds', key=abs, ascending=False)

def safe_csv(frame):
    """Neutralise spreadsheet formulas in free-text exports."""
    out = frame.copy()
    for c in out.select_dtypes(include=['object', 'string']).columns:
        out[c] = out[c].map(lambda v: "'"+v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return out.to_csv(index=False).encode('utf-8')
