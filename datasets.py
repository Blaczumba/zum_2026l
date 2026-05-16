import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple


def _detect_target_column(df: pd.DataFrame) -> str:
    # Common names
    for name in df.columns:
        if name.lower() in ('class', 'label', 'target', 'type', 'classification'):
            return name

    # Prefer object dtype columns (likely categorical)
    for name in df.columns:
        if df[name].dtype == object:
            return name

    # Otherwise prefer a column with relatively few unique values
    n = len(df)
    for name in df.columns:
        nunique = df[name].nunique()
        if 1 < nunique <= max(50, int(0.05 * n)):
            return name

    return None


def load_stellar(filepath: str, sample_size: int = None, target_col: str = None, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(filepath)

    # Normalize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # If user provided a target_col, try to match it case-insensitively
    if target_col:
        matches = [c for c in df.columns if c.lower() == target_col.lower()]
        if len(matches) == 0:
            raise ValueError(f"Provided target_col '{target_col}' not found in CSV columns: {df.columns.tolist()}")
        target_col = matches[0]
    else:
        # Detect which column is the target (default expected: 'class')
        target_col = _detect_target_column(df)
        if target_col is None:
            raise ValueError("Could not detect target column in stellar dataset. Expected a 'class' or 'label' column.")

    # Optionally subsample while keeping class balance
    if sample_size and sample_size < len(df):
        try:
            n_classes = df[target_col].nunique()
            per_class = max(1, sample_size // n_classes)
            df = df.groupby(target_col, group_keys=False).apply(lambda x: x.sample(min(len(x), per_class), random_state=random_state)).reset_index(drop=True)
        except Exception:
            # If grouping fails for any reason, fallback to a plain random sample
            df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # Ensure columns are normalized after any transformations
    df.columns = df.columns.str.strip()

    # If the target column disappeared or changed, try to re-detect
    if target_col not in df.columns:
        # try case-insensitive match
        matches = [c for c in df.columns if c.lower() == str(target_col).lower()]
        if matches:
            target_col = matches[0]
        else:
            target_col = _detect_target_column(df)
            if target_col is None:
                raise ValueError("Could not detect target column after sampling. Inspect the CSV headers.")

    # Identify a possible identifier column to drop (common names)
    id_candidates = [c for c in df.columns if c.lower() in ('obj_id', 'objid', 'obj id', 'id', 'objectid')]
    drop_cols = [c for c in id_candidates if c in df.columns]

    # Features: drop target and id columns if present
    features = df.drop(columns=[c for c in ([target_col] + drop_cols) if c in df.columns])
    y_raw = df[target_col]

    # One-hot encode categorical feature columns (but keep target as integer labels)
    cat_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        features_processed = pd.get_dummies(features, columns=cat_cols, drop_first=False)
    else:
        features_processed = features.copy()

    # Coerce any remaining non-numeric columns to numeric (if possible), filling NaNs with column mean
    for col in features_processed.columns:
        if features_processed[col].dtype == object:
            features_processed[col] = pd.to_numeric(features_processed[col], errors='coerce')
    # Fill NaNs with column mean
    features_processed = features_processed.fillna(features_processed.mean())

    # Encode target (labels) as integers for classifiers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Scale features (now numeric)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_processed.values)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, le


def load_sgemm(filepath: str, sample_size: int = None, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Expected run columns (case-insensitive matching)
    expected = ['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']
    run_cols = []
    for exp in expected:
        matches = [c for c in df.columns if c.lower() == exp.lower()]
        if matches:
            run_cols.append(matches[0])
        else:
            # try more flexible matching (e.g. missing parentheses or different spacing)
            lowered = exp.lower().replace(' ', '').replace('(', '').replace(')', '')
            for c in df.columns:
                if c.lower().replace(' ', '').replace('(', '').replace(')', '') == lowered:
                    run_cols.append(c)
                    break

    if len(run_cols) != len(expected):
        raise ValueError(f"Expected run columns {expected} not all found in CSV. Found: {df.columns.tolist()}")

    # Drop rows where ALL run measurements are missing (keep rows with at least one run)
    df = df[~df[run_cols].isnull().all(axis=1)].reset_index(drop=True)

    # Optionally subsample after cleaning
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # Target: average of available run columns (pandas mean skips NaN by default)
    y = df[run_cols].astype(float).mean(axis=1).values
    X = df.drop(columns=run_cols).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test
