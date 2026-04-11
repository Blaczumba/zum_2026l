import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

def load_stellar(filepath: str, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(filepath)
    
    if sample_size and sample_size < len(df):
        # We balance the sample slightly if subsampling
        df = df.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size // 3)))
        
    X = df.drop(columns=['class', 'obj_ID']) # drop identifier and target
    y_raw = df['class']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, le

def load_sgemm(filepath: str, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        
    run_cols = ['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']
    
    # Target: average of 4 runs
    y = df[run_cols].mean(axis=1).values
    X = df.drop(columns=run_cols).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test