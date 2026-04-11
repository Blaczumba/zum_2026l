import yaml
import time
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from datasets import load_stellar, load_sgemm
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from neural_network import train_nn_classifier, train_nn_regressor

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_regression_experiment(config: dict):
    print("\n--- REGRESSION EXPERIMENT (SGEMM GPU Kernel Performance) ---")
    # Using a subsample of 10,000 to keep pure python Custom GBM execution time reasonable.
    X_train, X_test, y_train, y_test = load_sgemm("sgemm_product.csv", sample_size=10000)
    print(f"Data shape: Train: {X_train.shape}, Test: {X_test.shape}")

    cfg = config['regression']
    
    # 1. Custom GBM
    print("\nTraining Custom Gradient Boosting...")
    custom_model = GradientBoostingRegressor(
        n_estimators=cfg['n_estimators'], 
        learning_rate=cfg['learning_rate'], 
        max_depth=cfg['max_depth']
    )
    start_time = time.time()
    custom_model.fit(X_train, y_train)
    train_time_custom = time.time() - start_time
    
    start_time = time.time()
    preds_custom = custom_model.predict(X_test)
    inf_time_custom = time.time() - start_time
    
    # 2. XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=cfg['n_estimators'], 
        learning_rate=cfg['learning_rate'], 
        max_depth=cfg['max_depth'],
        n_jobs=-1
    )
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    
    start_time = time.time()
    preds_xgb = xgb_model.predict(X_test)
    inf_time_xgb = time.time() - start_time

    # 3. Neural Network (PyTorch)
    print("Training Neural Network (PyTorch)...")
    start_time = time.time()
    preds_nn = train_nn_regressor(X_train, y_train, X_test)
    train_inf_time_nn = time.time() - start_time

    # Results
    print("\n--- REGRESSION RESULTS ---")
    print(f"{'Model':<20} | {'MSE':<15} | {'R2 Score':<15} | {'Train Time (s)':<15} | {'Inf Time (s)':<15}")
    print("-" * 88)
    print(f"{'Custom GBM':<20} | {mean_squared_error(y_test, preds_custom):<15.4f} | {r2_score(y_test, preds_custom):<15.4f} | {train_time_custom:<15.4f} | {inf_time_custom:<15.4f}")
    print(f"{'XGBoost':<20} | {mean_squared_error(y_test, preds_xgb):<15.4f} | {r2_score(y_test, preds_xgb):<15.4f} | {train_time_xgb:<15.4f} | {inf_time_xgb:<15.4f}")
    print(f"{'PyTorch NN':<20} | {mean_squared_error(y_test, preds_nn):<15.4f} | {r2_score(y_test, preds_nn):<15.4f} | {train_inf_time_nn:<15.4f} | {'N/A':<15}")


def run_classification_experiment(config: dict):
    print("\n--- CLASSIFICATION EXPERIMENT (Stellar Classification Dataset) ---")
    # Subsample 6000 items
    X_train, X_test, y_train, y_test, le = load_stellar("star_classification.csv", sample_size=6000)
    print(f"Data shape: Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {le.classes_}")

    cfg = config['classification']
    
    # 1. Custom GBM
    print("\nTraining Custom Gradient Boosting Classifier...")
    custom_model = GradientBoostingClassifier(
        n_estimators=cfg['n_estimators'], 
        learning_rate=cfg['learning_rate'], 
        max_depth=cfg['max_depth']
    )
    start_time = time.time()
    custom_model.fit(X_train, y_train)
    train_time_custom = time.time() - start_time
    
    start_time = time.time()
    preds_custom = custom_model.predict(X_test)
    inf_time_custom = time.time() - start_time

    # 2. XGBoost
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=cfg['n_estimators'], 
        learning_rate=cfg['learning_rate'], 
        max_depth=cfg['max_depth'],
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    
    start_time = time.time()
    preds_xgb = xgb_model.predict(X_test)
    inf_time_xgb = time.time() - start_time

    # 3. Neural Network (PyTorch)
    print("Training Neural Network (PyTorch)...")
    start_time = time.time()
    preds_nn = train_nn_classifier(X_train, y_train, X_test)
    train_inf_time_nn = time.time() - start_time

    # Results
    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"{'Model':<20} | {'Accuracy':<15} | {'Train Time (s)':<15} | {'Inf Time (s)':<15}")
    print("-" * 75)
    print(f"{'Custom GBM':<20} | {accuracy_score(y_test, preds_custom):<15.4f} | {train_time_custom:<15.4f} | {inf_time_custom:<15.4f}")
    print(f"{'XGBoost':<20} | {accuracy_score(y_test, preds_xgb):<15.4f} | {train_time_xgb:<15.4f} | {inf_time_xgb:<15.4f}")
    print(f"{'PyTorch NN':<20} | {accuracy_score(y_test, preds_nn):<15.4f} | {train_inf_time_nn:<15.4f} | {'N/A':<15}")


if __name__ == "__main__":
    conf = load_config("config.yaml")
    
    try:
        run_regression_experiment(conf)
    except FileNotFoundError:
        print("\nERROR: Please download 'sgemm_product.csv' and place it in the project root.")
        
    try:
        run_classification_experiment(conf)
    except FileNotFoundError:
        print("\nERROR: Please download 'star_classification.csv' and place it in the project root.")