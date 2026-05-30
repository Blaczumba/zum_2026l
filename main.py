import yaml
import time
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from datasets import load_stellar, load_sgemm
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from neural_network import train_nn_classifier, train_nn_regressor

# NOWY IMPORT
from visualization import plot_regression_scatter, plot_confusion_matrices, plot_bar_comparison

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_regression_experiment(config: dict):
    print("\n--- REGRESSION EXPERIMENT (SGEMM GPU Kernel Performance) ---")
    X_train, X_test, y_train, y_test = load_sgemm("data/sgemm_product.csv", sample_size=10000)
    print(f"Kształt danych: Train: {X_train.shape}, Test: {X_test.shape}")

    cfg = config['regression']
    
    # 1. Custom GBM
    print("Trenowanie niestandardowego Gradient Boostingu...")
    custom_model = GradientBoostingRegressor(**cfg)
    start_time = time.time()
    custom_model.fit(X_train, y_train)
    train_time_custom = time.time() - start_time
    preds_custom = custom_model.predict(X_test)
    
    # 2. XGBoost
    print("Trenowanie regresora XGBoost...")
    xgb_model = XGBRegressor(**cfg, n_jobs=-1)
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    preds_xgb = xgb_model.predict(X_test)

    # 3. Neural Network
    print("Trenowanie sieci neuronowej (PyTorch)...")
    start_time = time.time()
    model_nn, preds_nn = train_nn_regressor(X_train, y_train, X_test)
    train_time_nn = time.time() - start_time

    # Results calculation
    mse_custom = mean_squared_error(y_test, preds_custom)
    mse_xgb = mean_squared_error(y_test, preds_xgb)
    mse_nn = mean_squared_error(y_test, preds_nn)

    print("\n--- REGRESSION RESULTS ---")
    print(f"{'Model':<20} | {'MSE':<15} | {'R2':<15} | {'Czas treningu (s)':<15}")
    print("-" * 75)
    print(f"{'Custom GBM':<20} | {mse_custom:<15.4f} | {r2_score(y_test, preds_custom):<15.4f} | {train_time_custom:<15.4f}")
    print(f"{'XGBoost':<20} | {mse_xgb:<15.4f} | {r2_score(y_test, preds_xgb):<15.4f} | {train_time_xgb:<15.4f}")
    print(f"{'PyTorch NN':<20} | {mse_nn:<15.4f} | {r2_score(y_test, preds_nn):<15.4f} | {train_time_nn:<15.4f}")

    # --- VISUALIZATIONS ---
    print("\nGenerowanie wykresów regresji...")
    preds_dict = {
        "Custom GBM": preds_custom,
        "XGBoost": preds_xgb,
        "PyTorch NN": preds_nn
    }
    mse_dict = {"Custom GBM": mse_custom, "XGBoost": mse_xgb, "PyTorch NN": mse_nn}
    time_dict = {"Custom GBM": train_time_custom, "XGBoost": train_time_xgb, "PyTorch NN": train_time_nn}

    plot_regression_scatter(y_test, preds_dict)
    plot_bar_comparison(mse_dict, "Porównanie Błędu Średniokwadratowego (MSE)", "MSE", "regression_mse_bar.png")
    plot_bar_comparison(time_dict, "Porównanie Czasu Treningu (Regresja)", "Sekundy", "regression_time_bar.png")


def run_classification_experiment(config: dict):
    print("\n--- EKSPERYMENT KLASYFIKACYJNY (Zbiór danych Stellar Classification) ---")
    X_train, X_test, y_train, y_test, le = load_stellar("data/star_classification.csv", sample_size=6000, target_col='MJD')
    print(f"Kształt danych: Train: {X_train.shape}, Test: {X_test.shape}")
    class_names = list(le.classes_)

    cfg = config['classification']
    
    # 1. Custom GBM
    print("Trenowanie niestandardowego klasyfikatora Gradient Boosting...")
    custom_model = GradientBoostingClassifier(**cfg)
    start_time = time.time()
    custom_model.fit(X_train, y_train)
    train_time_custom = time.time() - start_time
    preds_custom = custom_model.predict(X_test)

    # 2. XGBoost
    print("Trenowanie klasyfikatora XGBoost...")
    xgb_model = XGBClassifier(**cfg, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    preds_xgb = xgb_model.predict(X_test)

    # 3. Neural Network
    print("Trenowanie sieci neuronowej (PyTorch)...")
    start_time = time.time()
    model_nn, preds_nn = train_nn_classifier(X_train, y_train, X_test)
    train_time_nn = time.time() - start_time

    # Results calculation
    acc_custom = accuracy_score(y_test, preds_custom)
    acc_xgb = accuracy_score(y_test, preds_xgb)
    acc_nn = accuracy_score(y_test, preds_nn)

    print("\n--- WYNIKI KLASYFIKACJI ---")
    print(f"{'Model':<20} | {'Dokładność':<15} | {'Czas treningu (s)':<15}")
    print("-" * 55)
    print(f"{'Custom GBM':<20} | {acc_custom:<15.4f} | {train_time_custom:<15.4f}")
    print(f"{'XGBoost':<20} | {acc_xgb:<15.4f} | {train_time_xgb:<15.4f}")
    print(f"{'PyTorch NN':<20} | {acc_nn:<15.4f} | {train_time_nn:<15.4f}")

    # --- VISUALIZATIONS ---
    print("\nGenerowanie wykresów klasyfikacji...")
    preds_dict = {
        "Custom GBM": preds_custom,
        "XGBoost": preds_xgb,
        "PyTorch NN": preds_nn
    }
    acc_dict = {"Custom GBM": acc_custom, "XGBoost": acc_xgb, "PyTorch NN": acc_nn}
    time_dict = {"Custom GBM": train_time_custom, "XGBoost": train_time_xgb, "PyTorch NN": train_time_nn}

    plot_confusion_matrices(y_test, preds_dict, class_names)
    plot_bar_comparison(acc_dict, "Porównanie Dokładności (Accuracy)", "Accuracy", "classification_acc_bar.png")
    plot_bar_comparison(time_dict, "Porównanie Czasu Treningu (Klasyfikacja)", "Sekundy", "classification_time_bar.png")


if __name__ == "__main__":
    conf = load_config("config.yaml")
    
    try:
        run_regression_experiment(conf)
    except FileNotFoundError:
        print("\nBŁĄD: Proszę pobrać 'sgemm_product.csv' i umieścić go w katalogu projektu.")
        
    try:
        run_classification_experiment(conf)
    except FileNotFoundError:
        print("\nBŁĄD: Proszę pobrać 'star_classification.csv' i umieścić go w katalogu projektu.")