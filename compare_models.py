import time
import numpy as np
import importlib.util as iu
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Try to import PyYAML; fall back to defaults if missing
try:
    import yaml
    def load_config(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
except Exception:
    print("PyYAML not installed; using default config values and ignoring config.yaml.")
    def load_config(path: str) -> dict:
        return {
            'regression': {'n_estimators': 20, 'learning_rate': 0.1, 'max_depth': 10},
            'classification': {'n_estimators': 20, 'learning_rate': 0.1, 'max_depth': 3}
        }

from datasets import load_stellar, load_sgemm
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from visualization import plot_regression_scatter, plot_confusion_matrices, plot_bar_comparison

# Helper to check availability
HAS_XGBOOST = iu.find_spec('xgboost') is not None
HAS_TORCH = iu.find_spec('torch') is not None

if HAS_XGBOOST:
    from xgboost import XGBClassifier, XGBRegressor

# We don't import neural_network at top-level because it depends on torch


def run_regression_experiment(config: dict, data_path: str = 'data/sgemm_product.csv'):
    print("\n--- REGRESSION EXPERIMENT (SGEMM GPU Kernel Performance) ---")

    cfg = config.get('regression', {})
    sample_size = cfg.get('sample_size', 10000)
    repeats = int(cfg.get('repeats', 1))
    base_random = int(cfg.get('random_state', 42))

    # configuration for model constructors (filter only expected keys)
    allowed_keys = ('n_estimators', 'learning_rate', 'max_depth')
    model_cfg = {k: v for k, v in cfg.items() if k in allowed_keys}

    # accumulators for repeated experiments
    accum = {
        'Custom GBM': {'mse': [], 'r2': [], 'time': [], 'last_preds': None},
        'XGBoost': {'mse': [], 'r2': [], 'time': [], 'last_preds': None},
        'PyTorch NN': {'mse': [], 'r2': [], 'time': [], 'last_preds': None}
    }

    last_y_test = None

    for rep in range(repeats):
        seed = base_random + rep
        print(f"\nRepeat {rep+1}/{repeats} (random_state={seed})")

        # Load data with reproducible randomness per repeat
        X_train, X_test, y_train, y_test = load_sgemm(data_path, sample_size=sample_size, random_state=seed)
        print(f"Data shape: Train: {X_train.shape}, Test: {X_test.shape}")

        # 1. Custom GBM
        print("Training Custom Gradient Boosting...")
        custom_model = GradientBoostingRegressor(**model_cfg)
        start_time = time.time()
        custom_model.fit(X_train, y_train)
        train_time_custom = time.time() - start_time
        preds_custom = custom_model.predict(X_test)
        accum['Custom GBM']['mse'].append(mean_squared_error(y_test, preds_custom))
        accum['Custom GBM']['r2'].append(r2_score(y_test, preds_custom))
        accum['Custom GBM']['time'].append(train_time_custom)
        accum['Custom GBM']['last_preds'] = preds_custom

        # 2. XGBoost (if available)
        if HAS_XGBOOST:
            try:
                print("Training XGBoost Regressor...")
                xgb_model = XGBRegressor(**model_cfg, n_jobs=-1)
                start_time = time.time()
                xgb_model.fit(X_train, y_train)
                train_time_xgb = time.time() - start_time
                preds_xgb = xgb_model.predict(X_test)
                accum['XGBoost']['mse'].append(mean_squared_error(y_test, preds_xgb))
                accum['XGBoost']['r2'].append(r2_score(y_test, preds_xgb))
                accum['XGBoost']['time'].append(train_time_xgb)
                accum['XGBoost']['last_preds'] = preds_xgb
            except Exception as e:
                print(f"XGBoost training failed: {e}")
        else:
            print("XGBoost not installed; skipping XGBoost regression.")

        # 3. Neural Network (if torch available)
        if HAS_TORCH:
            try:
                from neural_network import train_nn_regressor
                print("Training Neural Network (PyTorch)...")
                start_time = time.time()
                model_nn, preds_nn = train_nn_regressor(X_train, y_train, X_test)
                train_time_nn = time.time() - start_time
                accum['PyTorch NN']['mse'].append(mean_squared_error(y_test, preds_nn))
                accum['PyTorch NN']['r2'].append(r2_score(y_test, preds_nn))
                accum['PyTorch NN']['time'].append(train_time_nn)
                accum['PyTorch NN']['last_preds'] = preds_nn
            except Exception as e:
                print(f"Neural network training failed: {e}")
        else:
            print("PyTorch not installed; skipping neural network regression.")

        last_y_test = y_test

    # Summarize results across repeats
    print("\n--- REGRESSION RESULTS (averaged over repeats) ---")
    print(f"{'Model':<20} | {'MSE mean ± std':<30} | {'R2 mean ± std':<30} | {'Train Time mean ± std (s)':<30}")
    print("-" * 120)

    results = {}
    for name in ['Custom GBM', 'XGBoost', 'PyTorch NN']:
        entries = accum.get(name, {})
        if len(entries.get('mse', [])) > 0:
            mse_arr = np.array(entries['mse'])
            r2_arr = np.array(entries['r2'])
            time_arr = np.array(entries['time'])
            print(f"{name:<20} | {mse_arr.mean():<8.4f} ± {mse_arr.std():<8.4f} | {r2_arr.mean():<8.4f} ± {r2_arr.std():<8.4f} | {time_arr.mean():<8.4f} ± {time_arr.std():<8.4f}")
            results[name] = {
                'mse_mean': float(mse_arr.mean()), 'mse_std': float(mse_arr.std()),
                'r2_mean': float(r2_arr.mean()), 'r2_std': float(r2_arr.std()),
                'time_mean': float(time_arr.mean()), 'time_std': float(time_arr.std()),
                'last_preds': entries.get('last_preds')
            }
        else:
            print(f"{name:<20} | {'N/A':<30} | {'N/A':<30} | {'N/A':<30}")

    # --- VISUALIZATIONS ---
    # Use the last repeat's test set and predictions for plotting
    preds_dict = {name: v['last_preds'] for name, v in accum.items() if v.get('last_preds') is not None}
    mse_dict = {name: results[name]['mse_mean'] for name in results}
    time_dict = {name: results[name]['time_mean'] for name in results}

    if preds_dict and last_y_test is not None:
        print("\nGenerating regression plots (from last repeat)...")
        plot_regression_scatter(last_y_test, preds_dict)
        plot_bar_comparison(mse_dict, "Porównanie Błędu Średniokwadratowego (MSE)", "MSE", "regression_mse_bar.png")
        plot_bar_comparison(time_dict, "Porównanie Czasu Treningu (Regresja)", "Sekundy", "regression_time_bar.png")

    return results


def run_classification_experiment(config: dict, data_path: str = 'data/star_classification.csv'):
    print("\n--- CLASSIFICATION EXPERIMENT (Stellar Classification Dataset) ---")
    try:
        X_train, X_test, y_train, y_test, le = load_stellar(data_path, sample_size=6000, target_col='MJD')
    except FileNotFoundError:
        raise

    print(f"Data shape: Train: {X_train.shape}, Test: {X_test.shape}")
    class_names = list(le.classes_)

    cfg = config.get('classification', {})
    results = {}

    # 1. Custom GBM
    print("Training Custom Gradient Boosting Classifier...")
    custom_model = GradientBoostingClassifier(**cfg)
    start_time = time.time()
    custom_model.fit(X_train, y_train)
    train_time_custom = time.time() - start_time
    preds_custom = custom_model.predict(X_test)
    results['Custom GBM'] = {'preds': preds_custom, 'time': train_time_custom,
                             'acc': accuracy_score(y_test, preds_custom)}

    # 2. XGBoost
    if HAS_XGBOOST:
        try:
            print("Training XGBoost Classifier...")
            xgb_model = XGBClassifier(**cfg, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            train_time_xgb = time.time() - start_time
            preds_xgb = xgb_model.predict(X_test)
            results['XGBoost'] = {'preds': preds_xgb, 'time': train_time_xgb,
                                  'acc': accuracy_score(y_test, preds_xgb)}
        except Exception as e:
            print(f"XGBoost training failed: {e}")
    else:
        print("XGBoost not installed; skipping XGBoost classification.")

    # 3. Neural Network
    if HAS_TORCH:
        try:
            from neural_network import train_nn_classifier
            print("Training Neural Network (PyTorch)...")
            start_time = time.time()
            model_nn_clf, preds_nn = train_nn_classifier(X_train, y_train, X_test)
            train_time_nn = time.time() - start_time
            results['PyTorch NN'] = {'preds': preds_nn, 'time': train_time_nn,
                                        'acc': accuracy_score(y_test, preds_nn)}
        except Exception as e:
                print(f"Neural network training failed: {e}")
    else:
        print("PyTorch not installed; skipping neural network classification.")

    # Print results
    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"{'Model':<20} | {'Accuracy':<15} | {'Train Time (s)':<15}")
    print("-" * 55)
    for name in ['Custom GBM', 'XGBoost', 'PyTorch NN']:
        if name in results:
            r = results[name]
            print(f"{name:<20} | {r.get('acc', 0):<15.4f} | {r['time']:<15.4f}")
        else:
            print(f"{name:<20} | {'N/A':<15} | {'N/A':<15}")

    # Visualizations
    preds_dict = {name: v['preds'] for name, v in results.items()}
    acc_dict = {name: v['acc'] for name, v in results.items()}
    time_dict = {name: v['time'] for name, v in results.items()}

    if preds_dict:
        print("\nGenerating classification plots...")
        plot_confusion_matrices(y_test, preds_dict, class_names)
        plot_bar_comparison(acc_dict, "Porównanie Dokładności (Accuracy)", "Accuracy", "classification_acc_bar.png")
        plot_bar_comparison(time_dict, "Porównanie Czasu Treningu (Klasyfikacja)", "Sekundy", "classification_time_bar.png")

    return results


if __name__ == '__main__':
    conf = load_config('config.yaml')

    # Run regression; if dataset missing, inform user
    try:
        run_regression_experiment(conf)
    except FileNotFoundError:
        print("\nERROR: Please put 'sgemm_product.csv' in the 'data/' directory.")

    # Run classification if dataset exists
    try:
        run_classification_experiment(conf)
    except FileNotFoundError:
        print("\nERROR: Please put 'star_classification.csv' in the 'data/' directory.")
