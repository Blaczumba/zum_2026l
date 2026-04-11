import numpy as np

class MSELoss:
    """Mean Squared Error Loss for Regression."""
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(0.5 * (y_true - y_pred) ** 2)

    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Derivative of 0.5 * (y - y_pred)^2 w.r.t y_pred is (y_pred - y).
        # Negative gradient is y - y_pred
        return y_true - y_pred

class CrossEntropyLoss:
    """Cross Entropy Loss for Multiclass Classification."""
    def compute_loss(self, y_true_one_hot: np.ndarray, probs: np.ndarray) -> float:
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true_one_hot * np.log(probs), axis=1))

    def negative_gradient(self, y_true_one_hot: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return y_true_one_hot - probs