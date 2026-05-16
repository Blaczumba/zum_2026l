import numpy as np
from decision_tree import DecisionTreeRegressor
from losses import MSELoss, CrossEntropyLoss
from typing import List

class GradientBoostingRegressor:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = MSELoss()
        self.trees: List[DecisionTreeRegressor] = []
        self.initial_prediction = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)

        for i in range(self.n_estimators):
            pseudo_residuals = self.loss.negative_gradient(y, predictions)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, pseudo_residuals)
            update = tree.predict(X)
            predictions += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

class GradientBoostingClassifier:
    """Multiclass Gradient Boosting"""
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = CrossEntropyLoss()
        self.trees: List[List[DecisionTreeRegressor]] = [] # K trees per iteration
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(np.unique(y))

        # One-hot encode targets
        y_one_hot = np.zeros((y.size, self.n_classes))
        y_one_hot[np.arange(y.size), y] = 1

        # Initial logits (log odds) - initialized to 0
        logits = np.zeros((X.shape[0], self.n_classes))

        for i in range(self.n_estimators):
            probs = self._softmax(logits)
            pseudo_residuals = self.loss.negative_gradient(y_one_hot, probs)

            iteration_trees = []
            for k in range(self.n_classes):
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, pseudo_residuals[:, k])
                update = tree.predict(X)
                logits[:, k] += self.learning_rate * update
                iteration_trees.append(tree)

            self.trees.append(iteration_trees)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.zeros((X.shape[0], self.n_classes))
        for iteration_trees in self.trees:
            for k, tree in enumerate(iteration_trees):
                logits[:, k] += self.learning_rate * tree.predict(X)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
