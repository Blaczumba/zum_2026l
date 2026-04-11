import numpy as np
from typing import Optional, Tuple

class Node:
    def __init__(self, feature_idx: int = None, threshold: float = None,
                 left: 'Node' = None, right: 'Node' = None, value: float = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only for leaf nodes

class DecisionTreeRegressor:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            return Node(value=np.mean(y))

        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None: # Cannot split further
            return Node(value=np.mean(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[Optional[int], Optional[float]]:
        best_variance_reduction = -float('inf')
        best_feat, best_thresh = None, None
        parent_variance = np.var(y)
        
        # To speed up pure python implementation, we check only percentiles instead of all unique values
        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.percentile(X_col, np.arange(10, 100, 10)) 
            thresholds = np.unique(thresholds)
            
            for threshold in thresholds:
                left_idx = X_col <= threshold
                right_idx = X_col > threshold
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                    
                n_left, n_right = len(y[left_idx]), len(y[right_idx])
                n_total = len(y)
                
                var_left = np.var(y[left_idx])
                var_right = np.var(y[right_idx])
                
                # Weighted variance of children
                child_variance = (n_left / n_total) * var_left + (n_right / n_total) * var_right
                variance_reduction = parent_variance - child_variance
                
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)