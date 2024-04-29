import numpy as np
from collections import Counter



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(self.n_features, x.shape[1]) #x.shape[1]: Zwraca liczbę cech w macierzy x.
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))


        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)


        best_feat_idx, best_thresh = self._best_split(x, y, feat_idxs)


        left_idxs, right_idxs = self._split(x[:, best_feat_idx], best_thresh)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat_idx, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_split(self, x, y, feat_idxs):
        best_gain = -1
        best_feat_idx, best_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = x[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feat_idx = feat_idx
                    best_thresh = threshold
        return best_feat_idx, best_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left = self._entropy(y[left_idxs])
        entropy_right = self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, x):
        return np.array([self._traverse_tree(x_i, self.root) for x_i in x])
