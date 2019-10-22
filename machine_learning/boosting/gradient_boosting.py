import numpy as np
from scipy.optimize import line_search
from sklearn.tree import DecisionTreeRegressor


class LogisticGB:
    def __init__(self, max_depth, n_estimators):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.const_pred = 0
        self.trees = []
        self.alphas = []

    @staticmethod
    def logistic_loss(y_true, y_pred):
        return np.sum(np.log(1 + np.exp(-y_true * y_pred)))

    @staticmethod
    def logistic_function(t):
        return 1 / (1 + np.exp(-t))

    def logistic_grad(self, y_true, y_pred):
        return -y_true * self.logistic_function(-y_true * y_pred)

    def fit_step(self, X, y, curr_pred):
        residuals = -self.logistic_grad(y, curr_pred)
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X, residuals)
        tree_pred = tree.predict(X)

        alpha = self.find_alpha(y, curr_pred, tree_pred)
        curr_pred += alpha * tree_pred

        self.trees.append(tree)
        self.alphas.append(alpha)
        return curr_pred

    def find_alpha(self, y_true, curr_pred, tree_pred):

        def alpha_obj(x):
            return self.logistic_loss(y_true, x)

        def alpha_grad(x):
            return self.logistic_grad(y_true, x)

        alpha = line_search(alpha_obj, alpha_grad, xk=curr_pred, pk=tree_pred)
        if not alpha[0]:
            return 1.0
        return alpha[0]

    def fit(self, X, y):
        curr_pred = np.ones_like(y, dtype=np.float64) * self.const_pred
        for i in range(self.n_estimators):
            curr_pred = self.fit_step(X, y, curr_pred)

    def predict_proba(self, X):
        y_pred = np.ones(shape=X.shape[0]) * self.const_pred
        for i in range(self.n_estimators):
            y_pred += self.alphas[i] * self.trees[i].predict(X)
        return self.logistic_function(y_pred)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = -1.0
        return y_pred

