from models.base_model import BaseModel
import numpy as np

class Adaline(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) * 0.001  # small random weights

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            errors = y - y_pred

            self.weights += self.lr * X.T.dot(errors)
            self.bias += self.lr * errors.sum()

            mse = (errors ** 2).mean() / 2

            if mse < self.mse_threshold:
                break

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return np.where(y_pred >= 0.0, 1, -1)
