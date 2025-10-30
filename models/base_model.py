from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, learning_rate=0.01, n_iters=1000, bias=0.0, mse_threshold=None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.bias = bias
        self.mse_threshold = mse_threshold
        self.weights = None

    @abstractmethod
    def fit(self, X, y):
        pass
    @abstractmethod
    def predict(self, X):
        pass
