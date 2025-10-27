import numpy as np


def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

# Single class Preceptron Algorithm

class PerceptronAlgorithm:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.random.rand(1,size=(n_features)) # not sure about this one??
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        # weight tuning
        for _ in range(self.n_iters): # I limited the number of iterations to avoid infinite loops this is not the theory though.
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # update parameters based on the update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
