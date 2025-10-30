from models.base_model import BaseModel
import numpy as np
from activation_fns import ACTIVATIONS, ActivationEnum


class Preceptron(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_func = ACTIVATIONS[ActivationEnum.SIGN]


    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.random.rand(n_features) * 0.001 # to make them small

        y_ = np.where(y > 0 , 1, 0)

        # weight tuning
        for _ in range(self.n_iters): 
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                if y_predicted == y_[idx]:
                    continue
                
                # update parameters based on the update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
