import numpy as np
from algorithm import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from config import Config
from itertools import combinations

def unit_step_func(x):
    return np.where(x > 0 , 1, 0)


def signum(x):
    return np.where(x >= 0, 1, -1)

def merge_classes(df, target_col, combo):
    merged_name = "_".join(combo)
    df_copy = df.copy()
    
    df_copy[target_col] = df_copy[target_col].replace({combo[0]: merged_name, combo[1]: merged_name})
    
    df_copy[target_col] = df_copy[target_col].astype('category').cat.codes
    
    return df_copy



class Preceptron(BaseModel):

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = signum
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.random.rand(n_features) * 0.01 # to make them small
        self.bias = 0.0

        y_ = np.where(y > 0 , 1, 0)

        # weight tuning
        for _ in range(self.n_iters): 
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



class Adaline(BaseModel):
    def __init__(self, learning_rate=0.01, n_iters=2000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y):
        # Add bias
        X_bias = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X_bias.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iters):
            y_pred = X_bias.dot(self.weights)
            errors = y - y_pred
            self.weights += self.learning_rate * X_bias.T.dot(errors)

    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        y_pred = X_bias.dot(self.weights)
        return np.where(y_pred >= 0, 1, 0)  # discrete labels



if __name__ == "__main__":
    # Test the Preceptron implementation

    # Load dataset
    config = Config()
    df = pd.read_csv(config.DATASET_PATH)
    target_column = config.TARGET_COLUMN
    all_features = df.drop(target_column, axis=1).columns.tolist()
    features_combos = list(combinations(all_features, 2))

    for combo in features_combos:
        print("Using features combination:", combo)

        feature_cols = [f.strip() for f in combo]
        df_combo = df[feature_cols + [target_column]].copy()

        df_combo.fillna(df_combo.median(numeric_only=True), inplace=True)

        for col in config.CATEGORICAL_COLUMNS:
            if col in df_combo.columns:
                df_combo[col].fillna(df_combo[col].mode()[0], inplace=True)
        
        for col in config.CATEGORICAL_COLUMNS:
            if col in df_combo.columns:
                df_combo[col] = df_combo[col].astype('category').cat.codes


        target_column = config.TARGET_COLUMN

        classes = df_combo[target_column].unique().tolist()
        merge_combos = list(combinations(classes, 2))
        i = 1

        
        for combo in merge_combos:    
            print(f"  Merging classes combination {i}")
            i += 1
            df_merged = merge_classes(df_combo, target_column, combo)

            
            X = df_merged.drop(target_column, axis=1).values
            y = df_merged[target_column].values

            test_size = config.TEST_SIZE

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            # === Standardization ===
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_std[X_std == 0] = 1  # avoid division by zero

            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std

            p = Adaline(learning_rate=config.LEARNING_RATE, n_iters=config.N_ITERS)
            p.fit(X_train, y_train)

            predictions = p.predict(X_test)

            acc = accuracy_score(y_test, predictions)
            print(f"Adaline classification accuracy: {acc * 100:.2f}%")    
