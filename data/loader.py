import pandas as pd

def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_feature_target(df, feature_cols, target_col):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
