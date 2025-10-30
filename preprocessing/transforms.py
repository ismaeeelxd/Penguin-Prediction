import numpy as np
import pandas as pd
from typing import List

def fill_numeric_missing(df: pd.DataFrame) -> pd.DataFrame:
  medians = df.median(numeric_only=True)
  df_out = df.copy()
  df_out.fillna(medians, inplace=True)
  return df_out

def fill_categorical_missing(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
  df_out = df.copy()
  for col in categorical_cols:
      if col in df_out.columns:
        mode = df_out[col].mode()[0]
        df_out[col].fillna(mode, inplace=True)
  return df_out


def encode_categoricals(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
  df_out = df.copy()
  for col in categorical_cols:
      if col in df_out.columns:
        df_out[col] = df_out[col].astype('category').cat.codes
  return df_out



def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test  - mean) / std

    return X_train_scaled, X_test_scaled, mean, std

