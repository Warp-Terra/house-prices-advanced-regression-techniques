import os
from typing import Tuple
import numpy as np
import pandas as pd


TARGET_COL = "SalePrice"
ID_COL = "Id"


def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train/test CSVs from the Kaggle dataset directory."""
    # TODO: build train/test paths from data_dir (e.g., train.csv/test.csv)
    train_path = os.path.join(data_dir, "data", "train.csv")
    test_path = os.path.join(data_dir, "data", "test.csv")
    # TODO: read CSVs with pandas
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # TODO: return (train_df, test_df)
    return train_df, test_df


def split_features_target(
    train_df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split train dataframe into features X and target y."""
    # TODO: set y = train_df[target_col]
    y = train_df[target_col]
    # TODO: set X = train_df.drop(columns=[target_col])
    X = train_df.drop(columns=[target_col])
    # TODO: return (X, y)
    return X, y


def simple_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Simple missing value strategy for a baseline model."""
    # 获取数值列和分类列
    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    
    # 数值列用中位数填充
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # 分类列用 "None" 填充
    df[cat_cols] = df[cat_cols].fillna("None")
    
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    # 对所有 object 类型的列进行 One-Hot 编码
    df = pd.get_dummies(df, dtype=float)
    return df


def log_transform_target(y: pd.Series) -> pd.Series:
    """Apply log1p to target for a baseline linear model."""
    # TODO: return np.log1p(y)
    return np.log1p(y)


def preprocess_for_linear_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Minimal preprocessing pipeline for a linear baseline."""
    # TODO: drop ID_COL from train/test if present
    train_df = train_df.drop(columns=[ID_COL])
    test_df = test_df.drop(columns=[ID_COL])
    # TODO: split X/y from train_df
    X, y = split_features_target(train_df)
    # TODO: log-transform y
    y_log = log_transform_target(y)
    # TODO: combine train/test for consistent encoding
    combined_df = pd.concat([X, test_df], ignore_index=True)
    # TODO: fill missing values
    combined_df = simple_fill_missing(combined_df)
    # TODO: one-hot encode
    combined_df = one_hot_encode(combined_df)
    
    # 特征标准化（把所有特征缩放到均值0、标准差1）
    mean = combined_df.mean()
    std = combined_df.std()
    std[std == 0] = 1  # 避免除以0
    combined_df = (combined_df - mean) / std
    
    # TODO: split combined data back into X_train/X_test
    X_train = combined_df.iloc[:len(X)]
    X_test = combined_df.iloc[len(X):]
    # TODO: return (X_train, X_test, y_log)
    return X_train, X_test, y_log

