
from typing import Tuple
import pandas as pd
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_training_dataframe(csv_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    # Basic sanity
    if X.empty:
        raise ValueError("No feature columns found after dropping target.")
    return X, y
