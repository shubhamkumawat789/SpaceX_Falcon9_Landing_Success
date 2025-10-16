
import argparse
import joblib
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .data import load_config, load_training_dataframe
from .features import build_preprocessor

@dataclass
class TrainConfig:
    random_seed: int
    data_train_csv: str
    target: str
    model_type: str
    model_params: dict
    test_size: float
    stratify: bool
    min_roc_auc: float
    min_f1: float
    model_path: str

def parse_config(path: str) -> TrainConfig:
    cfg = load_config(path)
    return TrainConfig(
        random_seed=cfg['random_seed'],
        data_train_csv=cfg['data']['train_csv'],
        target=cfg['data']['target'],
        model_type=cfg['model']['type'],
        model_params=cfg['model']['params'] or {},
        test_size=cfg['train']['test_size'],
        stratify=bool(cfg['train'].get('stratify', True)),
        min_roc_auc=cfg['metrics_thresholds']['min_roc_auc'],
        min_f1=cfg['metrics_thresholds']['min_f1'],
        model_path=cfg['artifacts']['model_path'],
    )

def build_model(model_type: str, params: dict):
    if model_type == "LogisticRegression":
        return LogisticRegression(max_iter=200, **params)
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main(config_path: str):
    cfg = parse_config(config_path)
    X, y = load_training_dataframe(cfg.data_train_csv, cfg.target)

    pre, cat_cols, num_cols = build_preprocessor(X)
    model = build_model(cfg.model_type, cfg.model_params)

    pipe = Pipeline([('pre', pre), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size,
        stratify=y if cfg.stratify else None,
        random_state=cfg.random_seed
    )

    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, prob)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC-AUC: {roc:.3f}  |  F1: {f1:.3f}")

    if roc < cfg.min_roc_auc or f1 < cfg.min_f1:
        raise SystemExit("Metrics below thresholds. Tune features/model or thresholds and try again.")

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, cfg.model_path)
    print(f"Saved model to {cfg.model_path}")

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
