
import argparse, json, joblib, pandas as pd

def load_model(path: str):
    return joblib.load(path)

def predict_one(model, payload: dict):
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[0,1]
    return float(proba), int(proba >= 0.5)

def cli(model_path: str, input_json: str):
    model = load_model(model_path)
    with open(input_json) as f:
        payload = json.load(f)
    p, label = predict_one(model, payload)
    print({"probability": p, "predicted_label": label})

if __name__ == "__main__":
    import yaml
    import sys
    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=cfg['artifacts']['model_path'])
    ap.add_argument("--input_json", required=True)
    args = ap.parse_args()
    cli(args.model_path, args.input_json)
