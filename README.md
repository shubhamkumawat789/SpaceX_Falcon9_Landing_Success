
# SpaceX Falcon 9 Landing — Beginner-Friendly Production Template

A clean template to turn your Jupyter notebook into a real-world project.  
You get: reproducible training, a saved model, a simple REST API for predictions, and tests.

## 1) Quick start

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Put your training CSV here
#    data/spacex_training.csv  (see schema below)

# 4) Train (saves model to models/model.joblib)
python -m src.spacex_landing.train --config configs/config.yaml

# 5) Try local predictions
python -m src.spacex_landing.inference --input_json examples/sample_request.json

# 6) Run the API
uvicorn src.spacex_landing.serving.api:app --reload --port 8000

# 7) Call the API
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json"   -d @examples/sample_request.json
```

## 2) Data schema (CSV)

Save your training data as `data/spacex_training.csv` with **one row per launch** and these example columns:

- **landing_success** (target, 0/1)
- payload_mass_kg (float)
- flight_number (int)
- launch_site (str)
- orbit (str)
- booster_version (str, optional)
- reused_count (int)
- cores (int)
- gridfins (0/1)
- legs (0/1)
- block (int, optional)
- launch_year (int)
- launch_month (int)

> You can add more columns if you have them (weather, coordinates, etc.). The pipeline will automatically one-hot encode text columns and pass numeric columns through.

## 3) Project layout

```
spacex-landing-starter/
├─ configs/config.yaml
├─ data/                       # your CSV goes here (gitignored)
├─ examples/sample_request.json
├─ src/spacex_landing/
│  ├─ data.py                  # load CSV, basic checks
│  ├─ features.py              # feature preprocessing
│  ├─ train.py                 # train & save model
│  ├─ inference.py             # load & predict
│  └─ serving/
│     ├─ schemas.py            # pydantic request/response
│     └─ api.py                # FastAPI app
├─ tests/test_smoke.py         # very simple tests
└─ README.md
```

## 4) Config-driven training

Edit `configs/config.yaml` to point at your data, choose the model, and tweak parameters.

## 5) Notes for beginners

- Start with the defaults. Get a green end-to-end run first.
- Then iterate: feature selection, model choice (LogisticRegression, RandomForest, XGBoost if you add it), and thresholds.
- Focus on **ROC-AUC** and **F1** (class imbalance is common).
- Keep the notebook for **EDA only**; production code lives in `src/`.

## 6) Next steps (optional, when you're ready)

- Dockerize and deploy (Cloud Run/Render/Fly).
- Add experiment tracking (MLflow).
- Schedule batch scoring with cron/Prefect.
- Add drift/performance monitoring.
