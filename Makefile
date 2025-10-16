
.PHONY: install train api test

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.spacex_landing.train --config configs/config.yaml

api:
	uvicorn src.spacex_landing.serving.api:app --reload --port 8000

test:
	pytest -q
