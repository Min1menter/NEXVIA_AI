.PHONY: install data features eda train graph blend pipeline replay api dashboard drift test up down feast-apply materialize parity produce
PY=python
install:
	pip install -r requirements-train.txt -r requirements-dashboard.txt && pip install -e . --no-deps
data:      ; $(PY) -m fraudshield.data
features:  ; $(PY) -m fraudshield.features
eda:       ; $(PY) scripts/eda.py
train:     ; $(PY) -m fraudshield.train_xgb
graph:     ; $(PY) -m fraudshield.graph
blend:     ; $(PY) -m fraudshield.blend
# No-infra pipeline (memory feature store). For Feast: FEATURE_BACKEND=feast make pipeline-feast
pipeline: data features eda train graph blend
pipeline-feast:
	FEATURE_BACKEND=feast sh scripts/run_pipeline.sh
feast-apply: ; cd feature_repo && feast apply
materialize: ; $(PY) -m fraudshield.store materialize
parity:    ; $(PY) scripts/feast_parity.py
replay:    ; $(PY) -m fraudshield.replay_local --limit 20000      # Kafka-free online replay + parity check
api:       ; uvicorn fraudshield.api:create_default --factory --port 8000
dashboard: ; streamlit run dashboard/app.py
produce:   ; $(PY) -m fraudshield.producer --speed 3600 --limit 20000
drift:     ; $(PY) -m fraudshield.monitoring --retrain-if-needed
test:      ; pytest -q
up:        ; docker compose up -d --build
down:      ; docker compose down
