# Demo runbook (for the recording)
## A. Already done on Kaggle (evidence)
`fraudshield_kaggle_runbook.ipynb` -> pipeline, tests, Feast parity, Evidently, ablation. Save screenshots of: test results, blend printout (stack chosen), ablation table + CIs, Feast parity `identical: True`, MLflow runs table, drift report HTML.

## B. Local stack. USE THE LIGHT DEMO: see docs/DEMO_QUICKSTART.md (docker-compose.demo.yml). The full-pipeline compose below needs a very long image build and is optional.
```bash
cp .env.example .env
docker compose --profile pipeline run --rm pipeline            # data, features, Feast apply, train, materialize (~10 min)
docker compose up -d --build
docker compose --profile replay run --rm producer              # streams 20,000 test txns through Kafka
# open http://localhost:8501 (dashboard)  http://localhost:8000/docs (API)  http://localhost:5000 (MLflow)
docker compose --profile pipeline run --rm pipeline python -m fraudshield.drift_demo          # drift report -> shows in dashboard "Model health"
docker compose --profile pipeline run --rm pipeline python -m fraudshield.drift_demo --retrain # optional: full retrain
```
If something fails: `docker compose logs -f consumer api`. No Docker? Run without Kafka: `make pipeline && make replay && make api` and `make dashboard` in separate terminals.

## C. Minikube (optional): `k8s/README.md`.
## D. Recording order (about 4 min of live demo): alerts appear -> open case -> reasons -> confirm fraud -> KPI tab -> drift demo -> Model health tab.
Record the screen with OBS / Windows Game Bar / QuickTime. Keep a fallback: screenshots from Section A.
