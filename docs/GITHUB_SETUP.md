# Push to GitHub with a clean history
```bash
cd fraudshield && git init -b main
git add .gitignore pyproject.toml requirements*.txt params.yaml && git commit -m "chore: project scaffold and configuration"
git add src/fraudshield/{synthetic,data,config}.py && git commit -m "feat(data): synthetic PaySim-style generator with fraud rings, IEEE-CIS/PaySim loaders"
git add src/fraudshield/features.py tests/test_features.py tests/test_synthetic.py && git commit -m "feat(features): single-path stateful features with train/serve parity tests"
git add feature_repo src/fraudshield/store.py scripts/feast_parity.py && git commit -m "feat(feast): feature views, push source, memory/Feast stores, parity check"
git add src/fraudshield/train_xgb.py scripts/eda.py && git commit -m "feat(model): XGBoost with class weights/SMOTE, PR-AUC evaluation, MLflow tracking"
git add src/fraudshield/graph.py tests/test_graph.py && git commit -m "feat(graph): account-device-card graph and GraphSAGE ring detector"
git add src/fraudshield/{blend,ablation}.py tests/test_blend.py tests/test_stack.py && git commit -m "feat(blend): validation-selected stacking combiner, ablation with bootstrap CIs"
git add src/fraudshield/{model_bundle,scoring,db,api,consumer,producer,replay_local}.py tests/test_api.py && git commit -m "feat(serving): Kafka consumer/producer, FastAPI scoring and case management"
git add src/fraudshield/{monitoring,retrain,drift_demo}.py tests/test_drift.py && git commit -m "feat(monitoring): PSI/KS drift, Evidently report, retraining triggers, drift demo"
git add dashboard Dockerfile docker-compose.yml k8s scripts Makefile .env.example .github && git commit -m "feat(deploy): Streamlit console, Docker Compose, Minikube manifests, CI"
git add README.md docs && git commit -m "docs: architecture, API, results, model card, presentation and runbooks"
git remote add origin <your-repo-url> && git push -u origin main
```
Do not commit `data/`, `artifacts/`, `mlflow.db`, `.env`. Add your Kaggle result screenshots under `docs/img/`.
