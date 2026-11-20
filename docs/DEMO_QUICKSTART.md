# Demo quickstart (Windows PowerShell, ~10 min) - uses the models you already trained on Kaggle
Why: the full `pipeline` image (torch + PyG + Feast + MLflow + Evidently) can take over an hour to resolve/download. The demo does not need it:
training already happened on Kaggle, serving only needs XGBoost + FastAPI + Kafka client.

## 0. Stop anything stuck
Press `Ctrl+C` in the stuck terminal, then `docker compose down`.

## 1. Put the Kaggle outputs in `./demo` (IEEE-CIS run; use `run_paysim` for PaySim)
```powershell
cd <your fraudshield folder>            # the one containing docker-compose.demo.yml
Expand-Archive "C:\path\to\results (4).zip" -DestinationPath .\kaggle_out -Force      # file name may differ
mkdir demo, demo\reports -Force
Copy-Item .\kaggle_out\run_ieee\data      .\demo\data      -Recurse -Force
Copy-Item .\kaggle_out\run_ieee\artifacts .\demo\artifacts -Recurse -Force
```
(If the zip unpacks into another folder level, find `run_ieee` with `dir .\kaggle_out -Recurse -Directory -Filter run_ieee`.)

## 2. Start the stack
```powershell
docker compose -f docker-compose.demo.yml up -d --build
docker compose -f docker-compose.demo.yml ps          # api should be "healthy"; consumer, kafka, postgres, dashboard "running"
```
## 3. Stream transactions through Kafka
```powershell
docker compose -f docker-compose.demo.yml --profile replay run --rm producer
```
Open **http://localhost:8501** (dashboard) and **http://localhost:8000/docs** (API). Alerts appear within seconds.

## 4. Demo script (what to record)
1. Dashboard -> Alert queue: select the top alert, show scores and "why flagged", click **Confirm fraud**.
2. KPIs tab: scored / alerts / reviewed / precision.
3. Drift: `docker compose -f docker-compose.demo.yml run --rm api python -m fraudshield.drift_demo` -> dashboard "Model health" tab. (Needs >=500 scored txns from step 3.)
4. API docs page: `POST /score` with a sample transaction -> returns score in milliseconds.

## Troubleshooting
`docker compose -f docker-compose.demo.yml logs --tail 50 consumer api` and send me the output. Common: wrong folder names under `demo\` (must be `demo\data` and `demo\artifacts`).
Notes: this path uses the in-memory feature store (Feast was verified separately on Kaggle) and trains nothing; retraining (`--retrain`) needs the full image and is not part of this demo.
