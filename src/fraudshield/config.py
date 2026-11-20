from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ART_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
FEATURE_REPO = os.getenv("FEAST_REPO", str(ROOT / "feature_repo"))


REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_params() -> dict:
    """params.yaml, deep-merged with the file named by $FRAUD_PARAMS (e.g. params.ieee.yaml) if set."""
    base = yaml.safe_load((ROOT / "params.yaml").read_text())
    over = os.getenv("FRAUD_PARAMS")
    return _merge(base, yaml.safe_load(Path(over).read_text())) if over else base


def mlflow_setup(experiment: str | None = None):
    import mlflow
    experiment = experiment or os.getenv("FRAUD_EXPERIMENT", "fraudshield")
    # SQLite backend: newer MLflow versions reject the old file:./mlruns store
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(experiment)
    return mlflow
