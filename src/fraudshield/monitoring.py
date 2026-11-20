"""Drift monitoring (PSI + KS, plus an Evidently HTML report) and the automatic retraining trigger."""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from fraudshield.config import DATA_DIR, ROOT, load_params
from fraudshield.features import FEATURE_COLUMNS

from fraudshield.config import REPORTS_DIR

REPORT_DIR = REPORTS_DIR


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Discrete features get one bin per value; continuous use reference quantiles."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    uniq = np.unique(ref)
    if len(uniq) == 1:
        edges = np.array([-np.inf, uniq[0] - 1e-9, uniq[0] + 1e-9, np.inf])
    elif len(uniq) <= bins:
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> dict:
    feats = {}
    for col in FEATURE_COLUMNS:
        if col not in ref or col not in cur:
            continue
        a, b = ref[col].astype(float).to_numpy(), cur[col].astype(float).to_numpy()
        v = psi(a, b)
        p = float(ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue) if (~np.isnan(a)).any() and (~np.isnan(b)).any() else 1.0
        feats[col] = dict(psi=v, ks_pvalue=p, drifted=bool(v >= cfg["psi_threshold"]))
    share = float(np.mean([f["drifted"] for f in feats.values()])) if feats else 0.0
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), n_reference=len(ref), n_current=len(cur),
                features=feats, share_drifted=share, dataset_drift=share >= cfg["share_drifted_threshold"])


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path) -> bool:
    """Best-effort Evidently HTML report (API differs across versions); decision logic never depends on it."""
    try:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref[FEATURE_COLUMNS], current_data=cur[FEATURE_COLUMNS])
            rep.save_html(str(path))
        except ImportError:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            snap = Report([DataDriftPreset()]).run(cur[FEATURE_COLUMNS], ref[FEATURE_COLUMNS])
            snap.save_html(str(path))
        return True
    except Exception as e:  # noqa
        print("evidently report skipped:", e)
        return False


def decide_retrain(report: dict, st: dict, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if report["dataset_drift"]:
        reasons.append(f"data drift: {report['share_drifted']:.0%} of features drifted (PSI>={cfg['psi_threshold']})")
    if st["reviewed"] >= cfg["min_reviewed"] and st["alert_precision"] is not None and st["alert_precision"] < cfg["min_alert_precision"]:
        reasons.append(f"alert precision {st['alert_precision']:.0%} < {cfg['min_alert_precision']:.0%} over {st['reviewed']} reviewed alerts")
    return bool(reasons), reasons


def run_check(retrain_if_needed: bool = False) -> dict:
    from fraudshield import db as dbm
    cfg = load_params()["drift"]
    sf = dbm.make_session_factory()
    ref = pd.read_parquet(DATA_DIR / "reference_features.parquet")
    with sf() as db:
        cur, st = dbm.recent_features(db, cfg["window_rows"]), dbm.stats(db)
    REPORT_DIR.mkdir(exist_ok=True)
    if len(cur) < cfg["min_rows"]:
        out = dict(status="insufficient_data", n_current=len(cur), min_rows=cfg["min_rows"])
        json.dump(out, open(REPORT_DIR / "drift_latest.json", "w"), indent=2)
        return out
    rep = drift_report(ref, cur, cfg)
    rep["evidently_html"] = evidently_html(ref, cur, REPORT_DIR / "drift_report.html")
    trig, reasons = decide_retrain(rep, st, cfg)
    rep.update(retrain_triggered=trig, retrain_reasons=reasons, status="ok")
    json.dump(rep, open(REPORT_DIR / "drift_latest.json", "w"), indent=2)
    if trig and retrain_if_needed:
        from fraudshield.retrain import retrain
        rep["retrain_result"] = retrain(reasons)
    return rep


if __name__ == "__main__":
    out = run_check(retrain_if_needed="--retrain-if-needed" in sys.argv)
    print(json.dumps({k: v for k, v in out.items() if k != "features"}, indent=2, default=str))
