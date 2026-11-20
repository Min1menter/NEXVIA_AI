import numpy as np
import pandas as pd
from fraudshield.features import FEATURE_COLUMNS
from fraudshield.monitoring import decide_retrain, drift_report, psi

CFG = dict(psi_threshold=0.2, share_drifted_threshold=0.3, min_reviewed=30, min_alert_precision=0.25)


def _frame(rng, shift=0.0, n=3000):
    return pd.DataFrame({c: rng.normal(shift, 1, n) for c in FEATURE_COLUMNS})


def test_psi_same_vs_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.5


def test_report_and_trigger():
    rng = np.random.default_rng(0)
    ref = _frame(rng)
    ok = drift_report(ref, _frame(rng), CFG)
    bad = drift_report(ref, _frame(rng, shift=2.0), CFG)
    assert not ok["dataset_drift"] and bad["dataset_drift"]
    stats = dict(reviewed=0, alert_precision=None)
    assert decide_retrain(ok, stats, CFG)[0] is False
    assert decide_retrain(bad, stats, CFG)[0] is True
    low_prec = dict(reviewed=100, alert_precision=0.1)
    assert decide_retrain(ok, low_prec, CFG)[0] is True


def test_drift_demo_generator_triggers_retrain():
    from fraudshield.drift_demo import make_drifted
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({c: rng.random(4000) for c in FEATURE_COLUMNS})
    ref["device_changed"] = (rng.random(4000) < 0.1).astype(float)
    ref["geo_changed"] = (rng.random(4000) < 0.05).astype(float)
    rep = drift_report(ref, make_drifted(ref, np.random.default_rng(1), 3000), CFG)
    assert rep["dataset_drift"] and decide_retrain(rep, dict(reviewed=0, alert_precision=None), CFG)[0]
