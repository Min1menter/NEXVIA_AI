"""Loads the XGBoost booster + meta.json (feature order, blend alpha, threshold) and explains predictions."""
from __future__ import annotations
import json
import numpy as np
from fraudshield.config import ART_DIR


class ModelBundle:
    def __init__(self, art_dir=None):
        self.art_dir = art_dir or ART_DIR
        self.reload()

    def reload(self) -> None:
        import xgboost as xgb
        meta = json.load(open(self.art_dir / "meta.json"))
        booster = xgb.Booster()
        booster.load_model(str(self.art_dir / "xgb.json"))
        self.meta, self.booster = meta, booster
        self.cols, self.alpha, self.threshold = meta["feature_columns"], meta["alpha"], meta["threshold"]
        self.version = meta["model_version"]

    def combine(self, xgb_p: float, g_score, cluster) -> float:
        from fraudshield.blend import combine
        nan = float("nan")
        return float(combine(self.meta, [xgb_p], [nan if g_score is None else g_score], [nan if cluster is None else cluster])[0])

    def predict(self, feats: dict, top_k: int = 3):
        import xgboost as xgb
        x = np.array([[feats[c] for c in self.cols]], dtype=float)
        dm = xgb.DMatrix(x, feature_names=self.cols)
        prob = float(self.booster.predict(dm)[0])
        contrib = self.booster.predict(dm, pred_contribs=True)[0][:-1]       # per-feature margin contributions
        order = np.argsort(-contrib)[:top_k]
        reasons = [dict(feature=self.cols[i], value=None if np.isnan(x[0, i]) else float(x[0, i]), contribution=float(contrib[i]))
                   for i in order if contrib[i] > 0]
        return prob, reasons
