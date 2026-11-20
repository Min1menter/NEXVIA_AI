#!/usr/bin/env sh
# End-to-end batch pipeline: data -> features -> Feast -> models -> blend -> online store
set -e
python -m fraudshield.data
python -m fraudshield.features
python scripts/eda.py
if [ "$FEATURE_BACKEND" = "feast" ]; then (cd "${FEAST_REPO:-feature_repo}" && feast apply); fi
python -m fraudshield.train_xgb
python -m fraudshield.graph
python -m fraudshield.blend
if [ "$FEATURE_BACKEND" = "feast" ]; then python -m fraudshield.store materialize; python scripts/feast_parity.py; fi
echo "pipeline complete"
