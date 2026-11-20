import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Mirror the Docker layout: `src.*` from the repo root, `common.*` / `api.*` from services/.
for p in (ROOT, ROOT / "services"):
    sys.path.insert(0, str(p))
