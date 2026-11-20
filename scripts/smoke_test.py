"""End-to-end smoke test against a RUNNING stack (read-only; it never approves or rejects anything).

    python scripts/smoke_test.py --env-file .env
    python scripts/smoke_test.py --base http://localhost:8080 --user radiologist --password ...

Checks: health, login, auth required, worklist, scan + overlay delivery for a few studies,
path-traversal rejection, and the audit hash chain. Exit code is non-zero if anything fails.
"""
import argparse
import os
import sys
from pathlib import Path

import requests

PNG = b"\x89PNG\r\n\x1a\n"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def load_env(path):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--min-studies", type=int, default=1)
    ap.add_argument("--sample", type=int, default=3, help="how many studies to fetch images for")
    args = ap.parse_args()

    env = load_env(args.env_file)
    user = args.user or os.environ.get("RADIOLOGIST_USER") or env.get("RADIOLOGIST_USER")
    pw = args.password or os.environ.get("RADIOLOGIST_PASSWORD") or env.get("RADIOLOGIST_PASSWORD")
    if not user or not pw:
        sys.exit("Need --user/--password or RADIOLOGIST_USER/RADIOLOGIST_PASSWORD (in the environment or --env-file).")
    api = args.base.rstrip("/") + "/api"
    s = requests.Session()
    s.timeout = 30

    r = s.get(f"{api}/health", timeout=30)
    check("API health", r.status_code == 200, f"HTTP {r.status_code}")

    r = s.post(f"{api}/auth/login", data={"username": user, "password": pw}, timeout=30)
    if not check("login as radiologist", r.status_code == 200, f"HTTP {r.status_code}"):
        sys.exit("Cannot continue without a token.")
    token = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    check("worklist requires auth", s.get(f"{api}/studies", timeout=30).status_code == 401)

    r = s.get(f"{api}/studies", headers=auth, timeout=30)
    studies = r.json() if r.status_code == 200 else []
    check("worklist loads", r.status_code == 200 and len(studies) >= args.min_studies, f"{len(studies)} studies")
    by_status = {}
    for st in studies:
        by_status[st["status"]] = by_status.get(st["status"], 0) + 1
    print("      status counts:", by_status)
    check("no studies in error state", by_status.get("error", 0) == 0, f"{by_status.get('error', 0)} in error")

    good = [st for st in studies if st["status"] != "error"][: args.sample]
    for st in good:
        uid = st["study_uid"]
        tag = uid[-8:]
        r = s.get(f"{api}/studies/{uid}", headers=auth, timeout=30)
        detail = r.json() if r.status_code == 200 else {}
        check(f"study {tag}: detail", r.status_code == 200 and "predictions" in detail)
        r = s.get(f"{api}/studies/{uid}/image", headers=auth, timeout=60)
        check(f"study {tag}: scan is a PNG", r.status_code == 200 and r.content[:8] == PNG, f"HTTP {r.status_code}")
        labels = detail.get("overlays") or []
        if labels:
            r = s.get(f"{api}/studies/{uid}/overlay/{requests.utils.quote(labels[0], safe='')}", headers=auth, timeout=60)
            check(f"study {tag}: overlay '{labels[0]}' is a PNG", r.status_code == 200 and r.content[:8] == PNG)
        else:
            check(f"study {tag}: has an overlay", False, "no overlays listed")

    # These stay under /api/ so they really reach the API.
    r = s.get(f"{api}/studies/not-a-real-study/image", headers=auth, timeout=30)
    check("unknown study id returns 404", r.status_code == 404, f"HTTP {r.status_code}")
    if good:
        r = s.get(f"{api}/studies/{good[0]['study_uid']}/overlay/not-a-label", headers=auth, timeout=30)
        check("unknown overlay label returns 404", r.status_code == 404, f"HTTP {r.status_code}")
    # Traversal probes. nginx may normalise some of these away and answer with the SPA's index.html (HTTP 200),
    # so the meaningful check is that no file contents or images come back.
    leaked = False
    for probe in ("/studies/..%2f..%2fetc%2fpasswd/overlay/x",
                  "/studies/%2e%2e%2f%2e%2e%2fetc%2fpasswd/image",
                  "/studies/x/overlay/..%2f..%2fetc%2fpasswd"):
        r = s.get(api + probe, headers=auth, timeout=30)
        leaked = leaked or r.content[:8] == PNG or b"root:" in r.content
    check("path-traversal probes leak no files", not leaked)

    r = s.get(f"{api}/audit/verify", headers=auth, timeout=30)
    ok = r.status_code == 200 and r.json().get("ok") is True
    check("audit hash chain verifies", ok, f"checked {r.json().get('checked')}" if r.status_code == 200 else f"HTTP {r.status_code}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        sys.exit("Cannot reach the stack at --base (default http://localhost:8080). Is it running?\n"
                 "  Check:  docker compose ps\n  Start:  docker compose up -d")
