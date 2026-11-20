"""Upload every .dcm under a folder to Orthanc (triggers the inference worker).

    python scripts/upload_dicom.py data/dicom --url http://localhost:8042 --user orthanc --password ...
"""
import argparse
import os
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--url", default="http://localhost:8042")
    ap.add_argument("--user", default=os.environ.get("ORTHANC_USER", "orthanc"))
    ap.add_argument("--password", default=os.environ.get("ORTHANC_PASSWORD"))
    args = ap.parse_args()
    if not args.password:
        ap.error("--password or ORTHANC_PASSWORD required")

    s = requests.Session()
    s.auth = (args.user, args.password)
    files = sorted(Path(args.folder).rglob("*.dcm"))
    for i, f in enumerate(files, 1):
        r = s.post(f"{args.url}/instances", data=f.read_bytes(),
                   headers={"Content-Type": "application/dicom"}, timeout=60)
        r.raise_for_status()
        if i % 50 == 0 or i == len(files):
            print(f"uploaded {i}/{len(files)}")


if __name__ == "__main__":
    main()
