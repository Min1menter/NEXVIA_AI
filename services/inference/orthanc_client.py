"""Thin Orthanc REST client."""
import requests


class Orthanc:
    def __init__(self, url, user, password, timeout=60):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.auth = (user, password)
        self.timeout = timeout

    def get(self, path, **params):
        r = self.s.get(f"{self.url}{path}", params=params or None, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def raw(self, path) -> bytes:
        r = self.s.get(f"{self.url}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def changes(self, since: int, limit: int = 100):
        return self.get("/changes", since=since, limit=limit)

    def upload(self, dicom_bytes: bytes):
        r = self.s.post(f"{self.url}/instances", data=dicom_bytes,
                        headers={"Content-Type": "application/dicom"}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
