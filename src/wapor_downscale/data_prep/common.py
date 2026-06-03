"""Shared helpers for the Baixo data-prep pipeline."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(name: str = "baixo") -> dict:
    cfg_path = REPO_ROOT / "configs" / f"{name}.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abspath(rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- dekad helpers ----------------------------------------------------------
# WaPOR dekads: D1 = days 1..10, D2 = 11..20, D3 = 21..end-of-month.

def dekad_start(d: date) -> date:
    if d.day <= 10:
        return d.replace(day=1)
    if d.day <= 20:
        return d.replace(day=11)
    return d.replace(day=21)


def dekad_code(d: date) -> str:
    if d.day <= 10:
        idx = 1
    elif d.day <= 20:
        idx = 2
    else:
        idx = 3
    return f"{d.year:04d}-{d.month:02d}-D{idx}"


def dekads_between(start: date, end: date):
    """Yield (dekad_code, dekad_start_date, dekad_end_date) inclusive."""
    cur = dekad_start(start)
    while cur <= end:
        if cur.day == 1:
            nxt = cur.replace(day=11)
        elif cur.day == 11:
            nxt = cur.replace(day=21)
        else:
            # D3 -> first of next month
            if cur.month == 12:
                nxt = date(cur.year + 1, 1, 1)
            else:
                nxt = date(cur.year, cur.month + 1, 1)
        yield dekad_code(cur), cur, nxt - timedelta(days=1)
        cur = nxt


def parse_iso(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


# --- GCS helpers (anonymous, public bucket) --------------------------------

import requests


def gcs_list(bucket: str, prefix: str, delimiter: str | None = "/") -> dict:
    base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    params = {"prefix": prefix, "maxResults": 1000}
    if delimiter:
        params["delimiter"] = delimiter
    items, prefixes = [], []
    page_token = None
    while True:
        p = dict(params)
        if page_token:
            p["pageToken"] = page_token
        r = requests.get(base, params=p, timeout=60)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        prefixes.extend(data.get("prefixes", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return {"items": items, "prefixes": prefixes}


def gcs_public_url(bucket: str, object_name: str) -> str:
    return f"https://storage.googleapis.com/{bucket}/{object_name}"


def gcs_download(bucket: str, object_name: str, dst: Path, overwrite: bool = False) -> Path:
    dst = Path(dst)
    if dst.exists() and not overwrite and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = gcs_public_url(bucket, object_name)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)
    return dst


# --- regex helpers ----------------------------------------------------------

WAPOR_L3_TIF_RE = re.compile(
    r"WAPOR-3\.L3-[A-Z]+-D\.(?P<site>[A-Z0-9]+)\.(?P<year>\d{4})-(?P<month>\d{2})-D(?P<dekad>[123])\.tif$"
)
WAPOR_L2_TIF_RE = re.compile(
    r"WAPOR-3\.L2-[A-Z]+-D\.(?P<year>\d{4})-(?P<month>\d{2})-D(?P<dekad>[123])\.tif$"
)
