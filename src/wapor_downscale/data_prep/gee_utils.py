"""Earth Engine helpers — service-account authentication and tile-based image downloads."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ee


def init_ee(cfg: dict) -> None:
    gee_cfg = cfg.get("gee", {})
    sa_json = gee_cfg.get("service_account_json")
    if not sa_json:
        # Fall back to interactive auth if available.
        try:
            ee.Initialize()
        except Exception:
            print(
                "GEE service_account_json is not set in config; either set it or run "
                "`earthengine authenticate` once in this env."
            )
            raise
        return

    sa_path = Path(sa_json).expanduser()
    if not sa_path.exists():
        raise FileNotFoundError(f"GEE service-account JSON not found: {sa_path}")

    sa_email = gee_cfg.get("service_account_email")
    if not sa_email:
        with sa_path.open("r", encoding="utf-8") as f:
            sa_email = json.load(f).get("client_email")
        if not sa_email:
            raise ValueError("Could not read client_email from service-account JSON.")

    credentials = ee.ServiceAccountCredentials(sa_email, str(sa_path))
    ee.Initialize(credentials)
    print(f"GEE initialized for {sa_email}")


def export_image_to_local(
    image: "ee.Image",
    region,
    scale: int,
    out_path: Path,
    crs: str = "EPSG:4326",
) -> Path:
    """Download an EE image to a local GeoTIFF via getDownloadURL.

    For AOIs larger than the EE export-size cap this will fail; callers should tile.
    """
    import requests

    url = image.getDownloadURL(
        {
            "scale": scale,
            "region": region,
            "crs": crs,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out_path
