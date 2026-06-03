"""Download WaPOR L3 target rasters (20 m) for the selected site and date range."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (
    abspath,
    dekads_between,
    ensure_dir,
    gcs_download,
    load_config,
    parse_iso,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--site-code", default=None,
                    help="Override site.l3_site_code from config (e.g. LDA).")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD; defaults to config")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD; defaults to config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    site_code = args.site_code or cfg["site"].get("l3_site_code")
    if not site_code:
        sys.exit("ERROR: site.l3_site_code is not set. Run 00_probe_l3_aoi.py first.")

    start = parse_iso(args.start or cfg["time_range"]["start"])
    end = parse_iso(args.end or cfg["time_range"]["end"])
    bucket = cfg["wapor"]["bucket"]
    prefix = cfg["wapor"]["l3_data_prefix"]
    mapset = cfg["wapor"]["l3_target_mapset"]
    out_dir = ensure_dir(abspath(cfg["paths"]["wapor_l3_dir"]))

    print(f"Site={site_code}  mapset={mapset}  {start} -> {end}")
    n_ok = n_err = n_skip = 0
    for code, _, _ in dekads_between(start, end):
        fname = f"WAPOR-3.{mapset}.{site_code}.{code}.tif"
        rel = f"{prefix}/{fname}"
        dst = out_dir / fname
        if args.dry_run:
            print(f"[dry] {rel} -> {dst}")
            continue
        try:
            gcs_download(bucket, rel, dst)
            n_ok += 1
            print(f"OK    {fname}  ({dst.stat().st_size/1024:.0f} KB)")
        except Exception as e:
            msg = str(e)
            if "404" in msg:
                n_skip += 1
                print(f"MISS  {fname}  (not in bucket)")
            else:
                n_err += 1
                print(f"ERR   {fname}: {e}")

    print(f"\nDone. ok={n_ok}  miss={n_skip}  err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
