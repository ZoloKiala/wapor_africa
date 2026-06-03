# Configs

One YAML file per site/AOI. Edit and rename to your site name.

## Example: minimal new config

Copy `example_baixo.yaml` to `<your_site>.yaml` and edit:

```yaml
site:
  name: NEWSITE                       # short code
  l3_site_code: KOG                   # WaPOR L3 pilot code, or null if no L3 exists
  target_bbox_hint: [lon_min, lat_min, lon_max, lat_max]   # WGS84

time_range:
  start: "2024-01-01"
  end:   "2024-12-31"

gee:
  service_account_json: /path/to/your-gee-service-account.json

paths:
  data_root: data/newsite
  wapor_l3_dir: data/newsite/wapor_l3
  wapor_l2_dir: data/newsite/wapor_l2
  s2_dir:       data/newsite/s2
  stacks_dir:   data/newsite/stacks/NEWSITE_STACK_S2_MATCH_L3_20M_FULL_1
```

The rest of the YAML (Sentinel-2 settings, WaPOR mapsets, CHIRPS, etc.) you can leave alone unless you want to tune them.

## Notes

- **Without L3 ground truth**: set `l3_site_code: null`. The pipeline will still produce predictions but you can't compute accuracy metrics.
- **Without a GEE service account**: the Sentinel-2 fetch step won't work. Either set up GEE (https://earthengine.google.com/), or pre-stage your own B4/B8/B11 stacks and skip step 3 of the inference notebook.
- **Bbox**: must be in WGS84 (EPSG:4326). The pipeline reprojects everything internally to the L3 UTM grid.
