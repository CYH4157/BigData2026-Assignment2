#!/usr/bin/env bash
# Build the 8 tables from GH Archive with the course preprocessing script.
#   END=2026-11-21T12 bash scripts/build_data.sh
# 1) downloads the GH Archive hour files 2025-12-01 00:00 .. END (UTC) into gharchive/
# 2) runs assignment2/scripts/preprocess.py (unmodified) -> data/db/*.parquet
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
[[ -d assignment2 ]] || git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
START=${START:-2025-12-01} END=${END:-} bash scripts/download_gharchive.sh
rm -rf data
$PY assignment2/scripts/preprocess.py --raw gharchive --out data
