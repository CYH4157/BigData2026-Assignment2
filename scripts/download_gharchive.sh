#!/usr/bin/env bash
# Download GH Archive hour files from START up to END (UTC, default: 2 hours ago) into gharchive/.
# Already complete files are skipped (wget -c), so the script can be rerun to extend the range.
set -u
START=${START:-2025-12-01}
END=${END:-$(date -u -d "2 hours ago" +%Y-%m-%dT%H)}
OUT=${OUT:-gharchive}
JOBS=${JOBS:-24}
mkdir -p "$OUT"
python3 - "$START" "$END" > "$OUT/urls.txt" <<PY
import sys
from datetime import datetime, timedelta
t = datetime.fromisoformat(sys.argv[1]); end = datetime.fromisoformat(sys.argv[2])
while t <= end:
    print(f"https://data.gharchive.org/{t:%Y-%m-%d}-{t.hour}.json.gz")
    t += timedelta(hours=1)
PY
echo "$(wc -l < "$OUT/urls.txt") hour files, $START .. $END"
xargs -P "$JOBS" -n 1 wget -q -c -P "$OUT" < "$OUT/urls.txt"
missing=0
while read -r u; do [ -s "$OUT/$(basename "$u")" ] || { echo "missing: $u"; missing=$((missing+1)); }; done < "$OUT/urls.txt"
echo "done, $missing missing"
