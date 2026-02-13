#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output-manifest.json> <artifact1> [artifact2 ...]"
  exit 1
fi

OUT="$1"
shift

python - "$OUT" "$@" <<'PY'
import hashlib
import json
import os
import sys

out = sys.argv[1]
paths = sys.argv[2:]
items = []
for p in paths:
    if not os.path.exists(p):
        continue
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    items.append({
        "path": p.replace("\\", "/"),
        "sha256": h.hexdigest(),
        "size": size,
    })
data = {"artifacts": items}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY

echo "Wrote manifest: $OUT"
