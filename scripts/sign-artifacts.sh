#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <artifact1> [artifact2 ...]"
  exit 1
fi

if [[ -z "${UNBG_SIGNING_KEY_B64:-}" ]]; then
  echo "UNBG_SIGNING_KEY_B64 is required for signing"
  exit 1
fi

TMP_KEY="$(mktemp)"
if ! python - "$TMP_KEY" <<'PY'
import base64
import os
import re
import sys

out_path = sys.argv[1]
value = os.environ.get("UNBG_SIGNING_KEY_B64", "")
if not value:
    raise SystemExit("UNBG_SIGNING_KEY_B64 is required for signing")

text = value.strip()
if "BEGIN " in text and "PRIVATE KEY" in text:
    payload = text.replace("\r\n", "\n").encode("utf-8")
else:
    compact = re.sub(r"\s+", "", text)
    if len(compact) % 4:
        compact += "=" * (4 - (len(compact) % 4))
    payload = base64.b64decode(compact, validate=False)

with open(out_path, "wb") as handle:
    handle.write(payload)
PY
then
  echo "Warning: UNBG_SIGNING_KEY_B64 could not be decoded. Skipping signing."
  rm -f "$TMP_KEY"
  exit 0
fi

if ! openssl pkey -in "$TMP_KEY" -noout >/dev/null 2>&1; then
  echo "Warning: UNBG_SIGNING_KEY_B64 does not contain a valid private key. Skipping signing."
  rm -f "$TMP_KEY"
  exit 0
fi
chmod 600 "$TMP_KEY"

for artifact in "$@"; do
  if [[ ! -f "$artifact" ]]; then
    echo "Skipping missing artifact: $artifact"
    continue
  fi
  openssl dgst -sha256 -sign "$TMP_KEY" -out "${artifact}.sig" "$artifact"
  echo "Signed $artifact -> ${artifact}.sig"
done

rm -f "$TMP_KEY"
