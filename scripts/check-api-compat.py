#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "api-snapshots" / "api-compat-v1.json"
CORE = ROOT / "crates" / "unbg-core" / "src" / "lib.rs"
TAURI_TS = ROOT / "integrations" / "tauri-plugin-unbg" / "src" / "index.ts"


def extract_fields(body: str, struct_name: str):
    pat = rf"pub struct {struct_name}\s*\{{(?P<body>.*?)\n\s*\}}"
    m = re.search(pat, body, flags=re.S)
    if not m:
        raise RuntimeError(f"could not find struct {struct_name}")
    fields = re.findall(r"pub\s+([a-zA-Z0-9_]+)\s*:", m.group("body"))
    return fields


def main():
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    core = CORE.read_text(encoding="utf-8")
    ts = TAURI_TS.read_text(encoding="utf-8")

    req_fields = extract_fields(core, "RemoveBackgroundRequest")
    res_fields = extract_fields(core, "RemoveBackgroundResponse")

    m = re.search(r'removeBackground:\s*"([^"]+)"', ts)
    if not m:
        raise RuntimeError("could not find TAURI_UNBG_COMMANDS_V1.removeBackground")
    cmd = m.group(1)

    problems = []
    if cmd != snapshot["tauri_remove_command"]:
        problems.append(f"tauri command drift: {cmd} != {snapshot['tauri_remove_command']}")
    if req_fields != snapshot["request_fields"]:
        problems.append(f"request fields drift: {req_fields} != {snapshot['request_fields']}")
    if res_fields != snapshot["response_fields"]:
        problems.append(f"response fields drift: {res_fields} != {snapshot['response_fields']}")

    if problems:
        for p in problems:
            print(p)
        sys.exit(1)

    print("API compatibility check passed.")


if __name__ == "__main__":
    main()
