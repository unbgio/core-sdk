# API Compatibility Policy (`v1`)

This repository exposes public integration surfaces that are versioned under `v1`.

## Stable surfaces

- Rust DTOs in `unbg-core::v1` (`RemoveBackgroundRequest`, `RemoveBackgroundResponse`)
- Tauri TS command constant `TAURI_UNBG_COMMANDS_V1.removeBackground`
- Tauri TS request/response types exported from `integrations/tauri-plugin-unbg/src/index.ts`

## Breaking changes (not allowed in `v1`)

- Removing or renaming any existing `v1` field.
- Changing field meaning or type incompatibly.
- Changing the Tauri `v1` command id string.

## Allowed non-breaking changes

- Adding new optional fields.
- Internal implementation refactors that preserve the same external schema.
- New `v2` surfaces in parallel with `v1` (preferred path for breaking changes).

## CI enforcement

`scripts/check-api-compat.py` compares current public `v1` schema/command values against
`api-snapshots/api-compat-v1.json`. CI fails if a breaking drift is detected.
