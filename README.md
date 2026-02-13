# UNBG Rust SDK Workspace

Rust-first, local-only SDK workspace for background removal across CLI, Tauri, and native mobile.

## Start Here (Consumer Integrations)

If you are integrating this SDK into an app, use these guides first:

- Tauri consumer quickstart: `docs/TAURI_CONSUMER.md`
- Android consumer quickstart: `docs/ANDROID_CONSUMER.md`
- iOS consumer quickstart: `docs/IOS_CONSUMER.md`
- Common troubleshooting: `docs/TROUBLESHOOTING.md`

## Workspace crates

- `apps/unbg-cli`: CLI entrypoint with model management commands.
- `crates/unbg-core`: shared inference policy and core contracts.
- `crates/unbg-model-registry`: model directory and lockfile types.
- `crates/unbg-installer`: local model install and verification flow.
- `crates/unbg-runtime-ort`: ONNX runtime integration surface (stubbed).
- `crates/unbg-image`: image sizing helpers (placeholder utilities).
- `crates/unbg-bench`: benchmark case definitions.
- `crates/unbg-uniffi`: shared FFI-safe boundary for mobile bindings.
- `integrations/tauri-plugin-unbg`: Tauri adapter over the shared core/runtime.
- `integrations/android-unbg`: Android bridge contract for local inference.
- `integrations/ios-unbg`: iOS bridge contract for local inference.
- `apps/smoke-tests`: cross-runtime conformance smoke tests.

## Commands

```bash
cargo build
cargo run -p unbg-cli -- models install --model fast
cargo run -p unbg-cli -- models install --all
cargo run -p unbg-cli -- models list
cargo run -p unbg-cli -- models verify
cargo run -p unbg-cli -- exec -i ./input.jpg
cargo run -p unbg-cli -- exec -i ./input.jpg -o ./out/cutout.png
cargo run -p unbg-cli -- exec -i ./input.jpg -m ./out/mask.png -M quality
cargo run -p smoke-tests
cargo test
```

`exec` defaults to model `fast` and writes `<input>_cutout.<ext>` in the same directory when no output flags are provided.
If required models are missing, `exec` installs them automatically before inference.

## CLI Install One-Liners

Use these commands:

```powershell
irm https://unbg.io/install.ps1 | iex
```

```bash
curl -fsSL https://unbg.io/install.sh | sh
```

Installer scripts are included at the workspace root as `install.ps1` and `install.sh`.
They auto-detect OS and architecture, then download the matching GitHub Release archive for `unbg`.

Optional environment overrides:

- `UNBG_INSTALL_REPO` (default: `unbgio/core-sdk`)
- `UNBG_INSTALL_VERSION` (default: `latest`, or set a tag like `v0.1.0`)
- `UNBG_INSTALL_DIR` (default: per-user bin directory)
- `UNBG_BINARY_NAME` (default: `unbg` / `unbg.exe`)

## Model Bundle Prep

Use the helper script to prepare packaged model bundles for consumers:

```bash
# Tauri package: rmbg-1.4 + rmbg-2.0 (requires HF token)
HF_TOKEN=... scripts/prepare-model-bundle.sh --profile tauri-full --out ./dist/model-bundles/tauri

# Mobile package: rmbg-1.4 fp16 only
scripts/prepare-model-bundle.sh --profile mobile-lite --out ./dist/model-bundles/mobile
```

Both profiles run install + verify and write a full model directory that you can ship with your app and pass as `modelDir` in platform APIs.

Recommended profiles:

- `tauri-full`: `rmbg-1.4` + `rmbg-2.0` (requires `HF_TOKEN`)
- `mobile-lite`: `rmbg-1.4` fp16 only

`rmbg-2.0` is gated and requires `HF_TOKEN` during install.
Model aliases: `fast` -> `rmbg-1.4`, `quality` -> `rmbg-2.0`.

## Integration Packaging

- Android AAR: `scripts/build-android.sh`
- iOS XCFramework: `scripts/build-ios.sh`
- Tauri plugin mode: `cargo build -p tauri-plugin-unbg --features tauri-plugin`

CI workflows are defined under `.github/workflows/` for workspace checks plus Tauri/Android/iOS artifacts.

## Compatibility and Security

- API compatibility policy: `docs/API_COMPATIBILITY.md`
- Security/release hygiene: `docs/SECURITY_RELEASE.md`
- Compatibility snapshot check: `python scripts/check-api-compat.py`
- Artifact manifest generation: `scripts/release-manifest.sh`
- Artifact signing helper: `scripts/sign-artifacts.sh`

Telemetry sinks can be configured with:

- `UNBG_TELEMETRY_SINK=stdout|file|http`
- `UNBG_TELEMETRY_FILE=/path/to/telemetry.log` (for file sink)
- `UNBG_TELEMETRY_ENDPOINT=https://example.com/events` (for http sink)
