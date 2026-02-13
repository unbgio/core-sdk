# Troubleshooting

Common integration and CI issues with fixes.

## Models

### "model not found in lockfile" or "no .onnx file found"

- Cause: `modelDir` points to wrong folder or incomplete bundle.
- Fix:
  - Regenerate bundle with `scripts/prepare-model-bundle.sh`.
  - Verify with `cargo run -p unbg-cli -- models verify --model-dir <dir>`.
  - Pass the same `<dir>` as `modelDir` in runtime request.

### `rmbg-2.0` install fails

- Cause: `rmbg-2.0` is gated.
- Fix: set `HF_TOKEN` (or custom `--hf-token-env`) before running `tauri-full` profile.

## Signing and release workflows

### Sign step skipped

- Cause: signing secret missing or invalid key format.
- Fix:
  - Ensure `UNBG_SIGNING_KEY_B64` is configured in repository secrets.
  - Use a base64-encoded PEM private key in a single line.

### "Could not read private key" / "bad end line"

- Cause: malformed secret value.
- Fix:
  - Regenerate key and set secret again without truncation.
  - Verify key locally with `openssl pkey -in <pem> -noout`.

## Android build

### ring/cc-rs cannot find Android compiler or archiver

- Cause: missing NDK toolchain env wiring.
- Fix: use repository Android workflow/tooling setup (NDK + linker + `CC_*` + `AR_*`).

## iOS build

### XCFramework creation fails with duplicate simulator definitions

- Cause: passing both simulator libraries separately.
- Fix: merge simulator libs with `lipo` and pass one simulator library to `xcodebuild -create-xcframework`.

## Tauri integration

### Command not found: `plugin:unbg|tauri_remove_background_command`

- Cause: plugin not registered in Rust app.
- Fix: add `.plugin(tauri_plugin_unbg::init())` in Tauri builder.
