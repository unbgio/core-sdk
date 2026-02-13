# Tauri Consumer Quickstart

This guide is for app teams consuming UNBG in a Tauri app with minimal setup.

## 1) Prepare model bundle

From workspace root:

```bash
# Includes rmbg-1.4 + rmbg-2.0 (gated)
HF_TOKEN=... scripts/prepare-model-bundle.sh --profile tauri-full --out ./dist/model-bundles/tauri
```

If you do not want gated `rmbg-2.0`, use:

```bash
scripts/prepare-model-bundle.sh --profile mobile-lite --out ./dist/model-bundles/tauri
```

## 2) Register plugin in Tauri Rust app

In your Tauri app `Cargo.toml`:

```toml
[dependencies]
tauri-plugin-unbg = { path = "../path/to/node-sdk/integrations/tauri-plugin-unbg", features = ["tauri-plugin"] }
```

In your Tauri Rust entrypoint:

```rust
tauri::Builder::default()
    .plugin(tauri_plugin_unbg::init())
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
```

## 3) Add model bundle as app resource

Add your prepared model directory to Tauri resources (for example `src-tauri/tauri.conf.json`).
At runtime, resolve its absolute path and pass it as `modelDir`.

## 4) Call from frontend

Install/use the typed client from this workspace package (`@unbg/tauri-plugin-unbg`) and pass Tauri `invoke`.

```ts
import { invoke } from "@tauri-apps/api/core";
import { removeBackground } from "@unbg/tauri-plugin-unbg";

const response = await removeBackground(invoke, {
  imageBytes: Array.from(inputBytes),
  width,
  height,
  model: "auto",
  onnxVariant: "fp16",
  modelDir: resolvedModelDir
});
```

## 5) Notes

- `model: "fast"` maps to `rmbg-1.4`.
- `model: "quality"` maps to `rmbg-2.0`.
- `modelDir` should point to the root bundle directory created by `prepare-model-bundle.sh`.
- If `modelDir` is not passed, runtime uses default model paths.
