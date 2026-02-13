# Tauri Plugin UNBG Integration

This package provides:

- Rust Tauri plugin crate: `tauri-plugin-unbg` (command handler + runtime bridge)
- Typed TS client package: `@unbg/tauri-plugin-unbg`

For a consumer-first setup (plugin registration, frontend call usage, and model bundle `modelDir` wiring), use:

- `../../docs/TAURI_CONSUMER.md`

Build the typed TS client package:

```bash
npm run build
```

Build Rust plugin crate (workspace root):

```bash
cargo build -p tauri-plugin-unbg --features tauri-plugin
```
