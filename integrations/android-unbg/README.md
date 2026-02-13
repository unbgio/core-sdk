# Android UNBG Integration

This directory produces Android-consumable artifacts from the shared `unbg-uniffi` crate.

For a consumer-first setup (including model bundle packaging and `modelDir` runtime wiring), start here:

- `../../docs/ANDROID_CONSUMER.md`

## Build AAR

From workspace root:

```bash
scripts/build-android.sh
```

Output:

- `integrations/android-unbg/dist/aar/unbg-android.aar`
- `integrations/android-unbg/dist/aar/jni/<abi>/libunbg_uniffi.so`

## Gradle consumption

1. Copy `unbg-android.aar` into your Android app `libs/`.
2. Add to app `build.gradle(.kts)`:

```kotlin
dependencies {
    implementation(files("libs/unbg-android.aar"))
}
```

3. Generate Kotlin bindings from `crates/unbg-uniffi/src/unbg.udl` with `uniffi-bindgen`.

## Smoke check

Use workspace smoke tests:

```bash
cargo test -p smoke-tests
```
