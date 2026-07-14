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

3. Add the JNA runtime required by the generated UniFFI bindings:

```kotlin
dependencies {
    implementation(files("libs/unbg-android.aar"))
    implementation("net.java.dev.jna:jna:5.19.1@aar")
}
```

The build script generates and compiles the Kotlin UniFFI bindings and the
`com.unbg.sdk.UnbgClient` facade into the AAR; consumers do not run
`uniffi-bindgen`.

## Smoke check

Use workspace smoke tests:

```bash
cargo test -p smoke-tests
```
