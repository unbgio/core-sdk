# Android Consumer Quickstart

This guide shows the simplest integration path for Android apps.

## 1) Prepare model bundle (recommended mobile profile)

From workspace root:

```bash
scripts/prepare-model-bundle.sh --profile mobile-lite --out ./dist/model-bundles/mobile
```

This installs and verifies `rmbg-1.4` with fp16 defaults.

## 2) Build Android artifact

```bash
scripts/build-android.sh
```

Output:

- `integrations/android-unbg/dist/aar/unbg-android.aar`

## 3) Add AAR to Android app

Copy the AAR to your app `libs/` and add:

```kotlin
dependencies {
    implementation(files("libs/unbg-android.aar"))
}
```

## 4) Package model files

Bundle `./dist/model-bundles/mobile` with your app (usually under assets).
Because Android assets are not regular files, extract that folder to a writable directory (for example `filesDir`) on first launch.

Use the extracted absolute path as `modelDir` when calling UNBG.

## 5) Call API

`UnbgClient` request supports `modelDir` and `onnxVariant`.

```kotlin
val result = UnbgClient.removeBackground(
    UnbgClient.RemoveBackgroundRequest(
        imageBytes = imageBytes,
        width = width.toUInt(),
        height = height.toUInt(),
        model = "fast",
        onnxVariant = "fp16",
        modelDir = extractedModelDir
    )
)
```

## 6) Notes

- `model = "fast"` is recommended for mobile packaging.
- You can omit `modelDir` only if models are already installed in default runtime paths.
- Keep the lockfile + model files together from bundle output.
