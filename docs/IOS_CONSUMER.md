# iOS Consumer Quickstart

This guide shows the easiest path to consume UNBG from an iOS app.

## 1) Prepare model bundle (recommended mobile profile)

From workspace root:

```bash
scripts/prepare-model-bundle.sh --profile mobile-lite --out ./dist/model-bundles/mobile
```

## 2) Build XCFramework

```bash
scripts/build-ios.sh
```

Output:

- `integrations/ios-unbg/dist/UNBG.xcframework`

## 3) Add XCFramework to app

- Drag `UNBG.xcframework` into Xcode and link it, or
- Use your package manager flow around this artifact.

## 4) Package model files

Add `./dist/model-bundles/mobile` as app resources.
Resolve the bundled path at runtime and pass it as `modelDir`.

## 5) Call API

`UNBGClient.removeBackground` accepts `modelDir` and `onnxVariant`.

```swift
let req = UNBGRemoveBackgroundRequest(
    imageBytes: imageData,
    width: width,
    height: height,
    model: "fast",
    onnxVariant: "fp16",
    modelDir: modelDirPath
)
let res = try UNBGClient.removeBackground(req)
```

## 6) Notes

- `model = "fast"` maps to `rmbg-1.4`.
- `model = "quality"` maps to `rmbg-2.0` and should be used only if you bundle/install it.
- Keep bundle output intact so lockfile and files match verification expectations.
