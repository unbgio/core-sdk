# iOS UNBG Integration

This directory packages `unbg-uniffi` into an iOS XCFramework.

For a consumer-first setup (including model bundle packaging and `modelDir` runtime wiring), start here:

- `../../docs/IOS_CONSUMER.md`

## Build XCFramework

From workspace root:

```bash
scripts/build-ios.sh
```

Output:

- `integrations/ios-unbg/dist/UNBG.xcframework`
- CI release artifact `UNBG-SDK.zip`, containing the XCFramework, generated
  Swift bindings, typed facade, Swift package manifest, and podspec

## Consume in iOS app

Option A: drag `UNBG.xcframework` into Xcode and link it.

Option B: CocoaPods with `UNBG.podspec` in this directory.

The build script generates the Swift bindings and embeds the C header/module
map in the XCFramework. Consumers do not run `uniffi-bindgen`.

## Smoke check

Use shared workspace smoke tests:

```bash
cargo test -p smoke-tests
```
