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

## Consume in iOS app

Option A: drag `UNBG.xcframework` into Xcode and link it.

Option B: CocoaPods with `UNBG.podspec` in this directory.

## Generate Swift bindings

Generate Swift bindings from `crates/unbg-uniffi/src/unbg.udl` using `uniffi-bindgen`.

## Smoke check

Use shared workspace smoke tests:

```bash
cargo test -p smoke-tests
```
