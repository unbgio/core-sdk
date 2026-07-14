#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/integrations/ios-unbg/dist"
LIB_NAME="libunbg_uniffi.a"
GEN_DIR="$ROOT_DIR/integrations/ios-unbg/generated"
SWIFT_GEN_DIR="$ROOT_DIR/integrations/ios-unbg/Sources/UNBG/Generated"
HEADERS_DIR="$DIST_DIR/headers"
UDL_PATH="$ROOT_DIR/crates/unbg-uniffi/src/unbg.udl"
UNIFFI_BINDGEN_VERSION="${UNIFFI_BINDGEN_VERSION:-0.32.0}"

IOS_DEVICE_TARGET="aarch64-apple-ios"
IOS_SIM_ARM_TARGET="aarch64-apple-ios-sim"
IOS_SIM_X64_TARGET="x86_64-apple-ios"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"
rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"
rm -rf "$SWIFT_GEN_DIR"
mkdir -p "$SWIFT_GEN_DIR"

if ! command -v uniffi-bindgen >/dev/null 2>&1; then
  echo "uniffi-bindgen is required for deterministic iOS packaging."
  echo "Install with: cargo install uniffi --version $UNIFFI_BINDGEN_VERSION --features cli --bin uniffi-bindgen --locked"
  exit 1
fi
if ! uniffi-bindgen --version | grep -Fq "$UNIFFI_BINDGEN_VERSION"; then
  echo "uniffi-bindgen $UNIFFI_BINDGEN_VERSION is required."
  exit 1
fi
uniffi-bindgen generate "$UDL_PATH" --language swift --out-dir "$GEN_DIR"
if ! find "$GEN_DIR" -type f -name "*.swift" | grep -q .; then
  echo "No Swift binding output generated in $GEN_DIR"
  exit 1
fi
find "$GEN_DIR" -maxdepth 1 -type f -name "*.swift" -exec cp {} "$SWIFT_GEN_DIR/" \;

FFI_HEADER="$(find "$GEN_DIR" -maxdepth 1 -type f -name "*FFI.h" -print -quit)"
FFI_MODULEMAP="$(find "$GEN_DIR" -maxdepth 1 -type f -name "*FFI.modulemap" -print -quit)"
if [[ -z "$FFI_HEADER" || -z "$FFI_MODULEMAP" ]]; then
  echo "Generated Swift bindings did not include an FFI header and module map."
  exit 1
fi
mkdir -p "$HEADERS_DIR"
cp "$FFI_HEADER" "$HEADERS_DIR/"
cp "$FFI_MODULEMAP" "$HEADERS_DIR/module.modulemap"
echo "Generated Swift bindings in $GEN_DIR"

echo "Building iOS static libraries"
cargo build -p unbg-uniffi --release --target "$IOS_DEVICE_TARGET" --locked
cargo build -p unbg-uniffi --release --target "$IOS_SIM_ARM_TARGET" --locked
cargo build -p unbg-uniffi --release --target "$IOS_SIM_X64_TARGET" --locked

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "xcodebuild not found; skipping XCFramework assembly."
  echo "Built static libs under target/<triple>/release/$LIB_NAME"
  exit 0
fi

lipo -create \
  "$ROOT_DIR/target/$IOS_SIM_ARM_TARGET/release/$LIB_NAME" \
  "$ROOT_DIR/target/$IOS_SIM_X64_TARGET/release/$LIB_NAME" \
  -output "$DIST_DIR/$LIB_NAME-sim-universal.a"

xcodebuild -create-xcframework \
  -library "$ROOT_DIR/target/$IOS_DEVICE_TARGET/release/$LIB_NAME" -headers "$HEADERS_DIR" \
  -library "$DIST_DIR/$LIB_NAME-sim-universal.a" -headers "$HEADERS_DIR" \
  -output "$DIST_DIR/UNBG.xcframework"

echo "XCFramework available at: $DIST_DIR/UNBG.xcframework"
