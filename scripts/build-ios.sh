#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/integrations/ios-unbg/dist"
LIB_NAME="libunbg_uniffi.a"
GEN_DIR="$ROOT_DIR/integrations/ios-unbg/generated"
UDL_PATH="$ROOT_DIR/crates/unbg-uniffi/src/unbg.udl"
UNIFFI_BINDGEN_VERSION="${UNIFFI_BINDGEN_VERSION:-0.30.0}"

IOS_DEVICE_TARGET="aarch64-apple-ios"
IOS_SIM_ARM_TARGET="aarch64-apple-ios-sim"
IOS_SIM_X64_TARGET="x86_64-apple-ios"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"
rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"

if ! command -v uniffi-bindgen >/dev/null 2>&1; then
  echo "uniffi-bindgen is required for deterministic iOS packaging."
  echo "Install with: cargo install uniffi_bindgen --version $UNIFFI_BINDGEN_VERSION --locked"
  exit 1
fi
uniffi-bindgen generate "$UDL_PATH" --language swift --out-dir "$GEN_DIR"
if ! find "$GEN_DIR" -type f -name "*.swift" | grep -q .; then
  echo "No Swift binding output generated in $GEN_DIR"
  exit 1
fi
echo "Generated Swift bindings in $GEN_DIR"

echo "Building iOS static libraries"
cargo build -p unbg-uniffi --release --target "$IOS_DEVICE_TARGET"
cargo build -p unbg-uniffi --release --target "$IOS_SIM_ARM_TARGET"
cargo build -p unbg-uniffi --release --target "$IOS_SIM_X64_TARGET"

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
  -library "$ROOT_DIR/target/$IOS_DEVICE_TARGET/release/$LIB_NAME" \
  -library "$DIST_DIR/$LIB_NAME-sim-universal.a" \
  -output "$DIST_DIR/UNBG.xcframework"

echo "XCFramework available at: $DIST_DIR/UNBG.xcframework"
