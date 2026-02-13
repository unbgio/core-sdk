#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/integrations/android-unbg/dist/aar"
JNI_DIR="$DIST_DIR/jni"
MANIFEST="$ROOT_DIR/integrations/android-unbg/AndroidManifest.xml"
GEN_DIR="$ROOT_DIR/integrations/android-unbg/generated"
UDL_PATH="$ROOT_DIR/crates/unbg-uniffi/src/unbg.udl"
UNIFFI_BINDGEN_VERSION="${UNIFFI_BINDGEN_VERSION:-0.30.0}"

declare -A TARGET_TO_ABI=(
  ["aarch64-linux-android"]="arm64-v8a"
  ["armv7-linux-androideabi"]="armeabi-v7a"
  ["x86_64-linux-android"]="x86_64"
)

rm -rf "$DIST_DIR"
mkdir -p "$JNI_DIR"
rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"

if ! command -v uniffi-bindgen >/dev/null 2>&1; then
  echo "uniffi-bindgen is required for deterministic Android packaging."
  echo "Install with: cargo install uniffi_bindgen --version $UNIFFI_BINDGEN_VERSION --locked"
  exit 1
fi

uniffi-bindgen generate "$UDL_PATH" --language kotlin --out-dir "$GEN_DIR"
if ! find "$GEN_DIR" -type f -name "*.kt" | grep -q .; then
  echo "No Kotlin binding output generated in $GEN_DIR"
  exit 1
fi
echo "Generated Kotlin bindings in $GEN_DIR"

for TARGET in "${!TARGET_TO_ABI[@]}"; do
  ABI="${TARGET_TO_ABI[$TARGET]}"
  echo "Building unbg-uniffi for $TARGET ($ABI)"
  cargo build -p unbg-uniffi --release --target "$TARGET"
  mkdir -p "$JNI_DIR/$ABI"
  cp "$ROOT_DIR/target/$TARGET/release/libunbg_uniffi.so" "$JNI_DIR/$ABI/"
done

cp "$MANIFEST" "$DIST_DIR/"
(
  cd "$DIST_DIR"
  zip -r "unbg-android.aar" "AndroidManifest.xml" "jni"
)

echo "AAR available at: $DIST_DIR/unbg-android.aar"
