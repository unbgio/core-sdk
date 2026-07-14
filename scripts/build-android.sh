#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/integrations/android-unbg/dist/aar"
JNI_DIR="$DIST_DIR/jni"
GEN_DIR="$ROOT_DIR/integrations/android-unbg/generated"
RESOURCE_DIR="$ROOT_DIR/integrations/android-unbg/generated-resources/META-INF"
UDL_PATH="$ROOT_DIR/crates/unbg-uniffi/src/unbg.udl"
UNIFFI_BINDGEN_VERSION="${UNIFFI_BINDGEN_VERSION:-0.32.0}"
GRADLEW="$ROOT_DIR/integrations/android-unbg/gradlew"

declare -A TARGET_TO_ABI=(
  ["aarch64-linux-android"]="arm64-v8a"
  ["armv7-linux-androideabi"]="armeabi-v7a"
  ["x86_64-linux-android"]="x86_64"
)

rm -rf "$DIST_DIR"
mkdir -p "$JNI_DIR"
rm -rf "$GEN_DIR"
mkdir -p "$GEN_DIR"
rm -rf "$ROOT_DIR/integrations/android-unbg/generated-resources"
mkdir -p "$RESOURCE_DIR"
cp "$ROOT_DIR/LICENSE" "$RESOURCE_DIR/LICENSE"
cp "$ROOT_DIR/MODEL_LICENSES.md" "$RESOURCE_DIR/MODEL_LICENSES.md"

if ! command -v uniffi-bindgen >/dev/null 2>&1; then
  echo "uniffi-bindgen is required for deterministic Android packaging."
  echo "Install with: cargo install uniffi --version $UNIFFI_BINDGEN_VERSION --features cli --bin uniffi-bindgen --locked"
  exit 1
fi
if ! uniffi-bindgen --version | grep -Fq "$UNIFFI_BINDGEN_VERSION"; then
  echo "uniffi-bindgen $UNIFFI_BINDGEN_VERSION is required."
  exit 1
fi

if [[ ! -f "$GRADLEW" ]]; then
  echo "The pinned Gradle wrapper is required to assemble a complete AAR."
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
  cargo build -p unbg-uniffi --release --target "$TARGET" --locked
  mkdir -p "$JNI_DIR/$ABI"
  cp "$ROOT_DIR/target/$TARGET/release/libunbg_uniffi.so" "$JNI_DIR/$ABI/"
done

bash "$GRADLEW" --no-daemon -p "$ROOT_DIR/integrations/android-unbg" assembleRelease
AAR_SOURCE="$(find "$ROOT_DIR/integrations/android-unbg/build/outputs/aar" -maxdepth 1 -type f -name "*-release.aar" -print -quit)"
if [[ -z "$AAR_SOURCE" || ! -f "$AAR_SOURCE" ]]; then
  echo "Gradle did not produce a release AAR."
  exit 1
fi
cp "$AAR_SOURCE" "$DIST_DIR/unbg-android.aar"

echo "AAR available at: $DIST_DIR/unbg-android.aar"
