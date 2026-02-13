#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare a local model bundle for app packaging.

Usage:
  scripts/prepare-model-bundle.sh --profile <tauri-full|mobile-lite> --out <dir> [options]

Required:
  --profile <name>            Bundle profile: tauri-full | mobile-lite
  --out <dir>                 Output model directory

Options:
  --onnx-variant <variant>    fp16 | fp32 | quantized | auto (default: fp16)
  --revision-rmbg14 <rev>     Revision for RMBG-1.4 (default: main)
  --revision-rmbg20 <rev>     Revision for RMBG-2.0 (default: main)
  --hf-token-env <env>        Env var name for HF token (default: HF_TOKEN)
  -h, --help                  Show this help

Profiles:
  tauri-full                  Installs rmbg-1.4 + rmbg-2.0
  mobile-lite                 Installs rmbg-1.4 only
EOF
}

PROFILE=""
OUT_DIR=""
ONNX_VARIANT="fp16"
REV_RMBG14="main"
REV_RMBG20="main"
HF_TOKEN_ENV="HF_TOKEN"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --onnx-variant)
      ONNX_VARIANT="${2:-}"
      shift 2
      ;;
    --revision-rmbg14)
      REV_RMBG14="${2:-}"
      shift 2
      ;;
    --revision-rmbg20)
      REV_RMBG20="${2:-}"
      shift 2
      ;;
    --hf-token-env)
      HF_TOKEN_ENV="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROFILE" || -z "$OUT_DIR" ]]; then
  echo "--profile and --out are required." >&2
  usage
  exit 1
fi

MODELS=()
case "$PROFILE" in
  tauri-full)
    MODELS=("fast" "quality")
    if [[ -z "${!HF_TOKEN_ENV:-}" ]]; then
      echo "Profile '$PROFILE' requires gated model access." >&2
      echo "Set token env var '$HF_TOKEN_ENV' before running this script." >&2
      exit 1
    fi
    ;;
  mobile-lite)
    MODELS=("fast")
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

install_cmd=(
  cargo run -p unbg-cli -- models install
  --model-dir "$OUT_DIR"
  --onnx-variant "$ONNX_VARIANT"
  --revision-rmbg14 "$REV_RMBG14"
  --revision-rmbg20 "$REV_RMBG20"
  --hf-token-env "$HF_TOKEN_ENV"
)
for model in "${MODELS[@]}"; do
  install_cmd+=(--model "$model")
done

echo "Preparing model bundle profile '$PROFILE' in '$OUT_DIR'..."
"${install_cmd[@]}"

echo "Verifying installed model bundle..."
cargo run -p unbg-cli -- models verify --model-dir "$OUT_DIR" >/dev/null

echo "Model bundle ready: $OUT_DIR"
echo "Installed models: ${MODELS[*]}"
echo "ONNX variant: $ONNX_VARIANT"
