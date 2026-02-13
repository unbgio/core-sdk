#!/usr/bin/env sh
set -eu

REPO="${UNBG_INSTALL_REPO:-unbgio/core-sdk}"
VERSION="${UNBG_INSTALL_VERSION:-latest}"
BINARY_NAME="${UNBG_BINARY_NAME:-unbg}"
INSTALL_DIR="${UNBG_INSTALL_DIR:-$HOME/.local/bin}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install $BINARY_NAME." >&2
  exit 1
fi

detect_os() {
  case "$(uname -s)" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    MINGW*|MSYS*|CYGWIN*)
      echo "This shell is running on Windows (Git Bash/MSYS/Cygwin)." >&2
      echo "Use PowerShell installer instead:" >&2
      echo "  irm https://unbg.io/install.ps1 | iex" >&2
      exit 1
      ;;
    *)
      echo "Unsupported OS: $(uname -s)" >&2
      exit 1
      ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    arm64|aarch64) echo "aarch64" ;;
    *)
      echo "Unsupported CPU architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
}

extract_archive() {
  archive="$1"
  out_dir="$2"

  case "$archive" in
    *.zip)
      if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip is required to extract $archive." >&2
        exit 1
      fi
      unzip -q "$archive" -d "$out_dir"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$out_dir"
      ;;
    *)
      echo "Unsupported archive format: $archive" >&2
      exit 1
      ;;
  esac
}

os="$(detect_os)"
arch="$(detect_arch)"
target="$os-$arch"

case "$VERSION" in
  latest)
    asset_url="https://github.com/$REPO/releases/latest/download/unbg-$target.zip"
    ;;
  *)
    asset_url="https://github.com/$REPO/releases/download/$VERSION/unbg-$target.zip"
    ;;
esac

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

archive_name="unbg-$target.zip"
archive_path="$tmp_dir/$archive_name"
extract_dir="$tmp_dir/extract"
mkdir -p "$extract_dir"

echo "Downloading $asset_url"
if ! curl -fsSL "$asset_url" -o "$archive_path"; then
  echo "Could not download asset for target $target in $REPO ($VERSION)." >&2
  echo "Expected URL: $asset_url" >&2
  exit 1
fi

echo "Extracting $archive_name"
extract_archive "$archive_path" "$extract_dir"

binary_path="$(find "$extract_dir" -type f -name "$BINARY_NAME" | head -n 1 || true)"
if [ -z "$binary_path" ]; then
  echo "Could not find '$BINARY_NAME' inside the release archive." >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
find "$extract_dir" -type f -exec cp {} "$INSTALL_DIR/" \;
chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo "Installed $BINARY_NAME to $INSTALL_DIR/$BINARY_NAME"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    echo "Add this directory to PATH if needed:"
    echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac
