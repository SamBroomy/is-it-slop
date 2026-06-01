#!/bin/sh
# is-it-slop installer
# https://github.com/SamBroomy/is-it-slop
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.sh | sh
#
# Install specific version:
#   curl -fsSL https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.sh | sh -s -- v0.6.0
# Running the script on an existing installation will upgrade it automatically.
set -eu

# Configuration
REPO="SamBroomy/is-it-slop"
BINARY_NAME="is-it-slop"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"

# Colors for output (fallback if tput not available)
if command -v tput >/dev/null 2>&1 && [ -t 1 ]; then
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4)
  BOLD=$(tput bold)
  RESET=$(tput sgr0)
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  BOLD=''
  RESET=''
fi

# Logging functions
info() {
  printf "%s\n" "${GREEN}${1}${RESET}"
}

warn() {
  printf "%s\n" "${YELLOW}Warning: ${1}${RESET}" >&2
}

error() {
  printf "%s\n" "${RED}Error: ${1}${RESET}" >&2
}

# Check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Require a command to exist
need_cmd() {
  if ! command_exists "$1"; then
    error "$1 is required but not installed"
    exit 1
  fi
}

# Detect platform (OS and architecture)
detect_platform() {
  local os arch libc_type

  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)

  # Normalize architecture
  case "$arch" in
    x86_64|amd64)
      arch="x86_64"
      ;;
    aarch64|arm64)
      arch="aarch64"
      ;;
    i386|i486|i686)
      error "32-bit systems are not supported"
      exit 1
      ;;
    *)
      error "Unsupported architecture: $arch"
      exit 1
      ;;
  esac

  # Normalize OS and determine target triple
  case "$os" in
    linux)
      # Detect libc type (glibc vs musl)
      if command_exists ldd; then
        if ldd --version 2>&1 | grep -qi musl; then
          libc_type="musl"
        else
          libc_type="gnu"
        fi
      else
        # Default to gnu if ldd not available
        libc_type="gnu"
      fi

      TARGET="${arch}-unknown-linux-${libc_type}"
      ARCHIVE_EXT="tar.gz"
      ;;
    darwin)
      # Check if Intel Mac
      if [ "$arch" = "x86_64" ]; then
        error "Intel Mac (x86_64) is not supported due to ONNX Runtime limitations"
        info "Please use one of these alternatives:"
        info "  1. Install via Python: pip install is-it-slop"
        info "  2. Install via uv: uv tool install is-it-slop"
        info "  3. Run on Apple Silicon Mac with Rosetta 2"
        exit 1
      fi

      TARGET="${arch}-apple-darwin"
      ARCHIVE_EXT="tar.gz"
      ;;
    mingw*|msys*|cygwin*)
      TARGET="${arch}-pc-windows-msvc"
      ARCHIVE_EXT="zip"
      ;;
    *)
      error "Unsupported operating system: $os"
      exit 1
      ;;
  esac
}

# Download a file using curl or wget
download_file() {
  local url="$1"
  local output="$2"

  if command_exists curl; then
    curl --proto '=https' --tlsv1.2 -fsSL "$url" -o "$output"
  elif command_exists wget; then
    wget --https-only -q -O "$output" "$url"
  else
    error "curl or wget is required for downloading"
    exit 1
  fi
}

# Fetch latest release version from GitHub API
get_latest_version() {
  local api_url="https://api.github.com/repos/$REPO/releases/latest"
  local version

  info "Fetching latest release information..." >&2

  if command_exists curl; then
    version=$(curl --proto '=https' --tlsv1.2 -fsSL "$api_url" | grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name": "([^"]+)".*/\1/')
  elif command_exists wget; then
    version=$(wget --https-only -qO- "$api_url" | grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name": "([^"]+)".*/\1/')
  else
    error "curl or wget is required"
    exit 1
  fi

  if [ -z "$version" ]; then
    error "Could not determine latest version"
    exit 1
  fi

  echo "$version"
}

# Extract archive based on file extension
extract_archive() {
  local archive="$1"
  local dest_dir="$2"

  case "$ARCHIVE_EXT" in
    tar.gz)
      need_cmd tar
      tar -xzf "$archive" -C "$dest_dir"
      ;;
    zip)
      if command_exists unzip; then
        unzip -q "$archive" -d "$dest_dir"
      elif command_exists 7z; then
        7z x "$archive" -o"$dest_dir" >/dev/null
      else
        error "unzip or 7z is required for extracting zip files"
        exit 1
      fi
      ;;
    *)
      error "Unknown archive format: $ARCHIVE_EXT"
      exit 1
      ;;
  esac
}

# Check if binary is already installed and show current version
check_existing_installation() {
  if [ -x "$INSTALL_DIR/$BINARY_NAME" ]; then
    local current_version
    current_version=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
    info "Existing installation detected: ${BLUE}${current_version}${RESET}"
    info "Replacing existing installation..."
    return 0
  fi
  return 1
}

# Main installation function
install_binary() {
  local version="${1:-latest}"

  info "${BOLD}Installing $BINARY_NAME${RESET}"

  # Detect platform
  detect_platform
  info "Detected platform: ${BLUE}${TARGET}${RESET}"

  # Resolve version
  if [ "$version" = "latest" ]; then
    version=$(get_latest_version)
  fi

  info "Version: ${BLUE}${version}${RESET}"

  # Check for existing installation
  check_existing_installation

  # Create temporary directory
  temp_dir=$(mktemp -d)
  trap 'rm -rf "$temp_dir"' EXIT

  # Build download URL
  local binary_archive="${BINARY_NAME}-${TARGET}.${ARCHIVE_EXT}"
  local download_url="https://github.com/$REPO/releases/download/${version}/${binary_archive}"

  info "Downloading from ${BLUE}${download_url}${RESET}"

  # Download
  local archive_path="${temp_dir}/${binary_archive}"
  if ! download_file "$download_url" "$archive_path"; then
    error "Failed to download binary from $download_url"
    info "Please check:"
    info "  1. Version ${version} exists: https://github.com/$REPO/releases"
    info "  2. Binary is available for platform: $TARGET"
    info "  3. Your internet connection is working"
    exit 1
  fi

  # Extract
  info "Extracting archive..."
  extract_archive "$archive_path" "$temp_dir"

  # Find binary in extracted files (handle different archive structures)
  local binary_path=""
  if [ -f "${temp_dir}/${BINARY_NAME}" ]; then
    binary_path="${temp_dir}/${BINARY_NAME}"
  elif [ -f "${temp_dir}/${BINARY_NAME}.exe" ]; then
    binary_path="${temp_dir}/${BINARY_NAME}.exe"
  else
    # Search in subdirectories
    binary_path=$(find "$temp_dir" -name "$BINARY_NAME" -o -name "${BINARY_NAME}.exe" | head -1)
  fi

  if [ -z "$binary_path" ] || [ ! -f "$binary_path" ]; then
    error "Could not find binary in archive"
    exit 1
  fi

  # Create installation directory
  if [ ! -d "$INSTALL_DIR" ]; then
    info "Creating installation directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
  fi

  # Install binary
  info "Installing to ${BLUE}${INSTALL_DIR}${RESET}"
  mv "$binary_path" "$INSTALL_DIR/$BINARY_NAME"
  chmod +x "$INSTALL_DIR/$BINARY_NAME"

  # Verify installation
  if [ ! -x "$INSTALL_DIR/$BINARY_NAME" ]; then
    error "Installation failed: binary not executable"
    exit 1
  fi

  # Success message
  printf "\n"
  info "${GREEN}${BOLD}✓ Successfully installed $BINARY_NAME${RESET}"
  printf "\n"

  # Verify version
  local installed_version
  installed_version=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
  info "Installed version: ${BLUE}${installed_version}${RESET}"
  printf "\n"

  # Check PATH and provide instructions
  case ":$PATH:" in
    *:"$INSTALL_DIR":*)
      info "${GREEN}✓ $INSTALL_DIR is in your PATH${RESET}"
      printf "\n"
      info "Try it now: ${YELLOW}$BINARY_NAME --help${RESET}"
      ;;
    *)
      warn "$INSTALL_DIR is NOT in your PATH"
      printf "\n"
      info "Add this line to your shell config file:"
      info "  ${YELLOW}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
      printf "\n"
      info "Shell config files:"
      info "  • bash:  ~/.bashrc or ~/.bash_profile"
      info "  • zsh:   ~/.zshrc"
      info "  • fish:  ~/.config/fish/config.fish"
      printf "\n"
      info "Then restart your shell or run: ${YELLOW}source ~/.bashrc${RESET}"
      printf "\n"
      info "Or use the full path: ${YELLOW}$INSTALL_DIR/$BINARY_NAME --help${RESET}"
      ;;
  esac
}

# Main entry point
install_binary "${1:-latest}"
