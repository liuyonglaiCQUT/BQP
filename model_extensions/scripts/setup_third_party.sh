#!/usr/bin/env bash
set -euo pipefail

EXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMKGC_DIR="$EXT_ROOT/repos/SimKGC"
SIMKGC_URL="https://github.com/intfloat/SimKGC.git"
SIMKGC_COMMIT="97cc43e488f19ca5b0f6fbf60ffefd2ee56c0693"
PATCH="$EXT_ROOT/patches/SimKGC.patch"

command -v git >/dev/null || {
  echo "git is required" >&2
  exit 1
}

if [[ ! -e "$SIMKGC_DIR" ]]; then
  git clone "$SIMKGC_URL" "$SIMKGC_DIR"
elif [[ ! -d "$SIMKGC_DIR/.git" ]]; then
  echo "$SIMKGC_DIR exists but is not a Git checkout; move it aside and retry" >&2
  exit 1
fi

git -C "$SIMKGC_DIR" fetch origin "$SIMKGC_COMMIT"
git -C "$SIMKGC_DIR" checkout --detach "$SIMKGC_COMMIT"

if git -C "$SIMKGC_DIR" apply --check "$PATCH"; then
  git -C "$SIMKGC_DIR" apply "$PATCH"
  echo "Applied SimKGC BQP patch."
elif git -C "$SIMKGC_DIR" apply --reverse --check "$PATCH"; then
  echo "SimKGC BQP patch is already applied."
else
  echo "SimKGC checkout does not match the expected patch base." >&2
  exit 1
fi

actual="$(git -C "$SIMKGC_DIR" rev-parse HEAD)"
[[ "$actual" == "$SIMKGC_COMMIT" ]] || {
  echo "Unexpected SimKGC commit: $actual" >&2
  exit 1
}

echo "SimKGC source ready at $SIMKGC_DIR"
