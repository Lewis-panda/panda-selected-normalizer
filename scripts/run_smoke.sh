#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "No python or python3 executable found" >&2
    exit 127
  fi
fi

"$PYTHON_BIN" examples/minimal_selected_normalizer_smoke.py
"$PYTHON_BIN" examples/multiselected_parity_smoke.py
if "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pytest tests -q
else
  "$PYTHON_BIN" tests/test_selected_normalizer_parity.py
fi
