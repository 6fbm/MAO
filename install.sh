#!/usr/bin/env bash
# Multi AI Orchestrator - installation (creates .venv and installs the package)
set -euo pipefail
cd -- "$(dirname -- "$0")"

PY=".venv/bin/python"
# Keep __pycache__ out of the source tree without giving up the cache itself.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PWD/.venv/pycache}"

if [ -d ".venv" ] && [ ! -x "$PY" ]; then
  if [ -d ".venv/Scripts" ]; then
    echo "There is a Windows .venv in this folder (.venv/Scripts). Please remove it:" >&2
    echo "  rm -rf .venv" >&2
  else
    echo "'.venv' exists but has no $PY. Please check it or remove it." >&2
  fi
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  [ -x "$PY" ] || uv venv .venv --python 3.12
  uv pip install --python "$PY" -e ".[dev]"
else
  [ -x "$PY" ] || python3 -m venv .venv
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -e ".[dev]"
fi

"$PY" -m mao init

echo
echo "Installation finished."
echo "Start:        ./mao.sh"
echo "Offline demo: ./mao.sh demo-workspace demo-project  then  ./mao.sh --demo --workspace demo-project"
