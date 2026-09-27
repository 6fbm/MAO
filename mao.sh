#!/usr/bin/env bash
# Multi AI Orchestrator - launcher for Linux and macOS
set -euo pipefail
MAO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"

if [ ! -x "$MAO_ROOT/.venv/bin/python" ]; then
  echo "The virtual environment is missing. Please run ./install.sh first." >&2
  exit 1
fi

export MAO_HOME="$MAO_ROOT"
exec "$MAO_ROOT/.venv/bin/python" -m mao "$@"
