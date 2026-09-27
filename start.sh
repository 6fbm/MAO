#!/usr/bin/env bash
# Multi AI Orchestrator - one step from a fresh clone to a running program:
# sets the environment up on the first run, then starts mao. Every argument is
# passed straight through, so "./start.sh doctor" works too.
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Not just "does .venv exist" - an install that was interrupted leaves a venv
# behind without the package, and the error that follows is unhelpful.
if ! .venv/bin/python -c "import mao" >/dev/null 2>&1; then
  echo "Setting up the environment. This takes a moment."
  echo
  ./install.sh
  echo
fi

exec ./mao.sh "$@"
