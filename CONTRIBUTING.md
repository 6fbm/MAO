# Contributing

Thanks for taking a look. Issues, bug reports and pull requests are welcome.

## Development setup

Clone the repository, then:

```bash
cd MAO
./install.sh                     # Windows: install.cmd
```

(`./start.sh` does the same and launches the program straight after.)

This creates `.venv`, installs the package in editable mode with the dev extras, and generates your
local `config/` from `src/mao/config/templates/`. Your `config/` is not tracked by git — change the
templates when you want a default to change for everyone.

## Tests

```bash
.venv/bin/python -m pytest       # Windows: .venv\Scripts\python.exe -m pytest
```

The launchers set `PYTHONPYCACHEPREFIX` so that `__pycache__` lands in `.venv/pycache` instead of the
source tree. Running `pytest` straight from your shell bypasses them, so export it once per shell if
you want the tree to stay clean:

```bash
export PYTHONPYCACHEPREFIX="$PWD/.venv/pycache"
```

The suite runs offline and needs no API keys: the markers `ollama` and `network` are deselected by
default in `pyproject.toml`. To run them anyway:

```bash
.venv/bin/python -m pytest -m ollama    # needs a running Ollama with a tool-capable model
.venv/bin/python -m pytest -m network   # needs internet access
```

CI runs the same suite on Linux, macOS and Windows against Python 3.11–3.14. Please make sure it is
green before opening a pull request.

## Working without API keys

The offline demo exercises the whole PLAN→RUN pipeline with simulated model replies — real tools, real
files, real test runs:

```bash
./start.sh demo-workspace demo-project
./start.sh --demo --workspace demo-project
```

## Conventions

- **Language**: docstrings and code comments are English; everything a user sees (CLI output, error
  messages, the YAML template comments) is German. Please keep both sides consistent when you touch them.
- **Configuration**: every new option belongs in the matching model in `src/mao/config/schema.py` *and*
  in the commented template under `src/mao/config/templates/`. Configuration is validated strictly —
  unknown keys are an error, not a warning.
- **Security**: new tools must go through the existing permission and approval path
  (`src/mao/security/`). Never let a tool receive API keys in a subprocess environment, and never log a
  secret — see [docs/SECURITY.md](docs/SECURITY.md).
- **Line endings** are handled by `.gitattributes`: LF in the repository, `*.cmd` checked out as CRLF,
  `*.sh` always LF.
- Please add a test for what you change. The existing tests under `tests/unit/` and
  `tests/integration/` are a good template.

## Pull requests

- One topic per pull request, with a short description of what changes and why.
- Mention it in [CHANGELOG.md](CHANGELOG.md) under "Unreleased" when the change is user-visible.
- New provider? [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the extension points.
