@echo off
rem Multi AI Orchestrator - installation (creates .venv and installs the package)
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL%==0 (
  if not exist ".venv\Scripts\python.exe" (
    uv venv .venv --python 3.12 || goto :error
  )
  uv pip install --python .venv\Scripts\python.exe -e ".[dev]" || goto :error
) else (
  if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || goto :error
  )
  .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
  .venv\Scripts\python.exe -m pip install -e ".[dev]" || goto :error
)

.venv\Scripts\python.exe -m mao init || goto :error
echo.
echo Installation finished.
echo Start:        mao.cmd
echo Offline demo: mao.cmd demo-workspace demo-project  then  mao.cmd --demo --workspace demo-project
exit /b 0

:error
echo Installation failed.
exit /b 1
