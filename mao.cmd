@echo off
rem Multi AI Orchestrator - launcher for Windows (CMD / Windows Terminal)
setlocal
set "MAO_ROOT=%~dp0"
if not exist "%MAO_ROOT%.venv\Scripts\python.exe" (
  echo The virtual environment is missing. Please run install.cmd first.
  exit /b 1
)
set "MAO_HOME=%MAO_ROOT%"
rem Keep __pycache__ out of the source tree without giving up the cache itself.
if not defined PYTHONPYCACHEPREFIX set "PYTHONPYCACHEPREFIX=%MAO_ROOT%.venv\pycache"
"%MAO_ROOT%.venv\Scripts\python.exe" -m mao %*
exit /b %ERRORLEVEL%
