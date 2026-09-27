@echo off
rem Multi AI Orchestrator - one step from a fresh clone to a running program:
rem sets the environment up on the first run, then starts mao. Every argument
rem is passed straight through, so "start.bat doctor" works too.
setlocal
cd /d "%~dp0"

rem Not just "does .venv exist" - an install that was interrupted leaves a venv
rem behind without the package, and the error that follows is unhelpful.
set "MAO_READY="
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "import mao" >nul 2>nul && set "MAO_READY=1"
)

if not defined MAO_READY (
  echo Setting up the environment. This takes a moment.
  echo.
  call install.cmd
  if errorlevel 1 exit /b 1
  echo.
)

call mao.cmd %*
exit /b %ERRORLEVEL%
