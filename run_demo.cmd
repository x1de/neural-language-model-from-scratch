@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%~dp0"
if "%~1"=="" (
    "%PYTHON_EXE%" tui.py
) else (
    "%PYTHON_EXE%" demo.py %*
)
