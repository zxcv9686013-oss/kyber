@echo off
setlocal
set "XASM_DIR=%~dp0"

rem Accept bare IXASM variants as shorthand: IXASM, IXASM--TRUE, --ixasm, --ixasm=true
set "XASM_MODE="
if not "%~1"=="" (
  if /I "%~1"=="ixasm" (
    set "XASM_MODE=--ixasm"
    shift
  ) else if /I "%~1"=="ixasm--true" (
    set "XASM_MODE=--ixasm"
    shift
  ) else if /I "%~1"=="--ixasm" (
    set "XASM_MODE=--ixasm"
    shift
  ) else if /I "%~1"=="--ixasm=true" (
    set "XASM_MODE=--ixasm"
    shift
  ) else if /I "%~1"=="ixasm=true" (
    set "XASM_MODE=--ixasm"
    shift
  )
)

rem Prefer KYBER_PYTHON, then py, then python on PATH
if defined KYBER_PYTHON (
  set "PYTHON_EXE=%KYBER_PYTHON%"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    for /f "usebackq delims=" %%I in (`where py`) do set "PYTHON_EXE=%%I"
  ) else (
    where python >nul 2>nul
    if not errorlevel 1 (
      for /f "usebackq delims=" %%I in (`where python`) do set "PYTHON_EXE=%%I"
    )
  )
)

if not defined PYTHON_EXE (
  echo [xasm] Python not found. Set KYBER_PYTHON or add python/py to PATH.
  exit /b 1
)

if not exist "%XASM_DIR%xasm_compiler.py" (
  echo [xasm] xasm_compiler.py not found in %XASM_DIR%
  exit /b 1
)

if defined XASM_MODE (
  "%PYTHON_EXE%" "%XASM_DIR%xasm_compiler.py" %XASM_MODE% %*
) else (
  "%PYTHON_EXE%" "%XASM_DIR%xasm_compiler.py" %*
)
exit /b %ERRORLEVEL%
