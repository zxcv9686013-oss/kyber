@echo off
setlocal
set "KYBER_ROOT=%~dp0"
set "PYTHON_EXE=%KYBER_PYTHON%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=C:\Users\Admin\PyCharmMiscProject\.venv\Scripts\python.exe"

if "%1"=="run" (
    "%PYTHON_EXE%" "%KYBER_ROOT%compiler\kyber__compiler.PY" %*
    exit /b %errorlevel%
)

if "%1"=="build" (
    set "COMPILER_OBJ=C:\Users\Admin\kyber\compiler\kyber_compiler.obj"
    set "CLANG_EXE=C:\Users\Admin\kyber\bin\clang++.exe"
    set "COMPILER_SRC=C:\Users\Admin\kyber\compiler\kyber__compiler.cpp"
    echo [Kyber CLI] Compiling frontend source module safely with Clang...
    "%CLANG_EXE%" -O3 -std=c++17 -c "%COMPILER_SRC%" -o "%COMPILER_OBJ%"
    if %errorlevel% equ 0 (
        echo ✅ [Kyber CLI] Frontend code module compiled successfully to %COMPILER_OBJ%
    )
    goto end
)

if "%1"=="" (
    echo Usage:
    echo   kyber build
    echo   kyber run file.kyber [--target gpu|cpu|npu|tpu|lpu] [--json]
    goto end
)

"%PYTHON_EXE%" "%KYBER_ROOT%compiler\kyber__compiler.PY" %*
exit /b %errorlevel%

:end
