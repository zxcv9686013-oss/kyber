@echo off
setlocal
rem ixasm wrapper: forward to xasm.cmd with --ixasm so typing "ixasm" launches the REPL
set "HERE=%~dp0"
"%HERE%xasm.cmd" --ixasm %*
exit /b %ERRORLEVEL%
