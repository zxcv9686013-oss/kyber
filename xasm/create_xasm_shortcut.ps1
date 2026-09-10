# Create a Desktop shortcut to the XASM compiler (xasm_compiler.py)
# Usage: run this script from PowerShell (no admin required). It will create
# a shortcut named "XASM Compiler.lnk" on the current user's Desktop.

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$xasmCompiler = Join-Path $scriptRoot 'xasm_compiler.py'

# Resolve python: prefer KYBER_PYTHON env var, then python on PATH
$python = $env:KYBER_PYTHON
if (-not $python -or -not (Test-Path $python)) {
    try {
        $cmd = Get-Command python -ErrorAction Stop
        $python = $cmd.Source
    } catch {
        Write-Error "Python not found. Set KYBER_PYTHON env var or add Python to PATH."
        exit 1
    }
}

if (-not (Test-Path $xasmCompiler)) {
    Write-Error "xasm_compiler.py not found at: $xasmCompiler"
    exit 1
}

$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'XASM Compiler.lnk'

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($lnkPath)
$shortcut.TargetPath = $python
# Quote the script path so spaces are handled; pass through any args from shortcut
$shortcut.Arguments = "`"$xasmCompiler`""
$shortcut.WorkingDirectory = Split-Path $xasmCompiler -Parent
# Try to set a sensible icon (Python exe) — user can change after creation
$shortcut.IconLocation = "$python,0"
$shortcut.Description = 'Launch XASM compiler (xasm_compiler.py)'
$shortcut.Save()

Write-Host "Created shortcut: $lnkPath"