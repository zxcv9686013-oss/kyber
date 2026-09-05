#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "KyberInstaller.exe"
STAGING = ROOT / "_installer_staging"

# Files/folders that should NOT be shipped
EXCLUDE_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "build",
    "downloads",
    "_installer_staging",
}

EXCLUDE_FILES = {
    "KyberInstaller.exe",
    "build_installer.py",
    "install_log.txt",
    "tree.txt",
}

def copy_kyber_files():
    if STAGING.exists():
        shutil.rmtree(STAGING)

    STAGING.mkdir(parents=True)

    print("Copying Kyber files...")

    for source in ROOT.rglob("*"):
        relative = source.relative_to(ROOT)

        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue

        if source.is_file() and source.name in EXCLUDE_FILES:
            continue

        destination = STAGING / relative

        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    print("Kyber files copied.")


def create_launcher():
    launcher = STAGING / "_kyber_launcher.py"

    launcher.write_text(
        r'''import os
import sys
import subprocess
from pathlib import Path

INSTALLER_DIR = Path(__file__).resolve().parent
INSTALL_SCRIPT = INSTALLER_DIR / "install.py"

print()
print("=" * 70)
print("                 KYBER INSTALLER")
print("=" * 70)
print()
print("Kyber installer is starting...")
print()

if not INSTALL_SCRIPT.exists():
    print("ERROR: install.py was not found.")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    result = subprocess.run(
        [sys.executable, str(INSTALL_SCRIPT)],
        cwd=str(INSTALLER_DIR)
    )

    print()

    if result.returncode == 0:
        print("=" * 70)
        print("                 KYBER INSTALLED")
        print("=" * 70)
        print()
    else:
        print("Kyber installation failed.")
        print(f"Exit code: {result.returncode}")

except Exception as exc:
    print("Installer error:")
    print(exc)

input("Press Enter to close...")
''',
        encoding="utf-8"
    )

    return launcher


def build_exe():
    launcher = create_launcher()

    print()
    print("Building KyberInstaller.exe...")
    print()

    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller is not installed.")
        print()
        print("Run:")
        print("python -m pip install pyinstaller")
        sys.exit(1)

    dist = ROOT / "_installer_dist"
    work = ROOT / "_installer_work"

    if dist.exists():
        shutil.rmtree(dist)

    if work.exists():
        shutil.rmtree(work)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "KyberInstaller",
        "--console",
        "--clean",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        str(launcher),
    ]

    subprocess.run(command, check=True)

    built = dist / "KyberInstaller.exe"

    if not built.exists():
        print("ERROR: PyInstaller did not create the EXE.")
        sys.exit(1)

    shutil.copy2(built, OUTPUT)

    print()
    print("=" * 70)
    print("              KYBER INSTALLER CREATED")
    print("=" * 70)
    print()
    print(f"Installer: {OUTPUT}")
    print()

    launcher.unlink(missing_ok=True)

    if dist.exists():
        shutil.rmtree(dist)

    if work.exists():
        shutil.rmtree(work)

    if STAGING.exists():
        shutil.rmtree(STAGING)

    print("Temporary files cleaned.")


if __name__ == "__main__":
    copy_kyber_files()
    build_exe()