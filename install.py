#!/usr/bin/env python3
"""
KYBER FULL INSTALLER

Installs/configures the complete bundled Kyber environment.

Usage:
    python install.py
    python install.py --prefix "C:\\Users\\Admin\\kyber (7)\\kyber"
    python install.py --check
    python install.py --tree
    python install.py --uninstall
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SRC_ROOT = Path(__file__).resolve().parent

DEFAULT_PREFIX = Path(
    r"C:\Users\Admin\kyber (7)\kyber"
)

PYCHARM_PYTHON = Path(
    r"C:\Users\Admin\PyCharmMiscProject\.venv\Scripts\python.exe"
)

REQUIRED_PYTHON_PACKAGES = {
    "llvmlite": "llvmlite",
    "yaml": "pyyaml",
}

INSTALL_ITEMS = [
    "compiler",
    "nforce",
    "tools",
    "bin",
    "experiments",
    "packages",
    "third_party",
    "ngpu",
    "npu",
    "runtime",
    "xasm",

    "CUDA",
    "XLA",

    "project.env.yml",
    ".kyber.env",
    ".kyber.env.ps1",
    ".kyber.env.sh",

    "kyber.bat",
    "kyber.exe",
    "kyber-runner.exe",

    "LICENSE.txt",

    "sample.kyber",
    "class_sample.kyber",
    "matrix_demo.kyber",
    "oop_demo.kyber",
    "ai_train_demo.ngpu",
]

DIRECTORIES = [
    "bin",
    "lib",
    "include",
    "include/kyber",
    "include/fortran",
    "tools",

    "sdk",
    "sdk/cuda",
    "sdk/directml",
    "sdk/tpu",
    "sdk/lpu",

    "runtimes",
    "runtimes/cpu",
    "runtimes/gpu",
    "runtimes/npu",
    "runtimes/tpu",
    "runtimes/lpu",

    "vendor",
    "vendor/cuda",
    "vendor/directml",
    "vendor/tpu",
    "vendor/lpu",

    "docker",
    "docker/images",
    "docker/compose",

    "build",
    "build/.obj",

    "logs",
    "cache",
    "tmp",

    "third_party",
]


# ============================================================
# OUTPUT
# ============================================================

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def ok(message: str):
    print(f"{GREEN}[OK]{RESET} {message}")


def info(message: str):
    print(f"{CYAN}[..]{RESET} {message}")


def warn(message: str):
    print(f"{YELLOW}[!!]{RESET} {message}")


def error(message: str):
    print(f"{RED}[ERR]{RESET} {message}")


# ============================================================
# PATH HELPERS
# ============================================================

def same_path(a: Path, b: Path) -> bool:
    """
    Safely determine whether two paths refer to the same location.
    """

    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.normcase(
            os.path.abspath(str(a))
        ) == os.path.normcase(
            os.path.abspath(str(b))
        )


def normalize_path_string(path: str) -> str:
    return os.path.normcase(
        os.path.normpath(
            os.path.abspath(path)
        )
    )


# ============================================================
# PYTHON
# ============================================================

def get_python() -> str:
    """
    Prefer the configured PyCharm virtual environment.
    Otherwise use the Python running this installer.
    """

    if PYCHARM_PYTHON.exists():
        return str(PYCHARM_PYTHON)

    return sys.executable


def package_installed(
    python: str,
    module_name: str,
) -> bool:
    """
    Check the package using the selected Python interpreter,
    not the Python interpreter running install.py.
    """

    try:
        result = subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib.util; "
                    f"raise SystemExit("
                    f"0 if importlib.util.find_spec("
                    f"'{module_name}') else 1)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return result.returncode == 0

    except Exception:
        return False


def install_python_packages(python: str):
    print()
    info("Installing Python dependencies...")

    for module, package in REQUIRED_PYTHON_PACKAGES.items():

        if package_installed(python, module):
            ok(f"{package} already installed")
            continue

        info(f"Installing {package}...")

        result = subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                package,
            ],
            text=True,
        )

        if result.returncode == 0:
            ok(f"{package} installed")
        else:
            error(f"Failed to install {package}")


# ============================================================
# FILE INSTALLATION
# ============================================================

def copy_path(
    source: Path,
    destination: Path,
):
    """
    Copy files/directories.

    IMPORTANT:
    If source and destination are identical, nothing is copied.
    This allows the installer to safely run directly inside the
    Kyber directory.
    """

    if same_path(source, destination):
        ok(f"Already installed: {source.name}")
        return

    if source.is_dir():

        if destination.exists():
            shutil.rmtree(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copytree(
            source,
            destination,
        )

        ok(f"Copied directory: {source.name}")

    else:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        ok(f"Copied file: {source.name}")


def copy_item(
    item: str,
    prefix: Path,
) -> bool:

    source = SRC_ROOT / item
    destination = prefix / item

    if not source.exists():

        warn(f"Not bundled: {item}")

        return False

    if same_path(source, destination):

        ok(f"Already installed: {item}")

        return True

    try:

        copy_path(
            source,
            destination,
        )

        return True

    except Exception as exc:

        error(
            f"Failed to install {item}: {exc}"
        )

        return False


def create_directories(prefix: Path):

    info("Creating Kyber directory structure...")

    for directory in DIRECTORIES:

        path = prefix / directory

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    ok("Directory structure created")


# ============================================================
# TOOL DETECTION
# ============================================================

def find_executable(*names):

    for name in names:

        found = shutil.which(name)

        if found:
            return Path(found)

    return None


def find_local_executable(
    prefix: Path,
    *relative_paths,
):

    for relative in relative_paths:

        path = prefix / relative

        if path.exists():
            return path

    return None


def detect_tools(prefix: Path):

    info("Detecting toolchains...")

    tools = {}

    tools["python"] = Path(
        get_python()
    )

    tools["ninja"] = (
        find_executable(
            "ninja",
            "ninja.exe",
        )
        or find_local_executable(
            prefix,
            "bin/ninja.exe",
            "ninja.exe",
        )
    )

    tools["cmake"] = (
        find_executable(
            "cmake",
            "cmake.exe",
        )
        or find_local_executable(
            prefix,
            "bin/cmake.exe",
            "cmake.exe",
        )
    )

    tools["clang"] = (
        find_executable(
            "clang++",
            "clang++.exe",
        )
        or find_local_executable(
            prefix,
            "bin/clang++.exe",
            "bin/clang.exe",
        )
    )

    tools["nasm"] = (
        find_executable(
            "nasm",
            "nasm.exe",
        )
        or find_local_executable(
            prefix,
            "bin/nasm.exe",
            "xasm/nasm.exe",
        )
    )

    tools["nvcc"] = (
        find_executable(
            "nvcc",
            "nvcc.exe",
        )
        or find_local_executable(
            prefix,
            "CUDA/bin/nvcc.exe",
        )
    )

    return tools


# ============================================================
# CUDA
# ============================================================

def detect_cuda(prefix: Path):

    candidates = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("KYBER_CUDA_ROOT"),

        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",

        str(prefix / "CUDA"),
    ]

    root = None

    for candidate in candidates:

        if not candidate:
            continue

        path = Path(candidate)

        if path.exists():

            root = path

            # Prefer a root that actually contains nvcc.
            nvcc_candidate = (
                path
                / "bin"
                / "nvcc.exe"
            )

            if nvcc_candidate.exists():
                break

    nvcc = None

    if root:

        nvcc_candidates = [
            root / "bin" / "nvcc.exe",
            root / "bin" / "nvcc",
            root / "nvcc.exe",
        ]

        for candidate in nvcc_candidates:

            if candidate.exists():

                nvcc = candidate
                break

    return {
        "root": str(root) if root else None,
        "nvcc": str(nvcc) if nvcc else None,
        "available": bool(root and nvcc),
    }


# ============================================================
# DIRECTML
# ============================================================

def detect_directml():

    dlls = [
        Path(r"C:\Windows\System32\DirectML.dll"),
        Path(r"C:\Windows\System32\dml.dll"),
        Path(r"C:\Windows\System32\dml1.dll"),
    ]

    for dll in dlls:

        if dll.exists():

            return {
                "dll": str(dll),
                "available": True,
            }

    return {
        "dll": None,
        "available": False,
    }


# ============================================================
# DOCKER
# ============================================================

def detect_docker():

    docker = find_executable(
        "docker",
        "docker.exe",
    )

    return {
        "binary": str(docker) if docker else None,
        "available": bool(docker),
    }


# ============================================================
# TPU / LPU
# ============================================================

def detect_vendor(
    root_names,
    file_names,
):

    roots = []

    for env_name in root_names:

        value = os.environ.get(
            env_name
        )

        if value:
            roots.append(
                Path(value)
            )

    roots.extend(
        [
            Path(r"C:\Program Files\TPU"),
            Path(r"C:\tpu"),
            Path(r"C:\Program Files\LPU"),
            Path(r"C:\lpu"),
        ]
    )

    for root in roots:

        if not root.exists():
            continue

        for filename in file_names:

            candidate = root / filename

            if candidate.exists():

                return {
                    "root": str(root),
                    "library": str(candidate),
                    "available": True,
                }

    return {
        "root": None,
        "library": None,
        "available": False,
    }


# ============================================================
# WINDOWS USER ENVIRONMENT VARIABLES
# ============================================================

def set_user_environment_variable(
    name: str,
    value: str | None,
):

    if os.name != "nt":
        return

    if not value:
        return

    try:

        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ
            | winreg.KEY_WRITE,
        )

        winreg.SetValueEx(
            key,
            name,
            0,
            winreg.REG_EXPAND_SZ,
            value,
        )

        winreg.CloseKey(key)

        ok(f"Environment variable set: {name}")

    except Exception as exc:

        warn(
            f"Could not set {name}: {exc}"
        )


def set_process_environment(
    prefix: Path,
    python: str,
    cuda,
):

    os.environ["KYBER_ROOT"] = str(prefix)
    os.environ["KYBER_PYTHON"] = str(python)

    if cuda.get("root"):
        os.environ[
            "KYBER_CUDA_ROOT"
        ] = cuda["root"]

    if cuda.get("nvcc"):
        os.environ[
            "KYBER_NVCC"
        ] = cuda["nvcc"]


# ============================================================
# ENVIRONMENT FILES
# ============================================================

def write_environment(
    prefix,
    python,
    cuda,
    directml,
    docker,
):

    env_file = prefix / ".kyber.env"

    values = {
        "KYBER_ROOT": str(prefix),
        "KYBER_PYTHON": str(python),

        "KYBER_CUDA_ROOT": cuda.get("root"),
        "KYBER_NVCC": cuda.get("nvcc"),

        "KYBER_DIRECTML_ROOT":
            directml.get("dll"),

        "KYBER_DOCKER":
            docker.get("binary"),
    }

    lines = [
        "# Auto-generated by Kyber installer",
        "# DO NOT EDIT",
        "",
    ]

    for key, value in values.items():

        if value:

            lines.append(
                f"{key}={value}"
            )

    env_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    ok(f"Created {env_file}")


def write_powershell_environment(
    prefix,
    python,
):

    file = prefix / ".kyber.env.ps1"

    content = f'''# Auto-generated by Kyber installer

$env:KYBER_ROOT = "{prefix}"
$env:KYBER_PYTHON = "{python}"

$env:PATH = "{prefix};{prefix}\\bin;$env:PATH"
'''

    file.write_text(
        content,
        encoding="utf-8",
    )

    ok("Created PowerShell environment file")


def write_bash_environment(
    prefix,
    python,
):

    file = prefix / ".kyber.env.sh"

    content = f'''# Auto-generated by Kyber installer

export KYBER_ROOT="{prefix}"
export KYBER_PYTHON="{python}"

export PATH="{prefix}:{prefix}/bin:$PATH"
'''

    file.write_text(
        content,
        encoding="utf-8",
    )

    ok("Created shell environment file")


# ============================================================
# WINDOWS PATH
# ============================================================

def add_to_windows_path(
    directory: str,
):

    if os.name != "nt":
        return

    try:

        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ
            | winreg.KEY_WRITE,
        )

        try:

            current, _ = winreg.QueryValueEx(
                key,
                "Path",
            )

        except FileNotFoundError:

            current = ""

        entries = [
            x.strip()
            for x in current.split(";")
            if x.strip()
        ]

        normalized = {
            normalize_path_string(x)
            for x in entries
        }

        normalized_directory = (
            normalize_path_string(directory)
        )

        if normalized_directory not in normalized:

            entries.append(directory)

            winreg.SetValueEx(
                key,
                "Path",
                0,
                winreg.REG_EXPAND_SZ,
                ";".join(entries),
            )

            ok(
                f"Added to PATH: {directory}"
            )

        else:

            ok(
                f"Already in PATH: {directory}"
            )

        winreg.CloseKey(key)

    except Exception as exc:

        warn(
            f"Could not modify PATH: {exc}"
        )


# ============================================================
# LAUNCHERS
# ============================================================

def create_launchers(
    prefix,
    python,
):

    compiler = (
        prefix
        / "compiler"
        / "kyber__compiler.PY"
    )

    linker = (
        prefix
        / "compiler"
        / "kyber_linker.py"
    )

    if not compiler.exists():

        warn(
            "Compiler entrypoint not found"
        )

        return False

    common = f'''@echo off
set "KYBER_ROOT={prefix}"
set "KYBER_PYTHON={python}"
set "PATH={prefix};{prefix}\\bin;%PATH%"
'''

    kyber_bat = prefix / "kyber.bat"

    kyber_bat.write_text(
        common
        + f'''"{python}" "{compiler}" %*
''',
        encoding="utf-8",
    )

    kyberc_bat = prefix / "kyberc.bat"

    kyberc_bat.write_text(
        common
        + f'''"{python}" "{compiler}" %*
''',
        encoding="utf-8",
    )

    kyberlink_bat = (
        prefix / "kyberlink.bat"
    )

    if linker.exists():

        kyberlink_bat.write_text(
            common
            + f'''"{python}" "{linker}" %*
''',
            encoding="utf-8",
        )

    env_bat = prefix / "kyber_env.bat"

    env_bat.write_text(
        common
        + '''echo Kyber environment loaded.
''',
        encoding="utf-8",
    )

    ok("Created Kyber launchers")

    return True


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(
    prefix,
    tools,
    cuda,
    directml,
    docker,
    tpu,
    lpu,
):

    manifest = {
        "name": "Kyber",
        "installer": "install.py",
        "root": str(prefix),

        "tools": {
            name:
                str(path) if path else None
            for name, path in tools.items()
        },

        "backends": {
            "cuda": cuda,
            "directml": directml,
            "docker": docker,
            "tpu": tpu,
            "lpu": lpu,
        },
    }

    target = (
        prefix
        / "runtime"
        / "backend_manifest.json"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    ok("Created backend manifest")


# ============================================================
# TREE
# ============================================================

def generate_tree(root: Path):

    output = [
        str(root)
    ]

    if not root.exists():
        return "\n".join(output)

    def walk(
        directory: Path,
        prefix="",
    ):

        try:

            entries = sorted(
                directory.iterdir(),
                key=lambda p: (
                    not p.is_dir(),
                    p.name.lower(),
                ),
            )

        except PermissionError:

            output.append(
                prefix + "└── [ACCESS DENIED]"
            )

            return

        for index, entry in enumerate(
            entries
        ):

            last = (
                index == len(entries) - 1
            )

            branch = (
                "└── "
                if last
                else "├── "
            )

            output.append(
                prefix
                + branch
                + entry.name
            )

            if entry.is_dir():

                extension = (
                    "    "
                    if last
                    else "│   "
                )

                walk(
                    entry,
                    prefix + extension,
                )

    walk(root)

    return "\n".join(output)


def write_tree(prefix):

    tree = generate_tree(prefix)

    tree_file = prefix / "tree.txt"

    tree_file.write_text(
        tree,
        encoding="utf-8",
    )

    ok(
        f"Created {tree_file}"
    )


# ============================================================
# LOGGING
# ============================================================

class Tee:

    def __init__(self, path):

        self.console = sys.__stdout__

        self.file = open(
            path,
            "w",
            encoding="utf-8",
        )

    def write(self, data):

        self.console.write(data)
        self.file.write(data)

    def flush(self):

        self.console.flush()
        self.file.flush()

    def close(self):

        try:
            self.file.close()
        except Exception:
            pass


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(
    command,
    cwd=None,
    timeout=30,
):

    try:

        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.returncode == 0

    except Exception:

        return False


# ============================================================
# SELF TEST
# ============================================================

def self_test(
    prefix,
    python,
    tools,
):

    print()
    print("=" * 72)
    print("KYBER SELF TEST")
    print("=" * 72)

    tests = []

    tests.append(
        (
            "Python",
            run_command(
                [
                    python,
                    "--version",
                ]
            ),
        )
    )

    compiler = (
        prefix
        / "compiler"
        / "kyber__compiler.PY"
    )

    tests.append(
        (
            "Compiler",
            compiler.exists(),
        )
    )

    tests.append(
        (
            "Runtime",
            (
                prefix / "runtime"
            ).exists(),
        )
    )

    tests.append(
        (
            "XASM",
            (
                prefix / "xasm"
            ).exists(),
        )
    )

    tests.append(
        (
            "NGPU",
            (
                prefix / "ngpu"
            ).exists(),
        )
    )

    tests.append(
        (
            "NPU",
            (
                prefix / "npu"
            ).exists(),
        )
    )

    tests.append(
        (
            "Ninja",
            tools.get("ninja") is not None,
        )
    )

    tests.append(
        (
            "CMake",
            tools.get("cmake") is not None,
        )
    )

    tests.append(
        (
            "Clang",
            tools.get("clang") is not None,
        )
    )

    tests.append(
        (
            "NASM/XASM",
            (
                tools.get("nasm") is not None
                or (
                    prefix / "xasm"
                ).exists()
            ),
        )
    )

    tests.append(
        (
            "CUDA",
            tools.get("nvcc") is not None,
        )
    )

    passed = 0
    required_passed = 0
    required_total = 0

    for name, result in tests:

        if result:

            ok(name)
            passed += 1

        else:

            # CUDA and external tools can be unavailable
            # without making the whole Kyber installation invalid.
            warn(
                name + " unavailable"
            )

        if name != "CUDA":

            required_total += 1

            if result:
                required_passed += 1

    print()
    print(
        f"Self-test: {passed}/{len(tests)} checks passed"
    )

    return (
        required_passed == required_total
    )


# ============================================================
# CHECK
# ============================================================

def check_installation(
    prefix: Path,
):

    prefix = prefix.resolve()

    print()
    print("=" * 72)
    print("KYBER INSTALLATION CHECK")
    print("=" * 72)

    print(
        f"Root: {prefix}"
    )

    if not prefix.exists():

        error(
            f"Kyber is not installed at {prefix}"
        )

        return False

    required = [
        "compiler",
        "nforce",
        "runtime",
        "xasm",
        "ngpu",
        "npu",
        "bin",
        "include",
        "lib",
    ]

    success = True

    for item in required:

        path = prefix / item

        if path.exists():

            ok(item)

        else:

            error(
                f"Missing: {item}"
            )

            success = False

    cuda = detect_cuda(prefix)

    if cuda["available"]:
        ok("CUDA")
    else:
        warn("CUDA unavailable")

    directml = detect_directml()

    if directml["available"]:
        ok("DirectML")
    else:
        warn("DirectML unavailable")

    docker = detect_docker()

    if docker["available"]:
        ok("Docker")
    else:
        warn("Docker unavailable")

    print()

    tree_file = prefix / "tree.txt"

    if tree_file.exists():
        ok("tree.txt")
    else:
        warn("tree.txt missing")

    manifest = (
        prefix
        / "runtime"
        / "backend_manifest.json"
    )

    if manifest.exists():
        ok("backend_manifest.json")
    else:
        warn(
            "backend_manifest.json missing"
        )

    return success


# ============================================================
# UNINSTALL
# ============================================================

def uninstall(prefix: Path):

    prefix = prefix.resolve()
    source = SRC_ROOT.resolve()

    print()
    print("=" * 72)
    print("KYBER UNINSTALLER")
    print("=" * 72)

    # VERY IMPORTANT:
    # Never delete the source tree if the installer is
    # running directly from the installation directory.

    if same_path(
        source,
        prefix,
    ):

        error(
            "UNINSTALL BLOCKED"
        )

        error(
            "The selected prefix is the same directory "
            "as the installer source."
        )

        error(
            "This would delete the entire Kyber source tree."
        )

        error(
            "Choose a different --prefix to uninstall."
        )

        return False

    if not prefix.exists():

        warn(
            f"Nothing to remove: {prefix}"
        )

        return True

    try:

        shutil.rmtree(prefix)

        ok(
            f"Removed {prefix}"
        )

        return True

    except Exception as exc:

        error(
            f"Uninstall failed: {exc}"
        )

        return False


# ============================================================
# INSTALL
# ============================================================

def install(prefix: Path):

    source = SRC_ROOT.resolve()
    prefix = prefix.resolve()

    prefix.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = prefix / "install_log.txt"

    original_stdout = sys.stdout
    logger = None

    try:

        logger = Tee(log_file)
        sys.stdout = logger

        print()
        print("=" * 72)
        print("KYBER FULL INSTALLER")
        print("=" * 72)

        print(
            f"Source : {source}"
        )

        print(
            f"Target : {prefix}"
        )

        print()

        same_install_directory = same_path(
            source,
            prefix,
        )

        if same_install_directory:

            info(
                "Source and target are the same directory."
            )

            info(
                "Self-copy protection enabled."
            )

        # ----------------------------------------------------
        # 1. DIRECTORIES
        # ----------------------------------------------------

        create_directories(prefix)

        # ----------------------------------------------------
        # 2. BUNDLED COMPONENTS
        # ----------------------------------------------------

        print()
        info(
            "Installing bundled Kyber components..."
        )

        installed = 0
        missing = 0

        for item in INSTALL_ITEMS:

            if copy_item(
                item,
                prefix,
            ):

                installed += 1

            else:

                missing += 1

        print()

        ok(
            f"Bundled components processed: {installed}"
        )

        if missing:

            warn(
                f"Bundled components missing: {missing}"
            )

        # ----------------------------------------------------
        # 3. PYTHON
        # ----------------------------------------------------

        print()

        python = get_python()

        if Path(python).exists():

            ok(
                f"Python runtime: {python}"
            )

        else:

            error(
                f"Python runtime not found: {python}"
            )

        install_python_packages(
            python
        )

        # ----------------------------------------------------
        # 4. TOOLCHAINS
        # ----------------------------------------------------

        print()

        tools = detect_tools(prefix)

        for name, path in tools.items():

            if path:

                ok(
                    f"{name}: {path}"
                )

            else:

                warn(
                    f"{name}: not found"
                )

        # ----------------------------------------------------
        # 5. BACKENDS
        # ----------------------------------------------------

        print()

        info(
            "Detecting hardware/software backends..."
        )

        cuda = detect_cuda(prefix)

        directml = detect_directml()

        docker = detect_docker()

        tpu = detect_vendor(
            [
                "TPU_ROOT",
                "KYBER_TPU_ROOT",
            ],
            [
                "libtpu.dll",
            ],
        )

        lpu = detect_vendor(
            [
                "LPU_ROOT",
                "KYBER_LPU_ROOT",
            ],
            [
                "lpu_runtime.dll",
            ],
        )

        if cuda["available"]:
            ok("CUDA backend available")
        else:
            warn("CUDA backend unavailable")

        if directml["available"]:
            ok("DirectML backend available")
        else:
            warn("DirectML backend unavailable")

        if docker["available"]:
            ok("Docker available")
        else:
            warn("Docker unavailable")

        if tpu["available"]:
            ok("TPU backend available")
        else:
            warn("TPU backend unavailable")

        if lpu["available"]:
            ok("LPU backend available")
        else:
            warn("LPU backend unavailable")

        # ----------------------------------------------------
        # 6. PROCESS ENVIRONMENT
        # ----------------------------------------------------

        print()

        info(
            "Configuring Kyber environment..."
        )

        set_process_environment(
            prefix,
            python,
            cuda,
        )

        # ----------------------------------------------------
        # 7. WINDOWS USER ENVIRONMENT
        # ----------------------------------------------------

        if os.name == "nt":

            set_user_environment_variable(
                "KYBER_ROOT",
                str(prefix),
            )

            set_user_environment_variable(
                "KYBER_PYTHON",
                str(python),
            )

            if cuda.get("root"):

                set_user_environment_variable(
                    "KYBER_CUDA_ROOT",
                    cuda["root"],
                )

            if cuda.get("nvcc"):

                set_user_environment_variable(
                    "KYBER_NVCC",
                    cuda["nvcc"],
                )

        # ----------------------------------------------------
        # 8. ENVIRONMENT FILES
        # ----------------------------------------------------

        write_environment(
            prefix,
            python,
            cuda,
            directml,
            docker,
        )

        write_powershell_environment(
            prefix,
            python,
        )

        write_bash_environment(
            prefix,
            python,
        )

        # ----------------------------------------------------
        # 9. LAUNCHERS
        # ----------------------------------------------------

        create_launchers(
            prefix,
            python,
        )

        # ----------------------------------------------------
        # 10. PATH
        # ----------------------------------------------------

        add_to_windows_path(
            str(prefix)
        )

        bin_directory = prefix / "bin"

        if bin_directory.exists():

            add_to_windows_path(
                str(bin_directory)
            )

        # ----------------------------------------------------
        # 11. MANIFEST
        # ----------------------------------------------------

        write_manifest(
            prefix,
            tools,
            cuda,
            directml,
            docker,
            tpu,
            lpu,
        )

        # ----------------------------------------------------
        # 12. TREE
        # ----------------------------------------------------

        write_tree(prefix)

        # ----------------------------------------------------
        # 13. SELF TEST
        # ----------------------------------------------------

        test_passed = self_test(
            prefix,
            python,
            tools,
        )

        # ----------------------------------------------------
        # 14. FINAL RESULT
        # ----------------------------------------------------

        print()
        print("=" * 72)

        if test_passed:

            print(
                "KYBER INSTALLATION COMPLETE"
            )

        else:

            print(
                "KYBER INSTALLATION COMPLETE "
                "WITH WARNINGS"
            )

        print("=" * 72)

        print()
        print(
            "Kyber root:"
        )

        print(
            f"  {prefix}"
        )

        print()
        print(
            "Run Kyber with:"
        )

        print(
            f'  "{prefix}\\kyber.bat" yourfile.kyber'
        )

        print()
        print(
            "Or open a NEW terminal and run:"
        )

        print(
            "  kyber yourfile.kyber"
        )

        print()
        print(
            "Environment:"
        )

        print(
            f"  KYBER_ROOT={prefix}"
        )

        print(
            f"  KYBER_PYTHON={python}"
        )

        if cuda.get("root"):

            print(
                f"  KYBER_CUDA_ROOT={cuda['root']}"
            )

        print()
        print(
            "Installation log:"
        )

        print(
            f"  {log_file}"
        )

        print()
        print(
            "Directory tree:"
        )

        print(
            f"  {prefix / 'tree.txt'}"
        )

        print()

        return test_passed

    except Exception as exc:

        error(
            f"INSTALLER FAILED: {exc}"
        )

        return False

    finally:

        sys.stdout = original_stdout

        if logger:

            logger.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Install the complete Kyber toolchain."
        )
    )

    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Kyber installation directory",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check an existing installation",
    )

    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall Kyber",
    )

    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print the Kyber directory tree",
    )

    args = parser.parse_args()

    if os.name == "nt":

        try:
            os.system("color")
        except Exception:
            pass

    prefix = args.prefix.resolve()

    if args.tree:

        print(
            generate_tree(prefix)
        )

        return

    if args.check:

        success = check_installation(
            prefix
        )

        raise SystemExit(
            0 if success else 1
        )

    if args.uninstall:

        success = uninstall(
            prefix
        )

        raise SystemExit(
            0 if success else 1
        )

    success = install(
        prefix
    )

    raise SystemExit(
        0 if success else 1
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()