import os
import sys
import shlex
import shutil
import zipfile
import json
import subprocess
import urllib.request
import io
from pathlib import Path

VERSION = "1.2.0"

PACKAGE_DIR = os.path.join(os.path.expanduser("~"), ".nforce", "packages")
BUILD_DIR = os.path.join(os.path.expanduser("~"), ".nforce", "builds")
PLUGIN_DIR = os.path.join(os.path.expanduser("~"), ".nforce", "plugins")
OBJ_DIR = os.path.join(os.path.expanduser("~"), ".nforce", "obj")

REGISTRY_FILE = os.path.join(PACKAGE_DIR, "registry.json")
PLUGIN_REGISTRY_FILE = os.path.join(PLUGIN_DIR, "plugins.json")
BUILD_CACHE = ".nforce_build.json"

# Default location of the Ninja interpreter (NINJA--COMPILER.py),
# used when creating "ninja"-language plugins. Override per-plugin
# with the --interpreter flag, or globally with the
# NFORCE_NINJA_INTERPRETER environment variable.
DEFAULT_NINJA_INTERPRETER = os.path.join(
    os.path.expanduser("~"), ".nforce", "ninja", "NINJA--COMPILER.py"
)

PACKAGE_MARKER = "pkg_kyber.nforce"

os.makedirs(PACKAGE_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(PLUGIN_DIR, exist_ok=True)
os.makedirs(OBJ_DIR, exist_ok=True)

# ── Auto-register bundled Kyber packages ─────────────────────
# These ship with Kyber and are always available without a
# separate install step. They are registered into the local
# registry pointing at the bundled path so `nforce list` shows
# them and `nforce install <name>` works normally.
_KYBER_ROOT_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "packages",
)
_BUNDLED_PACKAGES = [
    "kyber.math",
    "kyber.io",
    "kyber.gpu",
    "kyber.net",
]

def _auto_register_bundled():
    reg = load_registry()
    changed = False
    for pkg in _BUNDLED_PACKAGES:
        if pkg not in reg:
            pkg_path = os.path.normpath(
                os.path.join(_KYBER_ROOT_DEFAULT, pkg)
            )
            if os.path.exists(pkg_path):
                reg[pkg] = pkg_path
                changed = True
    if changed:
        save_registry(reg)

try:
    _auto_register_bundled()
except Exception:
    pass


# ==========================
# HELP
# ==========================

def help_menu():
    print(f"""
NForce-Kyber {VERSION}

Package commands:

    scan
    pack <folder>
    register <name> <url>
    unregister <name>
    list
    build

Plugin commands:

    plugin install <name> <extension> <command...>
    plugin remove <name>
    plugin list
    run <plugin> <file>

Other:

    shell       start an interactive NForce shell
    help

Plugin commands shell out to whatever tool you register.
{{input}} and {{output}} in the command template are substituted
with the source path and a target .obj path. Mode is
auto-detected: if the command uses {{output}}, the plugin is
expected to produce a real .obj file ("compile" mode); if it
doesn't, the plugin just runs the source directly ("run" mode).

Example - a real compiler (produces an .obj):

    nforce plugin install kyber .kyber kyberc {{input}}
    nforce run kyber main.kyber

Example - Ninja (an interpreter, no .obj, "run" mode):

    nforce plugin install ninja .ninja python3 /path/to/NINJA--COMPILER.py {{input}}
    nforce run ninja main.ninja
""")


# ==========================
# REGISTRY
# ==========================

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(reg):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=4)


def register(name, url):
    reg = load_registry()
    reg[name] = url
    save_registry(reg)
    print(f"Registered {name} ✔")


def unregister(name):
    reg = load_registry()

    if name in reg:
        del reg[name]
        save_registry(reg)
        print(f"Unregistered {name} ✔")
    else:
        print("Package not found")


def list_packages():
    reg = load_registry()

    if not reg:
        print("No packages registered")
        return

    for name, source in reg.items():
        print(f"{name} → {source}")


# ==========================
# SCAN
# ==========================

def scan():
    print("Scanning Kyber project...\n")

    found = []

    for root, dirs, files in os.walk("."):
        if PACKAGE_MARKER in files:
            found.append(root)

    if not found:
        print("No Kyber packages found")
        return

    for folder in found:
        print("✔", os.path.abspath(folder))


# ==========================
# PACK
# ==========================

def pack(folder):
    folder_path = Path(folder)

    if not folder_path.exists():
        print("Folder not found")
        return

    zip_name = folder_path.name + ".zip"
    zip_path = os.path.join(BUILD_DIR, zip_name)

    print("Packing:", folder_path.name)

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for file in folder_path.rglob("*"):

            if not file.is_file():
                continue

            if "__pycache__" in file.parts:
                continue

            if ".git" in file.parts:
                continue

            z.write(
                file,
                file.relative_to(folder_path)
            )

    reg = load_registry()
    reg[folder_path.name] = zip_path
    save_registry(reg)

    print("Packed ✔ ->", zip_path)


# ==========================
# INSTALL (packages)
# ==========================

def install(name):
    reg = load_registry()

    if name not in reg:
        print("Unknown package:", name)
        return False

    source = reg[name]

    print(f"Installing {name}...")

    install_path = os.path.join(PACKAGE_DIR, name)

    try:

        os.makedirs(install_path, exist_ok=True)

        # Local ZIP
        if os.path.exists(source):

            with zipfile.ZipFile(source, "r") as z:
                z.extractall(install_path)

        # Remote ZIP
        else:

            response = urllib.request.urlopen(source)
            data = response.read()

            with zipfile.ZipFile(
                io.BytesIO(data),
                "r"
            ) as z:
                z.extractall(install_path)

        print(f"{name} installed ✔")

        return True

    except Exception as e:

        print("Install failed:", e)

        return False


# ==========================
# KYBER IMPORT PARSER
# ==========================

def find_imports(file):

    imports = []

    if not os.path.exists(file):
        return imports

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line.startswith("IMPORT(") and line.endswith(")"):

                name = line[7:-1].strip()

                if name:
                    imports.append(name)

    return imports


# ==========================
# RESOLVE PACKAGE
# ==========================

def resolve(name):

    local_path = os.path.join(
        PACKAGE_DIR,
        name
    )

    if os.path.exists(local_path):
        return local_path

    print(f"{name} not installed.")
    print("Attempting automatic installation...")

    if not install(name):
        return None

    return local_path


# ==========================
# BUILD
# ==========================

def build():

    print("NForce-Kyber Build\n")

    entry = "main.kyber"

    if not os.path.exists(entry):

        print("main.kyber not found")
        return

    imports = find_imports(entry)

    print("Imports:", imports)

    resolved = {}

    for package in imports:

        path = resolve(package)

        if not path:

            print("Build failed")
            return

        resolved[package] = path

    with open(
        BUILD_CACHE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "entry": entry,
                "packages": resolved
            },
            f,
            indent=4
        )

    print("\nAll dependencies ready ✔")
    print("Launching Kyber compiler...\n")

    # Kyber compiler
    subprocess.run(
        [
            sys.executable,
            "KYBER_COMPILER.py",
            entry
        ]
    )


# ==========================
# PLUGINS
#
# A plugin describes how to turn a language's source file into
# an .obj file (or just run it directly), so a backend
# (kyberlink.exe, another linker, or anything else consuming
# .obj files) can use it. Each plugin is a name, the file
# extension it handles, and a command template with {input} /
# {output} placeholders — the same idea as Kyber's own #extern
# blocks, just at the package-manager level.
#
# There are two ways to register a plugin:
#
#   plugin install  - fully manual: you supply the exact shell
#                      command yourself. Works for literally any
#                      toolchain, in any language.
#
#   plugin create    - guided: you point at a plugin's entry
#                      source file and its language (ninja, c,
#                      cpp, or rust), and NForce builds/wires it
#                      up using sane defaults for that language.
#                      For "another supported language" not
#                      listed here, fall back to plugin install.
# ==========================

# Known languages a "plugin create" entry file can be written in,
# and how to turn that entry file into something runnable.
#
#   compiled  - build the entry file into a binary once (at
#               creation time), then invoke that binary per file
#               with {input}/{output} substituted as argv.
#
#   not compiled (ninja) - no build step. NINJA--COMPILER.py only
#               accepts a single script path today (it doesn't
#               forward extra argv into the script), so a
#               ninja-language plugin runs as a fixed script
#               rather than receiving {input}/{output} per call.
PLUGIN_LANGUAGES = {
    "c": {
        "compiled": True,
        "extension": ".c",
        "build": ["clang", "-O2", "{entry}", "-o", "{bin}"],
        "build_fallback": ["gcc", "-O2", "{entry}", "-o", "{bin}"],
    },
    "cpp": {
        "compiled": True,
        "extension": ".cpp",
        "build": ["clang++", "-O2", "{entry}", "-o", "{bin}"],
        "build_fallback": ["g++", "-O2", "{entry}", "-o", "{bin}"],
    },
    "rust": {
        "compiled": True,
        "extension": ".rs",
        "build": ["rustc", "-O", "{entry}", "-o", "{bin}"],
        "build_fallback": None,
    },
    "ninja": {
        "compiled": False,
        "extension": ".ninja",
    },
}


def load_plugins():
    if os.path.exists(PLUGIN_REGISTRY_FILE):
        with open(PLUGIN_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_plugins(plugins):
    with open(PLUGIN_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(plugins, f, indent=4)


def plugin_install(name, extension, command_parts):

    if not command_parts:
        print("Usage: plugin install <name> <extension> <command...>")
        return

    if not extension.startswith("."):
        extension = "." + extension

    command = " ".join(command_parts)

    if "{input}" not in command:
        print("Warning: command has no {input} placeholder")

    # Plugins fall into two modes, auto-detected from the
    # command template:
    #
    #   "compile" - the command has an {output} placeholder and
    #               is expected to produce a real .obj file
    #               (e.g. a clang-backed compiler like Kyber's).
    #
    #   "run"     - no {output} placeholder, so nothing is
    #               expected to land on disk; the plugin just
    #               executes the source directly (e.g. the
    #               Ninja interpreter, which has no compile
    #               step at all).
    mode = "compile" if "{output}" in command else "run"

    plugins = load_plugins()

    plugins[name] = {
        "extension": extension,
        "command": command,
        "mode": mode
    }

    save_plugins(plugins)

    print(f"Plugin '{name}' installed ✔ (*{extension}, mode={mode}) -> {command}")


def _find_toolchain(primary, fallback):
    """Returns the argv list to use for a build step: the primary
    toolchain if it's on PATH, else the fallback if that's on PATH,
    else None."""

    if shutil.which(primary[0]):
        return primary

    if fallback and shutil.which(fallback[0]):
        print(f"Note: '{primary[0]}' not found, using '{fallback[0]}' instead")
        return fallback

    return None


def plugin_create(name, extension, language, entry_file, interpreter=None):
    """Guided plugin registration: point at an entry source file
    and a language, and NForce builds/wires it up. This is what
    lets someone implement an NForce plugin in Ninja, C, C++, or
    Rust without hand-writing the shell command themselves."""

    language = language.lower()

    if language not in PLUGIN_LANGUAGES:
        supported = ", ".join(sorted(PLUGIN_LANGUAGES))
        print(f"Unsupported language: {language}")
        print(f"Supported by 'plugin create': {supported}")
        print("For another language or toolchain, use 'plugin install' instead —")
        print("it accepts any shell command, so it works for anything.")
        return

    if not os.path.exists(entry_file):
        print("Entry file not found:", entry_file)
        return

    if not extension.startswith("."):
        extension = "." + extension

    spec = PLUGIN_LANGUAGES[language]

    plugin_home = os.path.join(PLUGIN_DIR, name)
    os.makedirs(plugin_home, exist_ok=True)

    entry_copy = os.path.join(
        plugin_home,
        "entry" + spec["extension"]
    )
    shutil.copyfile(entry_file, entry_copy)

    if spec["compiled"]:

        args = _find_toolchain(spec["build"], spec.get("build_fallback"))

        if not args:
            wanted = spec["build"][0]
            print(f"Could not find a '{wanted}' toolchain on PATH.")
            print(f"Install {wanted} (or a supported fallback) and try again.")
            return

        bin_path = os.path.join(
            plugin_home,
            "bin" + (".exe" if os.name == "nt" else "")
        )

        command_args = [
            part.format(entry=entry_copy, bin=bin_path)
            for part in args
        ]

        print(f"\n[NForce Plugin Create: {name}]")
        print("Language:", language)
        print("Building:", " ".join(command_args))

        result = subprocess.run(command_args)

        if result.returncode != 0:
            print(f"\nBuild failed (exit code {result.returncode})")
            return

        if not os.path.exists(bin_path):
            print("\nBuild reported success, but no binary was produced at:")
            print(bin_path)
            return

        print("Built ✔ ->", bin_path)

        command = f'"{bin_path}" {{input}} {{output}}'

    else:

        # ninja: no build step, entry file is run as-is.
        ninja_interpreter = (
            interpreter
            or os.environ.get("NFORCE_NINJA_INTERPRETER")
            or DEFAULT_NINJA_INTERPRETER
        )

        if not os.path.exists(ninja_interpreter):
            print("Warning: Ninja interpreter not found at:")
            print(f"  {ninja_interpreter}")
            print("Set --interpreter, or the NFORCE_NINJA_INTERPRETER env var, ")
            print("to point at NINJA--COMPILER.py.")

        command = f'python3 "{ninja_interpreter}" "{entry_copy}"'

        print(f"\n[NForce Plugin Create: {name}]")
        print("Language: ninja")
        print("Note: NINJA--COMPILER.py only takes a fixed script path today,")
        print("so this plugin runs 'entry.ninja' as-is on every invocation —")
        print("it does not receive {input}/{output} per call.")

    mode = "compile" if "{output}" in command else "run"

    plugins = load_plugins()

    plugins[name] = {
        "extension": extension,
        "command": command,
        "mode": mode,
        "language": language
    }

    save_plugins(plugins)

    print(f"\nPlugin '{name}' created ✔ (*{extension}, mode={mode}, language={language})")
    print("Command:", command)


def plugin_remove(name):

    plugins = load_plugins()

    if name in plugins:
        del plugins[name]
        save_plugins(plugins)
        print(f"Plugin '{name}' removed ✔")
    else:
        print("Plugin not found:", name)


def plugin_list():

    plugins = load_plugins()

    if not plugins:
        print("No plugins installed")
        return

    for name, info in plugins.items():
        mode = info.get("mode", "compile" if "{output}" in info["command"] else "run")
        lang = info.get("language")
        tag = f", language={lang}" if lang else ""
        print(f"{name}  (*{info['extension']}, mode={mode}{tag})  ->  {info['command']}")


def run_plugin(name, file):

    plugins = load_plugins()

    if name not in plugins:
        print(f"Unknown plugin: {name}")
        print("Install it first with: plugin install <name> <extension> <command...>")
        return False

    if not os.path.exists(file):
        print("Source file not found:", file)
        return False

    info = plugins[name]
    mode = info.get("mode", "compile" if "{output}" in info["command"] else "run")

    source_path = Path(file)
    output_path = os.path.join(
        OBJ_DIR,
        source_path.stem + ".obj"
    )

    command = info["command"].format(
        input=str(source_path),
        output=output_path
    )

    print(f"\n[NForce Plugin: {name}]  mode={mode}")
    print("Source:", source_path)
    if mode == "compile":
        print("Output:", output_path)
    print("Command:", command)
    print()

    try:
        args = shlex.split(command)
    except ValueError as e:
        print("Could not parse plugin command:", e)
        return False

    result = subprocess.run(args)

    if result.returncode != 0:
        print(f"\nPlugin '{name}' failed (exit code {result.returncode})")
        return False

    if mode == "compile":

        if not os.path.exists(output_path):
            print("\nPlugin reported success, but no .obj was produced at:")
            print(output_path)
            return False

        print(f"\n{name} build succeeded ✔ ->", output_path)

    else:

        print(f"\n{name} run finished ✔ (exit code 0)")

    return True


# ==========================
# COMMAND DISPATCH
#
# Shared by both the one-shot CLI (main) and the interactive
# shell, so "nforce run ninja main.ninja" and typing
# "run ninja main.ninja" inside the shell behave identically.
# ==========================

def run_command(argv):

    if not argv:
        help_menu()
        return

    cmd = argv[0].lower()

    if cmd == "scan":

        scan()

    elif cmd == "pack":

        if len(argv) < 2:
            print("Usage: pack <folder>")
            return

        pack(argv[1])

    elif cmd == "register":

        if len(argv) < 3:
            print("Usage: register <name> <url>")
            return

        register(argv[1], argv[2])

    elif cmd == "unregister":

        if len(argv) < 2:
            print("Usage: unregister <name>")
            return

        unregister(argv[1])

    elif cmd == "list":

        list_packages()

    elif cmd == "build":

        build()

    elif cmd == "plugin":

        if len(argv) < 2:
            print("Usage: plugin <install|remove|list> ...")
            return

        sub = argv[1].lower()

        if sub == "install":

            if len(argv) < 4:
                print("Usage: plugin install <name> <extension> <command...>")
                return

            plugin_install(argv[2], argv[3], argv[4:])

        elif sub == "remove":

            if len(argv) < 3:
                print("Usage: plugin remove <name>")
                return

            plugin_remove(argv[2])

        elif sub == "list":

            plugin_list()

        else:

            print("Unknown plugin command:", sub)

    elif cmd == "run":

        if len(argv) < 3:
            print("Usage: run <plugin> <file>")
            return

        run_plugin(argv[1], argv[2])

    elif cmd == "help":

        help_menu()

    elif cmd in ("exit", "quit"):

        # Only meaningful inside the shell; harmless no-op
        # from the one-shot CLI.
        pass

    else:

        print("Unknown command:", cmd)


# ==========================
# SHELL
# ==========================

def shell():

    print(f"NForce Shell {VERSION}")
    print("Type 'help' for commands, 'exit' to quit.\n")

    while True:

        try:
            line = input("nforce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        try:
            argv = shlex.split(line)
        except ValueError as e:
            print("Parse error:", e)
            continue

        if argv[0].lower() in ("exit", "quit"):
            break

        run_command(argv)


# ==========================
# MAIN
# ==========================

def main():

    if len(sys.argv) < 2:

        help_menu()
        return

    if sys.argv[1].lower() == "shell":

        shell()
        return

    run_command(sys.argv[1:])


if __name__ == "__main__":
    main()
