"""
project_env.py

YAML-based project.env loader with inheritance, folder creation, and Fortran include installer.

Features:
- Searches for project.env.yaml / project.env.yml / project.env in current working directory and parents.
- Loads YAML (requires PyYAML). If PyYAML missing, prints a warning and returns None.
- Applies `inherit` recursively (relative paths supported).
- Flattens config keys and exports them as environment variables named KYBER_<UPPERCASE_KEYS_WITH_UNDERSCORES>.
  e.g. project.name -> KYBER_PROJECT_NAME
- Creates project folders specified via `folders` (list or mapping) or under `project.folders`.
  When folders are provided as a mapping, creates KYBER_FOLDER_<KEY> environment variables for each.
- If `fortran.install: true` and `fortran.include_dirs` is provided, copies header/source files (*.h, *.mod, *.f90, *.f95) to the install dir.

This helper is defensive so importing/running it when PyYAML is unavailable won't abort kyberc.
"""
from pathlib import Path
import os
import shutil
import sys

SEARCH_FILENAMES = ["project.env.yaml", "project.env.yml", "project.env"]


def _find_project_file(start: Path = None):
    start = Path(start or Path.cwd()).resolve()
    for d in [start] + list(start.parents):
        for fname in SEARCH_FILENAMES:
            candidate = d / fname
            if candidate.exists():
                return candidate
    return None


def _load_yaml_file(path: Path):
    try:
        import yaml
    except Exception:
        print("[kyber] PyYAML not installed; skipping project.env load. Install PyYAML to enable project.env support.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dicts(parent: dict, child: dict) -> dict:
    # shallow-merge dicts; nested dicts are merged recursively
    out = dict(parent)
    for k, v in (child or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _copy_fortran_includes(cfg: dict, base_dir: Path):
    fortran = cfg.get("fortran")
    if not isinstance(fortran, dict):
        return
    install = fortran.get("install")
    if not install:
        return
    include_dirs = fortran.get("include_dirs") or []
    if isinstance(include_dirs, str):
        include_dirs = [include_dirs]
    install_dir = fortran.get("install_dir") or os.environ.get("KYBER_FORTRAN_INSTALL_DIR") or (base_dir / "build" / "include")
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["*.h", "*.mod", "*.f90", "*.f95", "*.for", "*.f"]
    copied = 0
    for src in include_dirs:
        srcp = (base_dir / src).resolve()
        if not srcp.exists():
            continue
        if srcp.is_file():
            shutil.copy2(srcp, install_dir / srcp.name)
            copied += 1
            continue
        for pattern in patterns:
            for f in srcp.rglob(pattern):
                try:
                    shutil.copy2(f, install_dir / f.name)
                    copied += 1
                except Exception:
                    pass
    if copied:
        print(f"[kyber] Installed {copied} Fortran/header files to {install_dir}")


def _create_project_folders(cfg: dict, base_dir: Path):
    """Create folders described in config.

    Supported shapes:
    - folders: ["src", "tests", "assets"]  # list of paths
    - folders:
        src: "src"
        tests: "tests"

    Also accepts project.folders similarly.
    """
    created = 0
    folders = cfg.get("folders")
    if not folders:
        project = cfg.get("project") or {}
        folders = project.get("folders")
    if not folders:
        return created
    # folders can be list or dict
    if isinstance(folders, dict):
        for key, rel in folders.items():
            try:
                p = (base_dir / rel).resolve()
                p.mkdir(parents=True, exist_ok=True)
                env_name = f"KYBER_FOLDER_{str(key).upper()}"
                os.environ[env_name] = str(p)
                created += 1
            except Exception:
                pass
    elif isinstance(folders, (list, tuple)):
        for rel in folders:
            try:
                p = (base_dir / rel).resolve()
                p.mkdir(parents=True, exist_ok=True)
                created += 1
            except Exception:
                pass
    else:
        # single string
        try:
            p = (base_dir / str(folders)).resolve()
            p.mkdir(parents=True, exist_ok=True)
            created += 1
        except Exception:
            pass
    if created:
        print(f"[kyber] Created {created} project folders under {base_dir}")
    return created


def _persist_env_file(flat: dict, base_dir: Path, filename: str = ".kyber.env"):
    """Write a simple KEY=VALUE env file containing the flattened KYBER_ entries.

    The file is written to base_dir/filename. Values are shell-quoted when they contain spaces or special chars.
    """
    out = []
    out.append("# Auto-generated Kyber env file. Do not commit unless intentional.")
    out.append("# Generated by compiler.project_env\n")
    for k, v in (flat or {}).items():
        env_name = "KYBER_" + k.upper().replace('.', '_')
        sval = str(v)
        # simple quoting for safety
        if any(c in sval for c in [' ', '"', "'", '\\', '$']):
            sval = '"' + sval.replace('"', '\\"') + '"'
        out.append(f"{env_name}={sval}")
    target = (base_dir / filename)
    try:
        with open(target, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(out) + '\n')
        print(f"[kyber] Wrote env file: {target}")
    except Exception as e:
        print(f"[kyber] Failed to write env file {target}: {e}")


def load_project_env(path: str = None):
    """Load project env, apply inheritance, create folders, copy Fortran includes, and export flattened keys to environment variables.

    Returns the final merged dict (or None if no file found / yaml missing).
    """
    base_dir = Path.cwd()
    file_path = None
    if path:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = (base_dir / file_path).resolve()
        if not file_path.exists():
            print(f"[kyber] project.env not found at {file_path}")
            return None
    else:
        file_path = _find_project_file(base_dir)
        if not file_path:
            return None
    # load chain of inherited files
    merged = {}
    seen = set()
    cur = file_path
    while cur and cur.exists() and str(cur) not in seen:
        seen.add(str(cur))
        data = _load_yaml_file(cur)
        if data is None:
            return None
        parent_path = data.get("inherit")
        # remove inherit before merging
        child = dict(data)
        child.pop("inherit", None)
        merged = _merge_dicts(merged, child)
        if parent_path:
            # resolve relative parent path
            parent = Path(parent_path)
            if not parent.is_absolute():
                parent = (cur.parent / parent).resolve()
            cur = parent
        else:
            break
    # flatten and export
    flat = _flatten(merged)
    for k, v in flat.items():
        env_name = "KYBER_" + k.upper().replace('.', '_')
        os.environ[env_name] = str(v)

    # persist env to a file (.kyber.env) when requested or by default
    try:
        persist = merged.get('persist_env') if isinstance(merged, dict) else None
        # default to True unless explicitly false
        if persist is None:
            persist = True
        if persist:
            _persist_env_file(flat, file_path.parent)
    except Exception:
        pass

    # create folders
    try:
        _create_project_folders(merged, file_path.parent)
    except Exception as e:
        print(f"[kyber] Failed to create project folders: {e}")
    # support fortran install
    try:
        _copy_fortran_includes(merged, file_path.parent)
    except Exception as e:
        print(f"[kyber] Fortran install step failed: {e}")
    return merged


if __name__ == "__main__":
    print("This module is a helper for kyber compiler. Import and call load_project_env() from kyberc.")
