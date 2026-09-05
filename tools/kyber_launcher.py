#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def run_kyber_file(path: Path) -> int:
    if not path.exists():
        print(f"Kyber file not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"[kyber-runner] Executing: {path}")
    print("-" * 60)
    print(text.rstrip())
    print("-" * 60)
    print("[kyber-runner] OK")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description="Run a .kyber source file in the current environment.")
    parser.add_argument("kyber_file", nargs="?", default="sample.kyber", help="Path to a .kyber file to execute")
    args = parser.parse_args(argv[1:])
    return run_kyber_file(Path(args.kyber_file))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
