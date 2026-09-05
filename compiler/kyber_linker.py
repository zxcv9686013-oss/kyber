from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import lief


# ============================================================
# KYBER CONFIGURATION
# ============================================================

KYBER_ROOT = Path(r"C:\Users\Admin\kyber (7)\kyber")

BUILD_DIR = Path(os.environ.get("KYBER_BUILD_DIR", "build"))
OBJECT_DIRECTORY = KYBER_ROOT / BUILD_DIR / ".obj"
if not OBJECT_DIRECTORY.exists():
    fallback = KYBER_ROOT / ".obj"
    if fallback.exists():
        OBJECT_DIRECTORY = fallback

NFORCE_PACKAGE_DIRECTORY = Path.home() / ".nforce" / "packages"
DEFAULT_OUTPUT = KYBER_ROOT / "kyber.exe"

CUDA_TOOLKIT_DEFAULT = Path(os.environ.get(
    "KYBER_CUDA_ROOT",
    r"C:\Users\Admin\kyber (7)\kyber\CUDA"
))

KYBER_PYTHON = os.environ.get(
    "KYBER_PYTHON",
    r"C:/Users/Admin/PyCharmMiscProject/.venv/Scripts/python.exe"
)

# PE image base for 64-bit executables
IMAGE_BASE = 0x140000000

# Section alignment in the PE image
SECTION_ALIGNMENT  = 0x1000  # 4 KB
FILE_ALIGNMENT     = 0x200   # 512 B


# ============================================================
# COFF RELOCATION TYPES (AMD64)
# ============================================================

IMAGE_REL_AMD64_ABSOLUTE = 0x0000  # no-op
IMAGE_REL_AMD64_ADDR64   = 0x0001  # 64-bit VA
IMAGE_REL_AMD64_ADDR32   = 0x0002  # 32-bit VA (truncated)
IMAGE_REL_AMD64_ADDR32NB = 0x0003  # 32-bit RVA (no image base)
IMAGE_REL_AMD64_REL32    = 0x0004  # 32-bit relative (PC-relative +4 bytes)
IMAGE_REL_AMD64_REL32_1  = 0x0005
IMAGE_REL_AMD64_REL32_2  = 0x0006
IMAGE_REL_AMD64_REL32_3  = 0x0007
IMAGE_REL_AMD64_REL32_4  = 0x0008
IMAGE_REL_AMD64_REL32_5  = 0x0009
IMAGE_REL_AMD64_SECTION  = 0x000A  # 16-bit section index
IMAGE_REL_AMD64_SECREL   = 0x000B  # 32-bit offset from section start


# ============================================================
# PE SECTION FLAGS
# ============================================================

IMAGE_SCN_CNT_CODE             = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
IMAGE_SCN_MEM_EXECUTE          = 0x20000000
IMAGE_SCN_MEM_READ             = 0x40000000
IMAGE_SCN_MEM_WRITE            = 0x80000000


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SectionFragment:
    object_path: Path
    obj_section_name: str   # original name in the .obj
    pe_section_name: str    # merged PE section name (.text/.data/.rdata etc.)
    data: bytearray         # mutable so we can patch relocations in-place
    characteristics: int
    # filled in after layout:
    rva: int = 0            # RVA of this fragment in the final PE
    file_offset: int = 0    # raw file offset (for reference)


@dataclass
class SymbolDef:
    name: str
    rva: int                # absolute RVA in the final PE


# ============================================================
# CUDA / NGPU PRE-LINK
# ============================================================

def resolve_nvcc(cuda_root: Path) -> str:
    for name in ("nvcc.exe", "nvcc"):
        candidate = cuda_root / "bin" / name
        if candidate.exists():
            return str(candidate)
    return "nvcc"


def cuda_prelink(cuda_objects: list[Path], cuda_root: Path) -> Path | None:
    if not cuda_objects:
        return None

    nvcc = resolve_nvcc(cuda_root)
    tmp_dir = Path(tempfile.gettempdir())
    merged_obj = tmp_dir / "kyber_cuda_merged.obj"
    merged_obj.unlink(missing_ok=True)

    cuda_lib_dir = cuda_root / "lib" / "x64"
    if not cuda_lib_dir.exists():
        cuda_lib_dir = cuda_root / "lib"

    cmd = [nvcc, "--link"] + [str(p) for p in cuda_objects]
    cmd += ["-o", str(merged_obj)]
    if cuda_lib_dir.exists():
        cmd += ["-L", str(cuda_lib_dir)]
    cmd += ["-lcudart"]

    print("\nCUDA pre-link")
    print("=============")
    print(f"nvcc   : {nvcc}")
    print(f"output : {merged_obj}")

    try:
        rc = subprocess.run(cmd).returncode
    except FileNotFoundError:
        print("WARNING: nvcc not found — CUDA/NGPU objects skipped.", file=sys.stderr)
        return None

    if rc != 0 or not merged_obj.exists():
        print(f"WARNING: nvcc --link failed (exit {rc}) — CUDA/NGPU objects skipped.", file=sys.stderr)
        return None

    print(f"CUDA pre-link OK: {merged_obj}")
    return merged_obj


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_files(directory: Path, extension: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() == extension.lower()
    )


def collect_objects() -> tuple[list[Path], list[Path]]:
    all_objects = find_files(OBJECT_DIRECTORY, ".obj")
    cuda_prefixes = ("kyber_cuda_", "kyber_ngpu_")
    cuda_objs   = [p for p in all_objects if any(p.name.startswith(x) for x in cuda_prefixes)]
    normal_objs = [p for p in all_objects if not any(p.name.startswith(x) for x in cuda_prefixes)]
    return normal_objs, cuda_objs


def collect_libraries() -> list[Path]:
    return find_files(NFORCE_PACKAGE_DIRECTORY, ".lib")


# ============================================================
# COFF PARSING HELPERS
# ============================================================

def parse_object(path: Path):
    try:
        obj = lief.COFF.parse(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not parse COFF object {path}: {exc}") from exc
    if obj is None:
        raise RuntimeError(f"LIEF returned no COFF object for {path}")
    return obj


def _pe_section_name(obj_name: str) -> str:
    """Map a COFF section name to the PE section it merges into."""
    n = obj_name.rstrip("\x00").split("$")[0]  # strip $xyz suffixes
    if n == ".text":      return ".text"
    if n == ".rdata":     return ".rdata"
    if n in (".data", ".bss"): return ".data"
    if n.startswith(".xdata"): return ".xdata"
    if n.startswith(".pdata"):  return ".pdata"
    return n


def _section_chars(name: str) -> int:
    if name == ".text":
        return IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
    if name in (".data", ".bss"):
        return IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE
    if name == ".rdata":
        return IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
    return IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ


# ============================================================
# LAYOUT ENGINE
#
# Assigns an RVA to every SectionFragment in the order they appear.
# Returns the symbol table mapping  name -> absolute VA.
# ============================================================

def layout_fragments(
    fragments: list[SectionFragment],
    parsed_objects: list[tuple[Path, object]],
    definitions_raw: dict,   # symbol name -> (obj_idx, section_idx_1based, value)
) -> tuple[dict[str, SymbolDef], int]:
    """
    Assign RVAs to all fragments.
    Returns (symbol_defs, total_image_size).

    symbol_defs maps every defined symbol name to its final RVA.
    """

    # Group fragments by PE section name, preserving order
    pe_sections: dict[str, list[SectionFragment]] = {}
    for frag in fragments:
        pe_sections.setdefault(frag.pe_section_name, []).append(frag)

    # Assign RVAs section by section, page-aligned
    current_rva = SECTION_ALIGNMENT  # leave page 0 empty (null pointer trap)

    # Per-fragment RVA assignment
    frag_rva_map: dict[int, int] = {}  # id(frag) -> rva

    for pe_name, frags in pe_sections.items():
        for frag in frags:
            # align to 16 bytes within a section (COFF default)
            current_rva = (current_rva + 15) & ~15
            frag.rva = current_rva
            frag_rva_map[id(frag)] = current_rva
            current_rva += len(frag.data)
        # pad to section alignment at end of each PE section
        current_rva = (current_rva + SECTION_ALIGNMENT - 1) & ~(SECTION_ALIGNMENT - 1)

    total_size = current_rva

    # Build symbol RVA table
    # definitions_raw: name -> SymbolInfoRaw(obj_idx, section_idx_1based, value)
    symbol_defs: dict[str, SymbolDef] = {}

    for name, raw in definitions_raw.items():
        obj_idx, sec_idx_1, value = raw
        # find the fragment that corresponds to this (object, section)
        target_frag = None
        for frag in fragments:
            if frag.object_path == parsed_objects[obj_idx][0]:
                # match by section name — works for non-comdat sections
                if frag.obj_section_name == raw[3]:
                    target_frag = frag
                    break
        if target_frag is not None:
            symbol_defs[name] = SymbolDef(name=name, rva=target_frag.rva + value)
        # symbols without a matching fragment (absolute, etc.) are skipped

    return symbol_defs, total_size


# ============================================================
# RELOCATION PATCHING
#
# For each COFF relocation record, patch the 4 or 8 bytes at the
# relocation offset inside the fragment's data buffer.
# ============================================================

def apply_relocations(
    fragments: list[SectionFragment],
    parsed_objects: list[tuple[Path, object]],
    symbol_defs: dict[str, SymbolDef],
) -> int:
    """
    Patch all relocations in-place inside fragment.data.
    Returns the number of relocations applied.
    """
    applied = 0
    skipped = 0

    # Build a lookup: (object_path, obj_section_name) -> fragment
    frag_lookup: dict[tuple[Path, str], SectionFragment] = {}
    for frag in fragments:
        frag_lookup[(frag.object_path, frag.obj_section_name)] = frag

    for obj_path, obj in parsed_objects:
        for section in obj.sections:
            sec_name = section.name.rstrip("\x00")
            key = (obj_path, sec_name)
            frag = frag_lookup.get(key)
            if frag is None:
                continue

            for reloc in section.relocations:
                try:
                    sym_name = reloc.symbol.name
                except Exception:
                    skipped += 1
                    continue

                if sym_name not in symbol_defs:
                    # unresolved — leave as-is (will be 0 / invalid)
                    skipped += 1
                    continue

                sym_rva   = symbol_defs[sym_name].rva
                sym_va    = IMAGE_BASE + sym_rva
                offset    = reloc.address  # byte offset within section

                try:
                    rel_type = int(reloc.type)
                except Exception:
                    skipped += 1
                    continue

                if rel_type == IMAGE_REL_AMD64_ABSOLUTE:
                    # no-op
                    applied += 1
                    continue

                if rel_type == IMAGE_REL_AMD64_ADDR64:
                    # Write 64-bit absolute VA
                    if offset + 8 <= len(frag.data):
                        addend = struct.unpack_from("<q", frag.data, offset)[0]
                        struct.pack_into("<Q", frag.data, offset, sym_va + addend)
                        applied += 1
                    else:
                        skipped += 1

                elif rel_type == IMAGE_REL_AMD64_ADDR32NB:
                    # 32-bit RVA (no image base)
                    if offset + 4 <= len(frag.data):
                        addend = struct.unpack_from("<i", frag.data, offset)[0]
                        struct.pack_into("<I", frag.data, offset, (sym_rva + addend) & 0xFFFFFFFF)
                        applied += 1
                    else:
                        skipped += 1

                elif rel_type == IMAGE_REL_AMD64_ADDR32:
                    # 32-bit absolute VA (truncated — unusual in 64-bit)
                    if offset + 4 <= len(frag.data):
                        addend = struct.unpack_from("<i", frag.data, offset)[0]
                        struct.pack_into("<I", frag.data, offset, (sym_va + addend) & 0xFFFFFFFF)
                        applied += 1
                    else:
                        skipped += 1

                elif IMAGE_REL_AMD64_REL32 <= rel_type <= IMAGE_REL_AMD64_REL32_5:
                    # PC-relative 32-bit: value = sym_rva - (patch_rva + 4 + extra)
                    extra = rel_type - IMAGE_REL_AMD64_REL32  # 0..5
                    patch_rva = frag.rva + offset
                    if offset + 4 <= len(frag.data):
                        addend = struct.unpack_from("<i", frag.data, offset)[0]
                        rel_val = sym_rva - (patch_rva + 4 + extra) + addend
                        struct.pack_into("<i", frag.data, offset, rel_val)
                        applied += 1
                    else:
                        skipped += 1

                elif rel_type == IMAGE_REL_AMD64_SECREL:
                    # 32-bit offset from the symbol's section start
                    if offset + 4 <= len(frag.data):
                        struct.pack_into("<I", frag.data, offset, sym_rva & 0xFFFFFFFF)
                        applied += 1
                    else:
                        skipped += 1

                elif rel_type == IMAGE_REL_AMD64_SECTION:
                    # 16-bit PE section index — not critical for execution
                    applied += 1

                else:
                    skipped += 1

    if skipped:
        print(f"  Relocations skipped (unresolved/unknown): {skipped}", file=sys.stderr)

    return applied


# ============================================================
# COLLECT FRAGMENTS
# ============================================================

def collect_fragments(parsed_objects: list[tuple[Path, object]]) -> list[SectionFragment]:
    fragments: list[SectionFragment] = []

    SKIP = {".debug", ".llvm_addrsig", "/$"}

    print()
    print("Section collection")
    print("==================")

    for obj_path, obj in parsed_objects:
        for section in obj.sections:
            raw_name = section.name.rstrip("\x00")

            # Skip debug / metadata sections
            if any(raw_name.startswith(s) for s in SKIP):
                continue

            pe_name = _pe_section_name(raw_name)
            data    = bytearray(section.content)
            chars   = _section_chars(pe_name)

            frag = SectionFragment(
                object_path=obj_path,
                obj_section_name=raw_name,
                pe_section_name=pe_name,
                data=data,
                characteristics=chars,
            )
            fragments.append(frag)

            print(f"  {obj_path.name:<28} {raw_name:<12} {len(data):>8} bytes")

    return fragments


# ============================================================
# COLLECT SYMBOLS (raw, before layout)
# ============================================================

def collect_symbols_raw(
    parsed_objects: list[tuple[Path, object]],
) -> tuple[
    dict[str, tuple],   # definitions:  name -> (obj_idx, sec_idx_1based, value, sec_name)
    dict[str, list],    # undefined:    name -> [(obj_idx, ...)]
]:
    definitions: dict[str, tuple] = {}
    undefined: dict[str, list] = {}

    print()
    print("Symbol table")
    print("============")

    for obj_idx, (obj_path, obj) in enumerate(parsed_objects):
        for symbol in obj.symbols:
            name = symbol.name
            if not name or name.startswith("."):
                continue

            sec_idx = symbol.section_idx

            if sec_idx == 0:
                # undefined external
                undefined.setdefault(name, []).append(obj_idx)
                continue

            if sec_idx < 0:
                continue  # absolute

            try:
                section = obj.sections[sec_idx - 1]
                sec_name = section.name.rstrip("\x00")
            except Exception:
                continue

            if name in definitions:
                prev = definitions[name]
                print(f"  WARNING: duplicate symbol '{name}' — keeping first definition from {parsed_objects[prev[0]][0].name}")
                continue

            definitions[name] = (obj_idx, sec_idx, symbol.value, sec_name)

    print(f"  Definitions : {len(definitions)}")
    print(f"  Undefined   : {len(undefined)}")

    if undefined:
        print()
        print("  External symbols (will be zero-patched if unresolved):")
        for n in sorted(undefined):
            print(f"    {n}")
    else:
        print()
        print("  No unresolved external symbols.")

    return definitions, undefined


# ============================================================
# PE CONSTRUCTION (using LIEF)
# ============================================================

def build_pe(
    fragments: list[SectionFragment],
    output: Path,
    entry_rva: int,
) -> None:
    """
    Build a PE32+ executable from the patched fragment data using LIEF.
    """
    print()
    print("PE construction")
    print("===============")

    # Group fragments into PE sections
    pe_groups: dict[str, bytearray] = {}
    pe_chars:  dict[str, int] = {}
    for frag in fragments:
        name = frag.pe_section_name
        pe_groups.setdefault(name, bytearray())
        pe_groups[name].extend(frag.data)
        pe_chars[name] = frag.characteristics

    if ".text" not in pe_groups or not pe_groups[".text"]:
        raise RuntimeError(
            "No .text section data — cannot create executable. "
            "Make sure the Kyber source produces at least one object with machine code."
        )

    factory = lief.PE.Factory.create(lief.PE.PE_TYPE.PE32_PLUS)
    if factory is None:
        raise RuntimeError("LIEF could not create PE32+ factory.")

    # Add sections in order: .text first, then the rest
    section_order = [".text"] + [k for k in pe_groups if k != ".text"]

    for sec_name in section_order:
        data = pe_groups[sec_name]
        if not data:
            continue

        sec = lief.PE.Section(sec_name, list(data))
        sec.characteristics = pe_chars.get(sec_name, IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ)

        if sec_name == ".text":
            factory.add_section(sec, lief.PE.SECTION_TYPES.TEXT)
        elif sec_name in (".data",):
            factory.add_section(sec, lief.PE.SECTION_TYPES.DATA)
        else:
            factory.add_section(sec)

        print(f"  {sec_name:<12} {len(data):>8} bytes")

    binary = factory.get()
    if binary is None:
        raise RuntimeError("LIEF failed to construct PE binary.")

    # Set entry point
    binary.optional_header.addressof_entrypoint = entry_rva
    binary.optional_header.imagebase = IMAGE_BASE

    # Console subsystem
    try:
        binary.optional_header.subsystem = lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_CUI
    except Exception:
        pass

    output.parent.mkdir(parents=True, exist_ok=True)
    binary.write(str(output))

    if not output.exists():
        raise RuntimeError("LIEF did not produce the output file.")

    print()
    print("================================")
    print("Kyber linking successful!")
    print("================================")
    print(f"Executable : {output}")
    print(f"Size       : {output.stat().st_size} bytes")
    print(f"Entry RVA  : 0x{entry_rva:08X}")


# ============================================================
# MAIN LINK FUNCTION
# ============================================================

def link(output: Path) -> bool:
    print()
    print("Kyber Linker")
    print("============")

    # ── 1. Collect objects ────────────────────────────────────────
    normal_objects, cuda_objects = collect_objects()

    if cuda_objects:
        print(f"\nFound {len(cuda_objects)} CUDA/NGPU object(s).")
        merged = cuda_prelink(cuda_objects, CUDA_TOOLKIT_DEFAULT)
        if merged:
            normal_objects.append(merged)
        else:
            print("WARNING: CUDA/NGPU kernels will NOT be available.", file=sys.stderr)
    else:
        print("No CUDA/NGPU objects — skipping CUDA pre-link.")

    if not normal_objects:
        raise RuntimeError(f"No .obj files found in:\n  {OBJECT_DIRECTORY}")

    print("\nObjects:")
    for p in normal_objects:
        print(f"  {p}")

    libraries = collect_libraries()
    if libraries:
        print("\nNForce libraries:")
        for p in libraries:
            print(f"  {p}")

    # ── 2. Parse COFF ─────────────────────────────────────────────
    print("\nParsing COFF objects...")
    parsed: list[tuple[Path, object]] = []
    for path in normal_objects:
        obj = parse_object(path)
        parsed.append((path, obj))
        print(f"  {path.name:<30} sections={len(obj.sections)}  symbols={len(obj.symbols)}")

    # ── 3. Collect fragments ──────────────────────────────────────
    fragments = collect_fragments(parsed)

    # ── 4. Collect symbols (raw) ──────────────────────────────────
    definitions_raw, undefined = collect_symbols_raw(parsed)

    # ── 5. Layout — assign RVAs ───────────────────────────────────
    print("\nLayout")
    print("======")

    current_rva = SECTION_ALIGNMENT

    # Build fragment lookup by (obj_path, section_name)
    for frag in fragments:
        current_rva = (current_rva + 15) & ~15
        frag.rva = current_rva
        current_rva += len(frag.data)

    # Round up to section alignment
    total_rva = (current_rva + SECTION_ALIGNMENT - 1) & ~(SECTION_ALIGNMENT - 1)

    # Build symbol_defs: name -> SymbolDef with correct RVA
    frag_lookup: dict[tuple[Path, str], SectionFragment] = {
        (f.object_path, f.obj_section_name): f for f in fragments
    }

    symbol_defs: dict[str, SymbolDef] = {}
    for name, (obj_idx, sec_idx, value, sec_name) in definitions_raw.items():
        obj_path = parsed[obj_idx][0]
        frag = frag_lookup.get((obj_path, sec_name))
        if frag is not None:
            symbol_defs[name] = SymbolDef(name=name, rva=frag.rva + value)

    print(f"  Total image size : 0x{total_rva:08X}  ({total_rva} bytes)")
    print(f"  Symbols resolved : {len(symbol_defs)}")

    # ── 6. Apply relocations ──────────────────────────────────────
    print("\nRelocation patching")
    print("===================")
    applied = apply_relocations(fragments, parsed, symbol_defs)
    print(f"  Applied : {applied}")

    # ── 7. Find entry point ───────────────────────────────────────
    entry_rva = 0
    for candidate in ("main", "_main", "kyber_main", "_kyber_main"):
        if candidate in symbol_defs:
            entry_rva = symbol_defs[candidate].rva
            print(f"\nEntry point: {candidate}  RVA=0x{entry_rva:08X}")
            break
    else:
        print("\nWARNING: No main/kyber_main entry symbol found. Entry RVA = 0.")

    # ── 8. Build PE ───────────────────────────────────────────────
    build_pe(fragments, output, entry_rva)

    return True


# ============================================================
# CLI
# ============================================================

def main() -> int:
    if len(sys.argv) >= 2:
        output = Path(sys.argv[1])
        if not output.is_absolute():
            output = KYBER_ROOT / output
    else:
        output = DEFAULT_OUTPUT

    try:
        link(output)
    except Exception as exc:
        print()
        print("================================")
        print("Kyber linker failed")
        print("================================")
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
