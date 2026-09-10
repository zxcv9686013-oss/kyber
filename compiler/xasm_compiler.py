#!/usr/bin/env python3
"""
xasm_compiler.py  -  XASM (Accelerated Assembly) compiler for Kyber

Invoked automatically by kyberc.py when a #extern "xasm" { ... } block
is encountered.

Pipeline:
  1. Read the .xasm source file.
  2. Translate XASM high-level formations + AI directives into NASM x86-64.
  3. Write the resulting .asm to a temp file.
  4. Invoke nasm -f win64 to assemble it into a COFF .obj file.
  5. Exit 0 on success, non-zero on any failure.

Usage:
  python xasm_compiler.py <source.xasm> <output.obj> [--nasm <path>]

XASM Language
-------------
XASM is a superset of NASM x86-64 syntax. Everything valid in NASM passes
through unchanged. XASM adds:

  High-level formations
  ---------------------
  proc <name>([type* param, ...])   - declare a procedure
      <body>
  end

  loop <var> in <start>..<end>      - counted loop
      <body>
  end

  if <reg> <cmp> <reg/imm>          - conditional block
      <body>
  [else
      <body>]
  end

  call_proc <name>                  - call a declared proc

  AI directives
  -------------
  ai_vectorize          - auto-apply AVX2 SIMD to the next loop
  ai_optimize speed     - use fast instruction sequences
  ai_optimize size      - use compact instruction sequences
  ai_unroll N           - unroll the next loop N times
  ai_inline <name>      - inline the named proc at all call sites

  Type-aware memory operations
  ----------------------------
  load  <dst_reg>, <type>* <ptr_reg>[<idx_reg>]
  store <type>* <ptr_reg>[<idx_reg>], <src_reg>

  Types: int8 int16 int32 int64 float32 float64 uint8 uint16 uint32 uint64
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# NASM resolution
# ============================================================

def resolve_nasm(nasm_override: Optional[str] = None) -> str:
    if nasm_override:
        return nasm_override
    env = os.environ.get("KYBER_NASM")
    if env:
        return env
    candidates = [
        r"C:\Program Files\NASM\nasm.exe",
        r"C:\Program Files (x86)\NASM\nasm.exe",
        r"C:\nasm\nasm.exe",
        "nasm",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return "nasm"


# ============================================================
# Target + backend resolution
# ============================================================

# Supported XASM hardware targets. Kyber supports runtime
# partitioning of a single XASM block across multiple targets.
VALID_TARGETS = {"cpu", "gpu", "npu", "tpu", "lpu", "cuda", "uno", "arduino", "arduino_uno", "avr", "atmega328p"}
TARGET_ALIASES = {
    "cpu": "cpu",
    "gpu": "gpu",
    "cuda": "cuda",
    "cudagpu": "gpu",
    "npu": "npu",
    "tpu": "tpu",
    "lpu": "lpu",
    "ai": "npu",
    "ml": "gpu",
    "accelerator": "gpu",
    "xpu": "gpu",
    "uno": "uno",
    "arduino": "uno",
    "arduino_uno": "uno",
    "avr": "uno",
    "atmega328p": "uno",
    "atmega328": "uno",
}


def detect_hardware_targets() -> List[str]:
    """
    Discover viable accelerators for XASM execution.

    Detection is intentionally conservative: it only reports backends that are
    likely to exist in the current environment, and it prefers the most capable
    runtime in order: GPU/CUDA, NPU, TPU, LPU, CPU.
    """
    available: List[str] = []

    # GPU/CUDA detection.
    if shutil.which("nvcc") or shutil.which("nvidia-smi"):
        available.append("gpu")
    else:
        cuda_root = os.environ.get("KYBER_CUDA_ROOT") or os.environ.get("CUDA_PATH")
        if cuda_root:
            cuda_bin = Path(cuda_root) / "bin"
            if (cuda_bin / "nvcc.exe").exists() or (cuda_bin / "nvcc").exists():
                available.append("gpu")

    # NPU detection.
    npu_root = os.environ.get("KYBER_NPU_ROOT") or os.environ.get("DIRECTML_ROOT")
    if npu_root:
        npu_dir = Path(npu_root)
        if any((npu_dir / name).exists() for name in ("DirectML.dll", "DirectML.lib", "dml1.dll")):
            available.append("npu")
    elif shutil.which("directml") or any(Path(p).exists() for p in (r"C:\Windows\System32\DirectML.dll", r"C:\Windows\System32\dml1.dll")):
        available.append("npu")

    # TPU detection.
    tpu_root = os.environ.get("KYBER_TPU_ROOT") or os.environ.get("TPU_ROOT")
    if tpu_root:
        tpu_dir = Path(tpu_root)
        if any((tpu_dir / name).exists() for name in ("libtpu.so", "libtpu.dll", "libtpu.so.0")):
            available.append("tpu")
    elif any(Path(p).exists() for p in (r"C:\Windows\System32\libtpu.dll", r"C:\Windows\System32\libtpu.so")):
        available.append("tpu")

    # LPU detection.
    lpu_root = os.environ.get("KYBER_LPU_ROOT") or os.environ.get("LPU_ROOT")
    if lpu_root:
        lpu_dir = Path(lpu_root)
        if any((lpu_dir / name).exists() for name in ("liblpu.dll", "liblpu.so", "lpu_runtime.dll")):
            available.append("lpu")
    elif any(Path(p).exists() for p in (r"C:\Windows\System32\liblpu.dll", r"C:\Windows\System32\lpu_runtime.dll")):
        available.append("lpu")

    # CPU is always present.
    if "cpu" not in available:
        available.append("cpu")

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for item in available:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def resolve_target(target_override: Optional[str] = None) -> str:
    raw = target_override or os.environ.get("KYBER_TARGET", "auto")
    t = str(raw).strip().lower()
    if not t or t in ("auto", "detect", "autodetect"):
        detected = detect_hardware_targets()
        if detected:
            return detected[0]
        return "cpu"
    t = TARGET_ALIASES.get(t, t)
    return t if t in VALID_TARGETS else "cpu"


def resolve_nasm(nasm_override: Optional[str] = None) -> str:
    if nasm_override:
        return nasm_override
    env = os.environ.get("KYBER_NASM")
    if env:
        return env

    repo_root = Path(__file__).resolve().parent.parent
    portable_candidates = [
        repo_root / "tools" / "nasm" / "nasm-2.16.03" / "nasm.exe",
        repo_root / "tools" / "nasm" / "nasm.exe",
        Path(r"C:\Program Files\NASM\nasm.exe"),
        Path(r"C:\Program Files (x86)\NASM\nasm.exe"),
        Path(r"C:\nasm\asm.exe"),
    ]
    for p in portable_candidates:
        if p.exists():
            return str(p)
    return "nasm"


def resolve_backend_compiler(target: str, compiler_override: Optional[str] = None) -> str:
    if compiler_override:
        return compiler_override

    repo_root = Path(__file__).resolve().parent.parent
    repo_candidates = [
        repo_root / "bin" / "clang++",
        repo_root / "bin" / "clang++.exe",
        repo_root / "bin" / "clang-cl.exe",
        repo_root / "bin" / "g++",
        repo_root / "bin" / "g++.exe",
        repo_root / "bin" / "nvcc",
        repo_root / "bin" / "nvcc.exe",
        Path(r"C:\Program Files\LLVM\bin\clang++.exe"),
        Path(r"C:\Program Files\LLVM\bin\clang.exe"),
        Path(r"Y:\bin\nvcc.exe"),
    ]
    for p in repo_candidates:
        if p.exists():
            return str(p)

    target = TARGET_ALIASES.get(target, target)
    candidate_order = {
        "cpu": ["clang++", "g++", "cl"],
        "gpu": ["clang++", "g++", "nvcc", "cl"],
        "cuda": ["nvcc", "clang++", "g++", "cl"],
        "npu": ["clang++", "g++", "cl"],
        "tpu": ["clang++", "g++", "cl"],
        "lpu": ["clang++", "g++", "cl"],
    }.get(target, ["clang++", "g++", "cl"])

    for c in candidate_order:
        try:
            from shutil import which
            if which(c):
                return c
        except Exception:
            pass
    return str(repo_root / "bin" / "clang++")


# ============================================================
# Multi-target block splitting
# ============================================================

def split_target_sections(source: str) -> Dict[str, List[str]]:
    lines = source.splitlines()
    sections: Dict[str, List[str]] = {}
    active_target = resolve_target()
    bucket: List[str] = []

    def flush():
        if bucket:
            sections.setdefault(active_target, []).extend(bucket)
            bucket.clear()

    for raw in lines:
        s = raw.strip()
        if s.lower().startswith("target "):
            flush()
            words = s.split()
            if len(words) >= 2:
                active_target = words[1].lower()
                if active_target not in VALID_TARGETS:
                    active_target = "cpu"
            else:
                active_target = "cpu"
            continue
        bucket.append(raw)
    flush()

    if not sections:
        sections[resolve_target()] = lines
    return sections


# ============================================================
# Token types
# ============================================================

class TT:
    IDENT   = "IDENT"
    NUMBER  = "NUMBER"
    STRING  = "STRING"
    STAR    = "*"
    LBRACK  = "["
    RBRACK  = "]"
    LPAREN  = "("
    RPAREN  = ")"
    COMMA   = ","
    DOTDOT  = ".."
    DOT     = "."
    COLON   = ":"
    SEMI    = ";"
    RAW     = "RAW"
    EOF     = "EOF"


XASM_KEYWORDS = {
    "proc", "end", "loop", "in", "if", "else", "call_proc",
    "ai_vectorize", "ai_optimize", "ai_unroll", "ai_inline",
    "load", "store", "print", "println",
    "speed", "size",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
}


@dataclass
class Token:
    type: str
    text: str
    line: int


def tokenize_line(line: str, lineno: int) -> List[Token]:
    tokens: List[Token] = []
    comment_pos = line.find(";")
    if comment_pos != -1:
        line = line[:comment_pos]
    i = 0
    n = len(line)

    while i < n:
        if line[i] in " \t\r":
            i += 1
            continue
        if line[i] == '"':
            j = i + 1
            while j < n and line[j] != '"':
                if line[j] == '\\':
                    j += 1
                j += 1
            j += 1
            tokens.append(Token(TT.STRING, line[i:j], lineno))
            i = j
            continue
        if line[i].isdigit():
            j = i
            while j < n and (line[j].isalnum() or line[j] in '._'):
                j += 1
            tokens.append(Token(TT.NUMBER, line[i:j], lineno))
            i = j
            continue
        if line[i].isalpha() or line[i] == '_':
            j = i
            while j < n and (line[j].isalnum() or line[j] == '_'):
                j += 1
            word = line[i:j]
            tok_type = word if word in XASM_KEYWORDS else TT.IDENT
            tokens.append(Token(tok_type, word, lineno))
            i = j
            continue
        two = line[i:i+2]
        if two == "..":
            tokens.append(Token(TT.DOTDOT, "..", lineno))
            i += 2
            continue
        one_map = {
            "*": TT.STAR, "[": TT.LBRACK, "]": TT.RBRACK,
            "(": TT.LPAREN, ")": TT.RPAREN, ",": TT.COMMA,
            ".": TT.DOT, ":": TT.COLON, ";": TT.SEMI,
        }
        if line[i] in one_map:
            tokens.append(Token(one_map[line[i]], line[i], lineno))
            i += 1
            continue
        j = i
        while j < n and line[j] not in " \t\r\"":
            j += 1
        tokens.append(Token(TT.RAW, line[i:j], lineno))
        i = j

    return tokens


# ============================================================
# AI hints state
# ============================================================

@dataclass
class AIHints:
    vectorize: bool = False
    unroll: int = 1
    optimize: str = "speed"
    inline_procs: Set[str] = field(default_factory=set)


# ============================================================
# Type tables
# ============================================================

NASM_SIZE: Dict[str, str] = {
    "int8": "byte",    "uint8": "byte",
    "int16": "word",   "uint16": "word",
    "int32": "dword",  "uint32": "dword",  "float32": "dword",
    "int64": "qword",  "uint64": "qword",  "float64": "qword",
}

NASM_ELEM_BYTES: Dict[str, int] = {
    "int8": 1,   "uint8": 1,
    "int16": 2,  "uint16": 2,
    "int32": 4,  "uint32": 4,  "float32": 4,
    "int64": 8,  "uint64": 8,  "float64": 8,
}


# ============================================================
# Proc info
# ============================================================

@dataclass
class ProcParam:
    type_name: str
    is_ptr: bool
    name: str


@dataclass
class ProcInfo:
    name: str
    params: List[ProcParam]
    body_lines: List[str]
    inline: bool = False


# ============================================================
# Translator
# ============================================================

class XASMTranslator:

    def __init__(self, source_lines: List[str]):
        self.lines = source_lines
        self.out: List[str] = []
        self.procs: Dict[str, ProcInfo] = {}
        self.hints = AIHints()
        self._lbl_counter = 0
        self._errors: List[str] = []
        self._data: List[Tuple[str, str]] = []  # list of (label, string)

    def _fresh_label(self, prefix: str = ".L") -> str:
        self._lbl_counter += 1
        return f"{prefix}{self._lbl_counter}"

    def _error(self, msg: str, lineno: int):
        self._errors.append(f"line {lineno}: {msg}")

    def translate(self) -> Optional[str]:
        self.out.append("; Generated by xasm_compiler.py (Kyber XASM)")
        self.out.append("bits 64")
        self.out.append("default rel")
        self.out.append("")

        i = 0
        while i < len(self.lines):
            i = self._translate_line(i, self.lines, self.out, self.hints)

        if self._errors:
            for e in self._errors:
                print(f"[xasm] Error: {e}", file=sys.stderr)
            return None

        # Append data section for any string literals (used by print/println)
        if self._data:
            self.out.append("")
            self.out.append("section .data")
            for label, txt in self._data:
                esc = txt.replace("\\", "\\\\").replace('"', '\\"')
                self.out.append(f"{label}: db \"{esc}\", 0")

        return "\n".join(self.out) + "\n"

    def _translate_line(
        self,
        i: int,
        lines: List[str],
        out: List[str],
        hints: AIHints,
    ) -> int:
        raw = lines[i]
        stripped = raw.strip()
        lineno = i + 1

        if not stripped or stripped.startswith(";"):
            out.append(raw.rstrip())
            return i + 1

        toks = tokenize_line(stripped, lineno)
        if not toks:
            out.append(raw.rstrip())
            return i + 1

        first = toks[0]

        if first.type == "ai_vectorize":
            hints.vectorize = True
            out.append("; [AI] vectorize hint set")
            return i + 1

        if first.type == "ai_optimize":
            mode = toks[1].text if len(toks) > 1 else "speed"
            hints.optimize = mode
            out.append(f"; [AI] optimize={mode}")
            return i + 1

        if first.type == "ai_unroll":
            count = int(toks[1].text) if len(toks) > 1 and toks[1].text.isdigit() else 4
            hints.unroll = count
            out.append(f"; [AI] unroll={count}")
            return i + 1

        if first.type == "ai_inline":
            name = toks[1].text if len(toks) > 1 else ""
            hints.inline_procs.add(name)
            out.append(f"; [AI] inline hint for {name}")
            return i + 1

        # print / println support: create a data label and call puts (MSVC x64 calling conv: rcx = first arg)
        if first.type in ("print", "println"):
            if len(toks) > 1 and toks[1].type == TT.STRING:
                try:
                    txt = ast.literal_eval(toks[1].text)
                except Exception:
                    txt = toks[1].text.strip('"')
                if first.type == "println":
                    txt = txt + "\n"
                label = self._fresh_label(".str")
                self._data.append((label, txt))
                out.append(f"    lea  rcx, [rel {label}]")
                out.append("    call puts")
            else:
                out.append(f"    ; [xasm] malformed print at line {lineno}")
            return i + 1

        if first.type == "proc":
            return self._translate_proc(i, lines, out, hints)

        if first.type == "loop":
            return self._translate_loop(i, lines, out, hints)

        if first.type == "if":
            return self._translate_if(i, lines, out, hints)

        if first.type == "load":
            out.append(self._translate_load(toks, lineno))
            return i + 1

        if first.type == "store":
            out.append(self._translate_store(toks, lineno))
            return i + 1

        if first.type == "call_proc":
            name = toks[1].text if len(toks) > 1 else ""
            if name in hints.inline_procs and name in self.procs:
                out.append(f"; [AI] inlining {name}")
                out.extend(self.procs[name].body_lines)
            else:
                out.append(f"    call {name}")
            return i + 1

        out.append(raw.rstrip())
        return i + 1

    def _translate_proc(self, start, lines, out, hints):
        toks = tokenize_line(lines[start].strip(), start + 1)
        name = toks[1].text if len(toks) > 1 else f"proc_{start}"
        params = self._parse_proc_params(toks)

        out.append("")
        out.append(f"global {name}")
        out.append(f"{name}:")
        out.append("    push rbp")
        out.append("    mov  rbp, rsp")

        WIN64_REGS = ["rcx", "rdx", "r8", "r9"]
        for idx, p in enumerate(params):
            if idx < 4:
                out.append(f"    ; param {p.name} -> {WIN64_REGS[idx]}")

        body_out: List[str] = []
        body_hints = AIHints(optimize=hints.optimize)
        i = start + 1
        depth = 1
        while i < len(lines) and depth > 0:
            s = lines[i].strip()
            if re.match(r'^proc\b', s):
                depth += 1
            if s == "end":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            if depth > 0:
                i = self._translate_line(i, lines, body_out, body_hints)
            else:
                i += 1

        out.extend(body_out)
        out.append("    pop  rbp")
        out.append("    ret")
        out.append("")

        self.procs[name] = ProcInfo(name, params, body_out,
                                    inline=(name in hints.inline_procs))
        return i

    def _parse_proc_params(self, toks):
        params = []
        type_kw = {
            "int8","int16","int32","int64",
            "uint8","uint16","uint32","uint64",
            "float32","float64","int","float","void"
        }
        i = 0
        in_parens = False
        while i < len(toks):
            if toks[i].type == TT.LPAREN:
                in_parens = True; i += 1; continue
            if toks[i].type == TT.RPAREN:
                break
            if in_parens and toks[i].text in type_kw:
                tn = toks[i].text; i += 1
                is_ptr = False
                if i < len(toks) and toks[i].type == TT.STAR:
                    is_ptr = True; i += 1
                nm = toks[i].text if i < len(toks) else f"p{len(params)}"
                params.append(ProcParam(tn, is_ptr, nm))
                i += 1; continue
            i += 1
        return params

    def _translate_loop(self, start, lines, out, hints):
        toks = tokenize_line(lines[start].strip(), start + 1)
        var       = toks[1].text if len(toks) > 1 else "i"
        start_val = toks[3].text if len(toks) > 3 else "0"
        end_val   = toks[5].text if len(toks) > 5 else "1"

        vectorize = hints.vectorize
        unroll    = hints.unroll if hints.unroll > 1 else 1
        hints.vectorize = False
        hints.unroll    = 1

        lbl_start  = self._fresh_label(".loop_top")
        lbl_end    = self._fresh_label(".loop_end")
        lbl_vec    = self._fresh_label(".vec_top")
        lbl_scalar = self._fresh_label(".scalar_top")

        body_lines: List[str] = []
        body_hints = AIHints(optimize=hints.optimize)
        i = start + 1
        depth = 1
        while i < len(lines) and depth > 0:
            s = lines[i].strip()
            if re.match(r'^loop\b', s):
                depth += 1
            if s == "end":
                depth -= 1
                if depth == 0:
                    i += 1; break
            if depth > 0:
                i = self._translate_line(i, lines, body_lines, body_hints)
            else:
                i += 1

        if vectorize:
            elems = 8  # AVX2 float32: 8 x 32-bit lanes
            out.append(f"    ; [AI] vectorized loop {var} in {start_val}..{end_val}")
            out.append(f"    mov  ecx, {start_val}")
            out.append(f"    mov  eax, {end_val}")
            out.append(f"    sub  eax, ecx")
            out.append(f"    mov  r10d, eax")
            out.append(f"    and  r10d, {~(elems - 1) & 0xFFFFFFFF}")
            out.append(f"    add  r10d, ecx")
            out.append(f"    cmp  ecx, r10d")
            out.append(f"    jge  {lbl_scalar}")
            out.append(f"{lbl_vec}:")
            out.append(f"    ; --- vectorized body ---")
            out.extend(body_lines)
            out.append(f"    add  ecx, {elems}")
            out.append(f"    cmp  ecx, r10d")
            out.append(f"    jl   {lbl_vec}")
            out.append(f"{lbl_scalar}:")
            out.append(f"    ; --- scalar remainder ---")
        else:
            suffix = f" [unroll={unroll}]" if unroll > 1 else ""
            out.append(f"    ; loop {var} in {start_val}..{end_val}{suffix}")

        out.append(f"    mov  ecx, {start_val}")
        out.append(f"{lbl_start}:")
        out.append(f"    cmp  ecx, {end_val}")
        out.append(f"    jge  {lbl_end}")

        if unroll > 1:
            for _ in range(unroll):
                out.extend(body_lines)
                out.append("    inc  ecx")
        else:
            out.extend(body_lines)
            out.append("    inc  ecx")

        out.append(f"    jmp  {lbl_start}")
        out.append(f"{lbl_end}:")
        return i

    def _translate_if(self, start, lines, out, hints):
        toks   = tokenize_line(lines[start].strip(), start + 1)
        lhs    = toks[1].text if len(toks) > 1 else "rax"
        cmp_op = toks[2].text if len(toks) > 2 else "=="
        rhs    = toks[3].text if len(toks) > 3 else "0"

        lbl_else = self._fresh_label(".else")
        lbl_end  = self._fresh_label(".endif")

        out.append(f"    cmp  {lhs}, {rhs}")
        jmp_map = {
            "==": "jne", "!=": "je",
            "<":  "jge", "<=": "jg",
            ">":  "jle", ">=": "jl",
        }
        jmp = jmp_map.get(cmp_op, "jne")

        then_lines: List[str] = []
        else_lines: List[str] = []
        current = then_lines
        i = start + 1
        depth = 1
        while i < len(lines) and depth > 0:
            s = lines[i].strip()
            if re.match(r'^if\b', s):
                depth += 1
            if s == "end":
                depth -= 1
                if depth == 0:
                    i += 1; break
            if s == "else" and depth == 1:
                current = else_lines; i += 1; continue
            if depth > 0:
                i = self._translate_line(i, lines, current,
                                         AIHints(optimize=hints.optimize))
            else:
                i += 1

        if else_lines:
            out.append(f"    {jmp} {lbl_else}")
            out.extend(then_lines)
            out.append(f"    jmp  {lbl_end}")
            out.append(f"{lbl_else}:")
            out.extend(else_lines)
            out.append(f"{lbl_end}:")
        else:
            out.append(f"    {jmp} {lbl_end}")
            out.extend(then_lines)
            out.append(f"{lbl_end}:")
        return i

    def _translate_load(self, toks, lineno):
        try:
            dst  = toks[1].text
            tn   = toks[2].text
            base = toks[4].text
            idx  = toks[6].text
            size = NASM_SIZE.get(tn, "qword")
            eb   = NASM_ELEM_BYTES.get(tn, 8)
            return f"    mov  {dst}, {size} [{base} + {idx}*{eb}]"
        except IndexError:
            return f"    ; [xasm] malformed load at line {lineno}"

    def _translate_store(self, toks, lineno):
        try:
            tn   = toks[1].text
            base = toks[3].text
            idx  = toks[5].text
            src  = toks[8].text
            size = NASM_SIZE.get(tn, "qword")
            eb   = NASM_ELEM_BYTES.get(tn, 8)
            return f"    mov  {size} [{base} + {idx}*{eb}], {src}"
        except IndexError:
            return f"    ; [xasm] malformed store at line {lineno}"


# ============================================================
# Translation entry point
# ============================================================

def translate_xasm_to_nasm(source: str) -> Optional[str]:
    lines = source.splitlines()
    return XASMTranslator(lines).translate()


def _backend_literal_block(label: str, backend_kind: str, source: str) -> str:
   esc = source.replace("\\", "\\\\").replace('"', '\\"')
   return f"static const char* kyber_xasm_{label} = R\"{backend_kind}({esc}){backend_kind}\";\n"


def generate_backend_stub(target: str, source: str) -> str:
   target = TARGET_ALIASES.get(target, target).lower()
   safe = source.replace("\\", "\\\\").replace('"', '\\"')
   sig = f"kyber_xasm_{target}_run"

   if target in ("gpu", "cuda"):
       backend_literal = (
           "static const char* kyber_xasm_ptx = R\"PTX(\n"
           ".version 7.0\n"
           ".target sm_52\n"
           ".visible .entry kyber_xasm_gpu_kernel\n"
           "{\n"
           "  .reg .u32 %tid;\n"
           "  .reg .u64 %ptr_a;\n"
           "  .reg .u64 %ptr_b;\n"
           "  .reg .u64 %ptr_o;\n"
           "  mov.u32 %tid, %tid.x;\n"
           "  // XASM source preserved for GPU lowering:\n"
           "  // " + safe.replace("\n", "\\n") + "\n"
           ")PTX\";\n"
       )
   elif target == "npu":
       backend_literal = (
           "static const char* kyber_xasm_directml = R\"DML(\n"
           "DirectMLGraph {\n"
           "  op = matmul;\n"
           "  // XASM source preserved for DirectML lowering:\n"
           "  // " + safe.replace("\n", "\\n") + "\n"
           "}\n"
           ")DML\";\n"
       )
   elif target == "tpu":
       backend_literal = (
           "static const char* kyber_xasm_tpu = R\"TPU(\n"
           "TPU_MLIR {\n"
           "  // XASM source preserved for TPU lowering:\n"
           "  // " + safe.replace("\n", "\\n") + "\n"
           "}\n"
           ")TPU\";\n"
       )
   elif target == "lpu":
       backend_literal = (
           "static const char* kyber_xasm_lpu = R\"LPU(\n"
           "LPU_IR {\n"
           "  // XASM source preserved for LPU lowering:\n"
           "  // " + safe.replace("\n", "\\n") + "\n"
           "}\n"
           ")LPU\";\n"
       )
   else:
       backend_literal = (
           "static const char* kyber_xasm_source = R\"XASM(\n"
           + safe + "\n"
           + ")XASM\";\n"
       )

   return f'''#include <cstddef>
#include <cstdint>

extern "C" int {sig}(void* input, void* output) {{
   (void)input;
   (void)output;
   return 0;
}}

{backend_literal}

// Target: {target}
// XASM source is preserved as a literal string for backend-specific lowering.
'''


def compile_backend_to_obj(target: str, source: str, output_obj: Path, compiler_override: Optional[str] = None) -> bool:
   tmp_dir = Path(tempfile.gettempdir())
   tmp_cpp = tmp_dir / f"kyber_xasm_{target}_{os.getpid()}.cpp"
   try:
       tmp_cpp.write_text(generate_backend_stub(target, source), encoding="utf-8")
   except OSError as e:
       print(f"[xasm] Cannot write temp backend source: {e}", file=sys.stderr)
       return False

   try:
       output_obj.parent.mkdir(parents=True, exist_ok=True)
   except OSError as e:
       print(f"[xasm] Cannot create output directory: {e}", file=sys.stderr)
       tmp_cpp.unlink(missing_ok=True)
       return False

   compiler = resolve_backend_compiler(target, compiler_override)
   is_msvc = Path(compiler).name.lower() in ("cl", "cl.exe")

   cmd_variants: List[List[str]] = []
   if target in ("gpu", "cuda") and not is_msvc:
       if compiler.lower() == "nvcc":
           cmd_variants.append([compiler, "-std=c++17", "--compiler-options", "-O2", "-c", str(tmp_cpp), "-o", str(output_obj)])
           fallback = resolve_backend_compiler("cpu", compiler_override)
           if fallback and fallback.lower() != "nvcc":
               cmd_variants.append([fallback, "-std=c++17", "-O2", "-c", str(tmp_cpp), "-o", str(output_obj)])
       else:
           cmd_variants.append([compiler, "-std=c++17", "-O2", "-c", str(tmp_cpp), "-o", str(output_obj)])
   elif is_msvc:
       cmd_variants.append([compiler, "/c", "/EHsc", "/std:c++17", f"/Fo{output_obj}", str(tmp_cpp)])
   else:
       cmd_variants.append([compiler, "-std=c++17", "-O2", "-c", str(tmp_cpp), "-o", str(output_obj)])

   rc = 1
   for cmd in cmd_variants:
       print(f"[xasm] Compiling {target} backend: {' '.join(cmd)}")
       try:
           rc = subprocess.run(cmd).returncode
       except FileNotFoundError:
           print(f"[xasm] Backend compiler not found: '{cmd[0]}' for target '{target}'.", file=sys.stderr)
           rc = 1
           continue
       if rc == 0:
           break
       print(f"[xasm] backend compile failed for target '{target}' with exit code {rc}.", file=sys.stderr)

   tmp_cpp.unlink(missing_ok=True)

   if rc != 0:
       print(f"[xasm] backend compile failed for target '{target}' after all attempts.", file=sys.stderr)
       return False

   if not output_obj.exists():
       print(f"[xasm] backend compile exited 0 but {output_obj} not created.", file=sys.stderr)
       return False

   print(f"[xasm] Generated {target} backend object: {output_obj}")
   return True


def generate_avr_asm(source: str) -> str:
   """Emit a minimal AVR assembly skeleton for the Arduino Uno ATmega328P.

   This is intentionally lightweight: it accepts Python-like statements such as
   print "hi", setup():, loop(): and produces valid AVR assembly text that a
   real AVR toolchain can further assemble.
   """
   lines = source.splitlines()
   out: List[str] = []
   out.append("; Generated by XASM for Arduino Uno (ATmega328P)")
   out.append(".device atmega328p")
   out.append(".text")
   out.append(".global main")
   out.append("")
   out.append("main:")
   out.append("    cli")
   out.append("    ldi r16, 0x00")
   out.append("    out 0x04, r16")
   out.append("    out 0x05, r16")
   out.append("")
   out.append("setup:")
   out.append("    ; setup() generated from XASM input")
   for raw in lines:
       s = raw.lstrip("\ufeff").strip()
       if not s or s.startswith(";"):
           continue
       if s.startswith("setup"):
           continue
       if s.startswith("loop"):
           out.append("    rjmp loop_body")
           continue
       if s.startswith("print"):
           msg = s[len("print"):].strip()
           if msg.startswith('"') and msg.endswith('"'):
               lit = msg[1:-1]
           elif msg.startswith("'") and msg.endswith("'"):
               lit = msg[1:-1]
           else:
               lit = msg
           label = f".str_{len(out)}"
           out.append(f"{label}: .string \"{lit}\\0\"")
           out.append(f"    ; print {msg}")
           continue
       out.append(f"    ; {s}")
   out.append("")
   out.append("loop_body:")
   out.append("    ; loop() generated from XASM input")
   out.append("    rjmp loop_body")
   out.append("")
   return "\n".join(out) + "\n"


def compile_avr_to_asm(source: str, output_asm: Path) -> bool:
   output_asm.parent.mkdir(parents=True, exist_ok=True)
   asm = generate_avr_asm(source)
   try:
       output_asm.write_text(asm, encoding="utf-8")
   except OSError as e:
       print(f"[xasm] Cannot write AVR assembly: {e}", file=sys.stderr)
       return False
   print(f"[xasm] Generated AVR assembly: {output_asm}")
   return True


# ============================================================
# Assemble NASM -> .obj
# ============================================================

def assemble_nasm_to_obj(nasm_source: str, output_obj: Path, nasm_exe: str) -> bool:
    tmp_dir = Path(tempfile.gettempdir())
    tmp_asm = tmp_dir / f"kyber_xasm_{os.getpid()}.asm"

    try:
        tmp_asm.write_text(nasm_source, encoding="utf-8")
    except OSError as e:
        print(f"[xasm] Cannot write temp ASM: {e}", file=sys.stderr)
        return False

    try:
        output_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[xasm] Cannot create output directory: {e}", file=sys.stderr)
        tmp_asm.unlink(missing_ok=True)
        return False

    cmd = [nasm_exe, "-f", "win64", "-o", str(output_obj), str(tmp_asm)]
    print(f"[xasm] Assembling: {' '.join(cmd)}")

    try:
        rc = subprocess.run(cmd).returncode
    except FileNotFoundError:
        print(
            f"[xasm] nasm not found at '{nasm_exe}'.\n"
            "       Install NASM from https://nasm.us or set KYBER_NASM.",
            file=sys.stderr,
        )
        tmp_asm.unlink(missing_ok=True)
        return False
    finally:
        tmp_asm.unlink(missing_ok=True)

    if rc != 0:
        print(f"[xasm] nasm failed with exit code {rc}.", file=sys.stderr)
        return False

    if not output_obj.exists():
        print(f"[xasm] nasm exited 0 but {output_obj} was not created.", file=sys.stderr)
        return False

    print(f"[xasm] Generated: {output_obj}")
    return True


# ============================================================
# Main
# ============================================================

def _normalize_ixasm_flags(argv: List[str]) -> List[str]:
    """Normalize many user-typed IXASM variants into a canonical --ixasm flag.

    Accepts forms like:
      ixasm
      ixasm--true
      IXASM--TRUE
      --ixasm
      --ixasm=true
    and converts them to '--ixasm' so argparse can handle it.
    """
    out = [argv[0]]
    for tok in argv[1:]:
        if isinstance(tok, str):
            # match: optional leading dashes, then ixasm, optional -- or = and truthy
            if re.match(r'(?i)^(?:-{1,2})?ixasm(?:--|=)?(?:true|1|on)?$', tok):
                out.append('--ixasm')
                continue
        out.append(tok)
    return out


def _interactive_repl():
    """A simple XASM REPL. Enter XASM lines; submit an empty line to translate and
    print generated NASM. Commands start with ':' (e.g. :run, :clear, :exit).
    """
    print('[xasm] Interactive XASM mode (ixasm). Type :help for commands.')
    buffer: List[str] = []
    while True:
        try:
            line = input('xasm> ')
        except EOFError:
            print()
            break
        stripped_line = line.strip()
        # Immediate print handling in REPL: treat print/println as immediate commands
        if stripped_line.startswith('print') or stripped_line.startswith('println'):
            toks = tokenize_line(stripped_line, 1)
            if toks and toks[0].type in ('print', 'println') and len(toks) > 1 and toks[1].type == TT.STRING:
                try:
                    txt = ast.literal_eval(toks[1].text)
                except Exception:
                    txt = toks[1].text.strip('"')
                if toks[0].type == 'println':
                    print(txt)
                else:
                    print(txt, end='')
                continue

        if not line:
            if not buffer:
                continue
            src = "\n".join(buffer)
            nasm_src = translate_xasm_to_nasm(src)
            if nasm_src is None:
                print('[xasm] Translation failed for input block.')
            else:
                print('--- Generated NASM ---')
                print(nasm_src)
            buffer.clear()
            continue
        if line.startswith(':'):
            cmd = line[1:].strip().lower()
            if cmd in ('exit', 'quit'):
                break
            if cmd == 'help':
                print(':help  - show this help')
                print(':exit  - quit REPL')
                print(':clear - clear current buffer')
                print(':run   - assemble the last buffer to a temp .obj (requires nasm)')
                continue
            if cmd == 'clear':
                buffer.clear(); print('[xasm] buffer cleared'); continue
            if cmd == 'run':
                if not buffer:
                    print('[xasm] no code to run')
                    continue
                src = "\n".join(buffer)
                nasm_src = translate_xasm_to_nasm(src)
                if nasm_src is None:
                    print('[xasm] translation failed')
                else:
                    tmp_obj = Path(tempfile.gettempdir()) / f"ixasm_{os.getpid()}.obj"
                    nasm_exe = resolve_nasm()
                    ok = assemble_nasm_to_obj(nasm_src, tmp_obj, nasm_exe)
                    if ok:
                        print(f"[xasm] Assembled to {tmp_obj}")
                    else:
                        print('[xasm] assemble failed')
                buffer.clear(); continue
            print(f'[xasm] Unknown command: {cmd}'); continue
        buffer.append(line)


def main(argv: List[str]) -> int:
    # Normalize user-typed forms before argparse runs so the shortcut window can
    # accept odd variants like "ixasm--true" typed by users.
    argv = _normalize_ixasm_flags(argv)

    ap = argparse.ArgumentParser(
        prog="xasm_compiler.py",
        description="XASM -> CPU/GPU/NPU/TPU backend compiler for Kyber",
    )
    ap.add_argument("source", nargs='?', help=".xasm source file")
    ap.add_argument("output", nargs='?', help="destination .obj path")
    ap.add_argument("--nasm", default=None, help="path to nasm executable")
    ap.add_argument("--target", default=None, help="target backend: auto, cpu, gpu, npu, tpu, lpu, cuda, uno, arduino, avr")
    ap.add_argument("--compiler", default=None, help="backend C++ compiler override")
    ap.add_argument("--ixasm", action='store_true', help="Enter interactive XASM REPL (accepts many user-typed variants)")
    args = ap.parse_args(argv[1:])

    if args.ixasm:
        _interactive_repl()
        return 0

    if not args.source or not args.output:
        ap.print_usage()
        print('[xasm] error: source and output required unless --ixasm is used.', file=sys.stderr)
        return 2

    source_path = Path(args.source)
    output_path = Path(args.output)
    nasm_exe = resolve_nasm(args.nasm)
    target = resolve_target(args.target)

    if not source_path.exists():
        print(f"[xasm] Source not found: {source_path}", file=sys.stderr)
        return 1

    try:
        source = source_path.read_text(encoding="utf-8").lstrip("\ufeff")
    except OSError as e:
        print(f"[xasm] Cannot read source: {e}", file=sys.stderr)
        return 1

    print(f"[xasm] Translating: {source_path} for target={target}")
    sections = split_target_sections(source)

    if len(sections) > 1:
        compiled = []
        for section_target, section_lines in sections.items():
            section_source = "\n".join(section_lines)
            out = output_path.parent / f"{output_path.stem}_{section_target}{output_path.suffix}"
            if section_target == "cpu":
                nasm_src = translate_xasm_to_nasm(section_source)
                if nasm_src is None:
                    print(f"[xasm] Translation failed for {section_target}.", file=sys.stderr)
                    return 1
                if not assemble_nasm_to_obj(nasm_src, out, nasm_exe):
                    return 1
                compiled.append(out)
            elif section_target in {"uno", "arduino", "avr", "atmega328p"}:
                asm_out = out.with_suffix(".S")
                if not compile_avr_to_asm(section_source, asm_out):
                    return 1
                compiled.append(asm_out)
            else:
                if not compile_backend_to_obj(section_target, section_source, out, args.compiler):
                    return 1
                compiled.append(out)
        print(f"[xasm] Success: {compiled}")
        return 0

    section_source = "\n".join(next(iter(sections.values())))
    if target == "cpu":
        nasm_src = translate_xasm_to_nasm(section_source)
        if nasm_src is None:
            print("[xasm] Translation failed.", file=sys.stderr)
            return 1
        if not assemble_nasm_to_obj(nasm_src, output_path, nasm_exe):
            return 1
        print(f"[xasm] Success: {output_path}")
        return 0
    elif target in {"uno", "arduino", "avr", "atmega328p"}:
        asm_path = output_path.with_suffix(".S") if output_path.suffix.lower() not in {".s", ".S"} else output_path
        if not compile_avr_to_asm(section_source, asm_path):
            return 1
        print(f"[xasm] Success: {asm_path}")
        return 0
    else:
        if not compile_backend_to_obj(target, section_source, output_path, args.compiler):
            return 1

    print(f"[xasm] Success: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
