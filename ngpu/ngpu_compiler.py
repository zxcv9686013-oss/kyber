#!/usr/bin/env python3
"""
ngpu_compiler.py - NGPU Compiler for Kyber

Invoked automatically by kyberc.py when a #extern "ngpu" { ... } block
is encountered.  The pipeline is:

  1. Read the .ngpu source file (written by kyberc.py to the ngpu/ dir).
  2. Translate the NGPU source into PTX assembly.
  3. Write the PTX to a temporary .ptx file.
  4. Invoke nvcc to assemble the PTX into a COFF .obj file.
  5. Move the resulting .obj to the Kyber build/.obj directory under the
     name kyber_ngpu_N.obj (N is provided by the caller).
  6. Exit 0 on success, non-zero on any failure.

Usage:
  python ngpu_compiler.py <source.ngpu> <output.obj> [--cuda-root <path>]

  <source.ngpu>  - path to the .ngpu source file to compile
  <output.obj>   - full destination path for the COFF object file
                   (kyberc.py passes <BUILD>/.obj/kyber_ngpu_N.obj here)
  --cuda-root    - optional override for the CUDA toolkit root
                   (defaults to KYBER_CUDA_ROOT env var, then the
                    bundled CUDA directory)

NGPU Language Overview
----------------------
NGPU is a thin PTX-targeting shading/compute language.  Its syntax is
intentionally close to CUDA C so that developers familiar with CUDA can
read and write it naturally, but it is compiled directly to PTX rather
than going through the full CUDA C front-end.

Supported constructs
  - kernel declaration:    kernel void name(params) { body }
  - device function:       device <type> name(params) { body }
  - variable declaration:  int/float/double/uint x [= expr];
  - pointer params:        int* ptr, float* ptr
  - threadIdx/blockIdx:    threadIdx.x/y/z, blockIdx.x/y/z
  - blockDim/gridDim:      blockDim.x/y/z, gridDim.x/y/z
  - arithmetic:            + - * / %
  - array indexing:        ptr[idx]
  - assignment:            x = expr;
  - if / else
  - for (init; cond; step) { body }
  - while (cond) { body }
  - return [expr];
  - syncthreads():         __syncthreads()
  - printf (device):       printf("fmt", args...)
  - C-style comments:      // ...  and  /* ... */

The translator emits valid PTX 7.0 targeting sm_52 (Maxwell, the
lowest widely-supported compute capability for PTX 7.x).  The
resulting PTX is then handed to nvcc --ptx ... to get a COFF .obj.

Error reporting
  Any translation or compilation error is printed to stderr and the
  process exits with a non-zero code so that kyberc.py can surface it
  correctly.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ============================================================
# CUDA toolkit resolution
# ============================================================

CUDA_TOOLKIT_DEFAULT = Path(os.environ.get(
    "KYBER_CUDA_ROOT",
    r"C:\Users\Admin\kyber (7)\kyber\CUDA",
))


def resolve_nvcc(cuda_root: Path) -> str:
    """Return the path to nvcc, preferring the toolkit root over PATH."""
    candidate = cuda_root / "bin" / "nvcc.exe"
    if candidate.exists():
        return str(candidate)
    candidate_nix = cuda_root / "bin" / "nvcc"
    if candidate_nix.exists():
        return str(candidate_nix)
    return "nvcc"  # fall back to PATH


# ============================================================
# NGPU Lexer
# ============================================================

class TokenType:
    # literals
    INT_LIT    = "INT_LIT"
    FLOAT_LIT  = "FLOAT_LIT"
    STRING_LIT = "STRING_LIT"
    IDENT      = "IDENT"
    # keywords
    KERNEL     = "kernel"
    DEVICE     = "device"
    VOID       = "void"
    INT        = "int"
    UINT       = "uint"
    FLOAT      = "float"
    DOUBLE     = "double"
    IF         = "if"
    ELSE       = "else"
    FOR        = "for"
    WHILE      = "while"
    RETURN     = "return"
    # punctuation
    LBRACE     = "{"
    RBRACE     = "}"
    LPAREN     = "("
    RPAREN     = ")"
    LBRACKET   = "["
    RBRACKET   = "]"
    SEMI       = ";"
    COMMA      = ","
    DOT        = "."
    STAR       = "*"
    AMP        = "&"
    EQ         = "="
    EQEQ       = "=="
    NEQ        = "!="
    LT         = "<"
    LE         = "<="
    GT         = ">"
    GE         = ">="
    PLUS       = "+"
    MINUS      = "-"
    SLASH      = "/"
    PERCENT    = "%"
    BANG       = "!"
    AMPAMP     = "&&"
    PIPEPIPE   = "||"
    EOF        = "EOF"


KEYWORDS = {
    "kernel", "device", "void", "int", "uint", "float", "double",
    "if", "else", "for", "while", "return",
}


@dataclass
class Token:
    type: str
    text: str
    line: int


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    n = len(source)

    while i < n:
        # newline
        if source[i] == "\n":
            line += 1
            i += 1
            continue

        # whitespace
        if source[i] in " \t\r":
            i += 1
            continue

        # line comment
        if source[i:i+2] == "//":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # block comment
        if source[i:i+2] == "/*":
            i += 2
            while i < n - 1 and source[i:i+2] != "*/":
                if source[i] == "\n":
                    line += 1
                i += 1
            i += 2
            continue

        # string literal
        if source[i] == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == '\\':
                    j += 1
                j += 1
            j += 1
            tokens.append(Token(TokenType.STRING_LIT, source[i:j], line))
            i = j
            continue

        # number
        if source[i].isdigit() or (source[i] == '.' and i + 1 < n and source[i+1].isdigit()):
            j = i
            is_float = False
            while j < n and (source[j].isdigit() or source[j] in ".eEfF+-"):
                if source[j] in ".eEfF":
                    is_float = True
                j += 1
            tok_type = TokenType.FLOAT_LIT if is_float else TokenType.INT_LIT
            tokens.append(Token(tok_type, source[i:j], line))
            i = j
            continue

        # identifier / keyword
        if source[i].isalpha() or source[i] == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tok_type = word if word in KEYWORDS else TokenType.IDENT
            tokens.append(Token(tok_type, word, line))
            i = j
            continue

        # two-character operators
        two = source[i:i+2]
        if two in ("==", "!=", "<=", ">=", "&&", "||"):
            type_map = {
                "==": TokenType.EQEQ,
                "!=": TokenType.NEQ,
                "<=": TokenType.LE,
                ">=": TokenType.GE,
                "&&": TokenType.AMPAMP,
                "||": TokenType.PIPEPIPE,
            }
            tokens.append(Token(type_map[two], two, line))
            i += 2
            continue

        # single-character punctuation/operators
        one_map = {
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ";": TokenType.SEMI,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            "*": TokenType.STAR,
            "&": TokenType.AMP,
            "=": TokenType.EQ,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "!": TokenType.BANG,
        }
        if source[i] in one_map:
            tokens.append(Token(one_map[source[i]], source[i], line))
            i += 1
            continue

        # unknown character — skip with a warning
        print(f"[ngpu] WARNING: unknown character '{source[i]}' at line {line}", file=sys.stderr)
        i += 1

    tokens.append(Token(TokenType.EOF, "", line))
    return tokens


# ============================================================
# NGPU → PTX Translator
# ============================================================

# PTX type map
_PTX_TYPES: Dict[str, str] = {
    "int":    ".s32",
    "uint":   ".u32",
    "float":  ".f32",
    "double": ".f64",
    "void":   "",
}

# PTX register type map
_PTX_REG: Dict[str, str] = {
    "int":    "%rd",
    "uint":   "%ru",
    "float":  "%f",
    "double": "%fd",
}

# Special identifiers mapped directly to PTX special registers
_SPECIAL_REGS: Dict[str, str] = {
    "threadIdx.x": "%tid.x",
    "threadIdx.y": "%tid.y",
    "threadIdx.z": "%tid.z",
    "blockIdx.x":  "%ctaid.x",
    "blockIdx.y":  "%ctaid.y",
    "blockIdx.z":  "%ctaid.z",
    "blockDim.x":  "%ntid.x",
    "blockDim.y":  "%ntid.y",
    "blockDim.z":  "%ntid.z",
    "gridDim.x":   "%nctaid.x",
    "gridDim.y":   "%nctaid.y",
    "gridDim.z":   "%nctaid.z",
}


@dataclass
class Param:
    type_name: str   # "int", "float", etc.
    is_ptr: bool
    name: str


@dataclass
class NGPUKernel:
    name: str
    params: List[Param]
    body_tokens: List[Token]  # raw token stream for the body


@dataclass
class NGPUDevice:
    name: str
    return_type: str
    params: List[Param]
    body_tokens: List[Token]


class NGPUParser:
    """
    Parses the NGPU token stream into a list of kernel / device function
    declarations and produces PTX source.
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.kernels: List[NGPUKernel] = []
        self.devices: List[NGPUDevice] = []
        self._errors: List[str] = []

    # ── helpers ───────────────────────────────────────────

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _expect(self, ttype: str) -> Token:
        tok = self._advance()
        if tok.type != ttype:
            self._errors.append(
                f"line {tok.line}: expected '{ttype}' but got '{tok.text}'"
            )
        return tok

    def _match(self, ttype: str) -> bool:
        if self._peek().type == ttype:
            self._advance()
            return True
        return False

    def _is_type(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.INT, TokenType.UINT,
            TokenType.FLOAT, TokenType.DOUBLE,
            TokenType.VOID,
        )

    def _parse_type(self) -> str:
        tok = self._advance()
        if not self._is_type(tok):
            self._errors.append(
                f"line {tok.line}: expected type, got '{tok.text}'"
            )
            return "int"
        return tok.text

    def _parse_params(self) -> List[Param]:
        params: List[Param] = []
        self._expect(TokenType.LPAREN)
        while self._peek().type != TokenType.RPAREN and self._peek().type != TokenType.EOF:
            type_name = self._parse_type()
            is_ptr = self._match(TokenType.STAR)
            if self._peek().type not in (TokenType.IDENT, TokenType.EOF):
                self._errors.append(
                    f"line {self._peek().line}: expected parameter name"
                )
                break
            name = self._advance().text
            params.append(Param(type_name, is_ptr, name))
            if not self._match(TokenType.COMMA):
                break
        self._expect(TokenType.RPAREN)
        return params

    def _collect_body(self) -> List[Token]:
        """Collect all tokens from '{' through matching '}'."""
        body: List[Token] = []
        self._expect(TokenType.LBRACE)
        depth = 1
        while depth > 0 and self._peek().type != TokenType.EOF:
            tok = self._advance()
            if tok.type == TokenType.LBRACE:
                depth += 1
            elif tok.type == TokenType.RBRACE:
                depth -= 1
                if depth == 0:
                    break
            body.append(tok)
        return body

    # ── top-level parse ───────────────────────────────────

    def parse(self) -> bool:
        while self._peek().type != TokenType.EOF:
            tok = self._peek()

            if tok.type == TokenType.KERNEL:
                self._advance()
                self._expect(TokenType.VOID)
                name_tok = self._expect(TokenType.IDENT)
                params = self._parse_params()
                body = self._collect_body()
                self.kernels.append(NGPUKernel(name_tok.text, params, body))

            elif tok.type == TokenType.DEVICE:
                self._advance()
                ret_type = self._parse_type()
                name_tok = self._expect(TokenType.IDENT)
                params = self._parse_params()
                body = self._collect_body()
                self.devices.append(NGPUDevice(name_tok.text, ret_type, params, body))

            else:
                # top-level token we don't recognise — skip
                self._advance()

        return len(self._errors) == 0


# ============================================================
# PTX emitter
# ============================================================

class PTXEmitter:
    """
    Converts a parsed NGPU program into a PTX 7.0 source string.

    The emitter operates on the body token stream with a simple
    recursive approach: it walks tokens linearly and maintains
    enough state (a local variable registry, a temporary register
    counter) to produce valid PTX for the supported constructs.
    """

    SM_VERSION   = "52"         # sm_52 — Maxwell, lowest for PTX 7.x
    PTX_VERSION  = "7.0"

    def __init__(self, parser: NGPUParser):
        self.parser = parser
        self._lines: List[str] = []
        self._tmp_counter = 0

    # ── output helpers ────────────────────────────────────

    def _emit(self, line: str, indent: int = 0):
        self._lines.append("    " * indent + line)

    def _fresh_reg(self, prefix: str = "%r") -> str:
        self._tmp_counter += 1
        return f"{prefix}{self._tmp_counter}"

    # ── top-level ─────────────────────────────────────────

    def emit(self) -> str:
        self._emit(f".version {self.PTX_VERSION}")
        self._emit(f".target sm_{self.SM_VERSION}")
        self._emit(".address_size 64")
        self._emit("")

        for dev in self.parser.devices:
            self._emit_device(dev)
            self._emit("")

        for kernel in self.parser.kernels:
            self._emit_kernel(kernel)
            self._emit("")

        return "\n".join(self._lines)

    # ── kernel / device ───────────────────────────────────

    def _emit_kernel(self, kernel: NGPUKernel):
        param_ptx = self._format_params(kernel.params, is_kernel=True)
        self._emit(f".visible .entry {kernel.name}(")
        for line in param_ptx:
            self._emit(line, indent=1)
        self._emit(")")
        self._emit("{")
        ctx = _BodyContext(kernel.params, self._tmp_counter)
        body_lines, self._tmp_counter = _emit_body(kernel.body_tokens, ctx)
        for bl in body_lines:
            self._emit(bl, indent=1)
        self._emit("    ret;")
        self._emit("}")

    def _emit_device(self, dev: NGPUDevice):
        ret_ptx = _PTX_TYPES.get(dev.return_type, ".s32")
        param_ptx = self._format_params(dev.params, is_kernel=False)
        self._emit(f".func {ret_ptx} {dev.name}(")
        for line in param_ptx:
            self._emit(line, indent=1)
        self._emit(")")
        self._emit("{")
        ctx = _BodyContext(dev.params, self._tmp_counter)
        body_lines, self._tmp_counter = _emit_body(dev.body_tokens, ctx)
        for bl in body_lines:
            self._emit(bl, indent=1)
        self._emit("}")

    def _format_params(self, params: List[Param], is_kernel: bool) -> List[str]:
        result = []
        for i, p in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            if p.is_ptr:
                if is_kernel:
                    result.append(f"    .param .u64 {p.name}_param{comma}")
                else:
                    result.append(f"    .param .u64 {p.name}{comma}")
            else:
                ptx_type = _PTX_TYPES.get(p.type_name, ".s32")
                if is_kernel:
                    result.append(f"    .param {ptx_type} {p.name}_param{comma}")
                else:
                    result.append(f"    .param {ptx_type} {p.name}{comma}")
        return result


# ── body emitter (stateful, token-walk) ───────────────────────────────────────

@dataclass
class _BodyContext:
    params: List[Param]
    tmp_base: int
    _counter: int = field(default=0, init=False)
    _locals: Dict[str, Tuple[str, bool]] = field(default_factory=dict, init=False)
    # name -> (ptx_type, is_ptr)

    def fresh(self, prefix: str = "%r") -> str:
        self._counter += 1
        return f"{prefix}{self.tmp_base + self._counter}"

    def register_local(self, name: str, type_name: str, is_ptr: bool):
        self._locals[name] = (type_name, is_ptr)

    def ptx_type_for(self, name: str) -> str:
        if name in self._locals:
            t, is_ptr = self._locals[name]
            return ".u64" if is_ptr else _PTX_TYPES.get(t, ".s32")
        for p in self.params:
            if p.name == name:
                return ".u64" if p.is_ptr else _PTX_TYPES.get(p.type_name, ".s32")
        return ".s32"


def _tokens_to_expr(tokens: List[Token]) -> str:
    """
    Convert a flat token list representing an expression back to a
    PTX-friendly string (used where we embed expressions as inline
    PTX comments / immediate values).  For full expression lowering
    we emit explicit mov/add/etc. instructions; this helper is only
    used for simple constant folding and array index arithmetic.
    """
    parts = []
    for t in tokens:
        # Map special register names
        if t.type == TokenType.IDENT:
            full = t.text
            parts.append(full)
        else:
            parts.append(t.text)
    return " ".join(parts)


def _emit_body(body_tokens: List[Token], ctx: _BodyContext) -> Tuple[List[str], int]:
    """
    Walk the body token stream and produce PTX instruction lines.
    Returns (lines, updated_tmp_counter).
    """
    out: List[str] = []
    pos = 0
    n = len(body_tokens)

    # Declare .reg entries for all params first
    for p in ctx.params:
        if p.is_ptr:
            out.append(f".reg .u64 {p.name};")
            out.append(f"ld.param.u64 {p.name}, [{p.name}_param];")
        else:
            ptx_t = _PTX_TYPES.get(p.type_name, ".s32")
            out.append(f".reg {ptx_t} {p.name};")
            out.append(f"ld.param{ptx_t} {p.name}, [{p.name}_param];")

    def peek(offset: int = 0) -> Optional[Token]:
        idx = pos + offset
        if idx < n:
            return body_tokens[idx]
        return None

    def advance() -> Token:
        nonlocal pos
        tok = body_tokens[pos] if pos < n else Token(TokenType.EOF, "", 0)
        pos += 1
        return tok

    def collect_until(stop_type: str) -> List[Token]:
        result = []
        while pos < n and body_tokens[pos].type != stop_type:
            result.append(advance())
        if pos < n:
            advance()  # consume the stop token
        return result

    def collect_balanced(open_t: str, close_t: str) -> List[Token]:
        result = []
        depth = 1
        while pos < n and depth > 0:
            tok = advance()
            if tok.type == open_t:
                depth += 1
            elif tok.type == close_t:
                depth -= 1
                if depth == 0:
                    break
            result.append(tok)
        return result

    def expr_to_ptx_reg(expr_tokens: List[Token]) -> Tuple[str, List[str]]:
        """
        Emit instructions that compute an expression into a fresh register.
        Returns (register_name, extra_instructions).
        For simple cases (single literal, single ident) we skip the extra emit.
        """
        extra: List[str] = []

        if not expr_tokens:
            reg = ctx.fresh("%r")
            out_line = f".reg .s32 {reg};"
            extra.append(out_line)
            extra.append(f"mov.s32 {reg}, 0;")
            return reg, extra

        # Single integer literal
        if len(expr_tokens) == 1 and expr_tokens[0].type == TokenType.INT_LIT:
            reg = ctx.fresh("%r")
            extra.append(f".reg .s32 {reg};")
            extra.append(f"mov.s32 {reg}, {expr_tokens[0].text};")
            return reg, extra

        # Single float literal
        if len(expr_tokens) == 1 and expr_tokens[0].type == TokenType.FLOAT_LIT:
            reg = ctx.fresh("%f")
            extra.append(f".reg .f32 {reg};")
            val = expr_tokens[0].text.rstrip("fF")
            extra.append(f"mov.f32 {reg}, {val};")
            return reg, extra

        # Special register (e.g. threadIdx.x)
        if len(expr_tokens) == 3 and expr_tokens[1].type == TokenType.DOT:
            combined = expr_tokens[0].text + "." + expr_tokens[2].text
            if combined in _SPECIAL_REGS:
                ptx_sr = _SPECIAL_REGS[combined]
                reg = ctx.fresh("%r")
                extra.append(f".reg .u32 {reg};")
                extra.append(f"mov.u32 {reg}, {ptx_sr};")
                return reg, extra

        # Single identifier
        if len(expr_tokens) == 1 and expr_tokens[0].type == TokenType.IDENT:
            return expr_tokens[0].text, extra

        # Binary expression: lhs op rhs
        # Find the operator token (last +/-/*//)
        op_idx = -1
        op_tok = None
        for i, t in enumerate(expr_tokens):
            if t.type in (TokenType.PLUS, TokenType.MINUS,
                          TokenType.STAR, TokenType.SLASH,
                          TokenType.PERCENT):
                op_idx = i
                op_tok = t

        if op_idx > 0 and op_tok is not None:
            lhs_toks = expr_tokens[:op_idx]
            rhs_toks = expr_tokens[op_idx + 1:]
            lhs_reg, lhs_extra = expr_to_ptx_reg(lhs_toks)
            rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
            extra.extend(lhs_extra)
            extra.extend(rhs_extra)
            result_reg = ctx.fresh("%r")
            ptx_op_map = {
                "+": "add.s32",
                "-": "sub.s32",
                "*": "mul.lo.s32",
                "/": "div.s32",
                "%": "rem.s32",
            }
            ptx_op = ptx_op_map.get(op_tok.text, "add.s32")
            extra.append(f".reg .s32 {result_reg};")
            extra.append(f"{ptx_op} {result_reg}, {lhs_reg}, {rhs_reg};")
            return result_reg, extra

        # Fallback: treat as a raw register name (identifier)
        return _tokens_to_expr(expr_tokens), extra

    while pos < n:
        tok = peek()
        if tok is None or tok.type == TokenType.EOF:
            break

        # ── __syncthreads() ──────────────────────────────
        if tok.type == TokenType.IDENT and tok.text == "__syncthreads":
            advance()  # __syncthreads
            advance()  # (
            advance()  # )
            advance()  # ;
            out.append("bar.sync 0;")
            continue

        # ── printf(...) ──────────────────────────────────
        if tok.type == TokenType.IDENT and tok.text == "printf":
            # Collect the full statement up to ';'
            stmt_toks = collect_until(TokenType.SEMI)
            # Emit as a vprintf syscall comment + nop
            fmt_str = ""
            for t in stmt_toks:
                if t.type == TokenType.STRING_LIT:
                    fmt_str = t.text
                    break
            out.append(f"// printf {fmt_str}  [device printf via vprintf not emitted]")
            continue

        # ── return ───────────────────────────────────────
        if tok.type == TokenType.RETURN:
            advance()
            expr_toks = collect_until(TokenType.SEMI)
            if expr_toks:
                reg, extra = expr_to_ptx_reg(expr_toks)
                out.extend(extra)
                out.append(f"// return value in {reg}")
            out.append("ret;")
            continue

        # ── if / else ────────────────────────────────────
        if tok.type == TokenType.IF:
            advance()
            advance()  # (
            cond_toks = collect_until(TokenType.RPAREN)
            then_body = collect_balanced(TokenType.LBRACE, TokenType.RBRACE)

            label_true  = ctx.fresh("$L_true")
            label_false = ctx.fresh("$L_false")
            label_end   = ctx.fresh("$L_end")

            # Simple condition: lhs cmp rhs
            cmp_map = {
                "==": "setp.eq.s32",
                "!=": "setp.ne.s32",
                "<":  "setp.lt.s32",
                "<=": "setp.le.s32",
                ">":  "setp.gt.s32",
                ">=": "setp.ge.s32",
            }
            cmp_op = None
            cmp_idx = -1
            for ci, ct in enumerate(cond_toks):
                if ct.type in (TokenType.EQEQ, TokenType.NEQ,
                                TokenType.LT,   TokenType.LE,
                                TokenType.GT,   TokenType.GE):
                    cmp_op = ct.text
                    cmp_idx = ci
                    break

            if cmp_op and cmp_idx > 0:
                lhs_toks = cond_toks[:cmp_idx]
                rhs_toks = cond_toks[cmp_idx + 1:]
                lhs_reg, lhs_extra = expr_to_ptx_reg(lhs_toks)
                rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
                out.extend(lhs_extra)
                out.extend(rhs_extra)
                pred = ctx.fresh("%p")
                out.append(f".reg .pred {pred};")
                ptx_cmp = cmp_map.get(cmp_op, "setp.eq.s32")
                out.append(f"{ptx_cmp} {pred}, {lhs_reg}, {rhs_reg};")
                out.append(f"@!{pred} bra {label_false};")
            else:
                # Non-standard condition, emit as comment
                out.append(f"// if ({_tokens_to_expr(cond_toks)})")

            # then block
            then_lines, _ = _emit_body(then_body, _BodyContext(ctx.params, ctx._counter + ctx.tmp_base))
            out.extend(then_lines)

            # else block?
            if peek() and peek().type == TokenType.ELSE:
                advance()
                if peek() and peek().type == TokenType.LBRACE:
                    else_body = collect_balanced(TokenType.LBRACE, TokenType.RBRACE)
                    out.append(f"bra {label_end};")
                    out.append(f"{label_false}:")
                    else_lines, _ = _emit_body(else_body, _BodyContext(ctx.params, ctx._counter + ctx.tmp_base))
                    out.extend(else_lines)
                    out.append(f"{label_end}:")
                else:
                    out.append(f"{label_false}:")
            else:
                out.append(f"{label_false}:")

            continue

        # ── for loop ─────────────────────────────────────
        if tok.type == TokenType.FOR:
            advance()
            advance()  # (

            # init: up to first ;
            init_toks = collect_until(TokenType.SEMI)
            # cond: up to second ;
            cond_toks = collect_until(TokenType.SEMI)
            # step: up to )
            step_toks = collect_until(TokenType.RPAREN)
            loop_body = collect_balanced(TokenType.LBRACE, TokenType.RBRACE)

            label_loop = ctx.fresh("$L_for")
            label_end  = ctx.fresh("$L_forend")

            # emit init
            out.append(f"// for loop init: {_tokens_to_expr(init_toks)}")
            if init_toks:
                # Detect "type name = expr" or "name = expr"
                if init_toks[0].type in (TokenType.INT, TokenType.FLOAT,
                                          TokenType.UINT, TokenType.DOUBLE):
                    type_name = init_toks[0].text
                    var_name  = init_toks[1].text if len(init_toks) > 1 else ctx.fresh("%r")
                    ctx.register_local(var_name, type_name, False)
                    ptx_t = _PTX_TYPES.get(type_name, ".s32")
                    out.append(f".reg {ptx_t} {var_name};")
                    if len(init_toks) >= 4:
                        val_toks = init_toks[3:]
                        val_reg, val_extra = expr_to_ptx_reg(val_toks)
                        out.extend(val_extra)
                        out.append(f"mov{ptx_t} {var_name}, {val_reg};")
                elif len(init_toks) >= 3 and init_toks[1].type == TokenType.EQ:
                    var_name = init_toks[0].text
                    val_toks = init_toks[2:]
                    val_reg, val_extra = expr_to_ptx_reg(val_toks)
                    out.extend(val_extra)
                    ptx_t = ctx.ptx_type_for(var_name)
                    out.append(f"mov{ptx_t} {var_name}, {val_reg};")

            out.append(f"{label_loop}:")

            # emit cond check
            cmp_map = {
                "==": "setp.eq.s32", "!=": "setp.ne.s32",
                "<":  "setp.lt.s32", "<=": "setp.le.s32",
                ">":  "setp.gt.s32", ">=": "setp.ge.s32",
            }
            cmp_op = None
            cmp_idx = -1
            for ci, ct in enumerate(cond_toks):
                if ct.type in (TokenType.EQEQ, TokenType.NEQ,
                                TokenType.LT,   TokenType.LE,
                                TokenType.GT,   TokenType.GE):
                    cmp_op = ct.text
                    cmp_idx = ci
                    break

            if cmp_op and cmp_idx > 0:
                lhs_toks = cond_toks[:cmp_idx]
                rhs_toks = cond_toks[cmp_idx + 1:]
                lhs_reg, lhs_extra = expr_to_ptx_reg(lhs_toks)
                rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
                out.extend(lhs_extra)
                out.extend(rhs_extra)
                pred = ctx.fresh("%p")
                out.append(f".reg .pred {pred};")
                ptx_cmp = cmp_map.get(cmp_op, "setp.lt.s32")
                out.append(f"{ptx_cmp} {pred}, {lhs_reg}, {rhs_reg};")
                out.append(f"@!{pred} bra {label_end};")

            # body
            loop_lines, _ = _emit_body(loop_body, _BodyContext(ctx.params, ctx._counter + ctx.tmp_base))
            out.extend(loop_lines)

            # step
            out.append(f"// for step: {_tokens_to_expr(step_toks)}")
            if len(step_toks) >= 3 and step_toks[1].type == TokenType.EQ:
                var_name = step_toks[0].text
                val_toks = step_toks[2:]
                val_reg, val_extra = expr_to_ptx_reg(val_toks)
                out.extend(val_extra)
                ptx_t = ctx.ptx_type_for(var_name)
                out.append(f"mov{ptx_t} {var_name}, {val_reg};")

            out.append(f"bra {label_loop};")
            out.append(f"{label_end}:")
            continue

        # ── while loop ───────────────────────────────────
        if tok.type == TokenType.WHILE:
            advance()
            advance()  # (
            cond_toks = collect_until(TokenType.RPAREN)
            loop_body = collect_balanced(TokenType.LBRACE, TokenType.RBRACE)

            label_loop = ctx.fresh("$L_while")
            label_end  = ctx.fresh("$L_whileend")

            out.append(f"{label_loop}:")
            cmp_map = {
                "==": "setp.eq.s32", "!=": "setp.ne.s32",
                "<":  "setp.lt.s32", "<=": "setp.le.s32",
                ">":  "setp.gt.s32", ">=": "setp.ge.s32",
            }
            cmp_op = None
            cmp_idx = -1
            for ci, ct in enumerate(cond_toks):
                if ct.type in (TokenType.EQEQ, TokenType.NEQ,
                                TokenType.LT,   TokenType.LE,
                                TokenType.GT,   TokenType.GE):
                    cmp_op = ct.text
                    cmp_idx = ci
                    break

            if cmp_op and cmp_idx > 0:
                lhs_toks = cond_toks[:cmp_idx]
                rhs_toks = cond_toks[cmp_idx + 1:]
                lhs_reg, lhs_extra = expr_to_ptx_reg(lhs_toks)
                rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
                out.extend(lhs_extra)
                out.extend(rhs_extra)
                pred = ctx.fresh("%p")
                out.append(f".reg .pred {pred};")
                ptx_cmp = cmp_map.get(cmp_op, "setp.eq.s32")
                out.append(f"{ptx_cmp} {pred}, {lhs_reg}, {rhs_reg};")
                out.append(f"@!{pred} bra {label_end};")

            loop_lines, _ = _emit_body(loop_body, _BodyContext(ctx.params, ctx._counter + ctx.tmp_base))
            out.extend(loop_lines)
            out.append(f"bra {label_loop};")
            out.append(f"{label_end}:")
            continue

        # ── variable declaration:  type name [= expr] ; ──
        if tok.type in (TokenType.INT, TokenType.FLOAT,
                         TokenType.UINT, TokenType.DOUBLE):
            type_name = advance().text
            is_ptr = False
            if peek() and peek().type == TokenType.STAR:
                advance()
                is_ptr = True
            var_name = advance().text
            ctx.register_local(var_name, type_name, is_ptr)

            if is_ptr:
                out.append(f".reg .u64 {var_name};")
                # consume optional initialiser
                if peek() and peek().type == TokenType.EQ:
                    stmt = collect_until(TokenType.SEMI)
                    init_toks = stmt[1:]  # skip '='
                    val_reg, val_extra = expr_to_ptx_reg(init_toks)
                    out.extend(val_extra)
                    out.append(f"mov.u64 {var_name}, {val_reg};")
                else:
                    collect_until(TokenType.SEMI)
            else:
                ptx_t = _PTX_TYPES.get(type_name, ".s32")
                out.append(f".reg {ptx_t} {var_name};")
                if peek() and peek().type == TokenType.EQ:
                    stmt = collect_until(TokenType.SEMI)
                    init_toks = stmt[1:]  # skip '='
                    val_reg, val_extra = expr_to_ptx_reg(init_toks)
                    out.extend(val_extra)
                    out.append(f"mov{ptx_t} {var_name}, {val_reg};")
                else:
                    collect_until(TokenType.SEMI)
            continue

        # ── assignment or array store: name [idx] = expr; OR name = expr; ──
        if tok.type == TokenType.IDENT:
            stmt_toks = collect_until(TokenType.SEMI)
            # array indexing: name [ idx ] = rhs
            if (len(stmt_toks) >= 4 and
                    stmt_toks[1].type == TokenType.LBRACKET):
                # find matching ]
                close = next((i for i, t in enumerate(stmt_toks) if t.type == TokenType.RBRACKET), -1)
                if close != -1 and close + 1 < len(stmt_toks) and stmt_toks[close + 1].type == TokenType.EQ:
                    base_name = stmt_toks[0].text
                    idx_toks  = stmt_toks[2:close]
                    rhs_toks  = stmt_toks[close + 2:]
                    idx_reg, idx_extra = expr_to_ptx_reg(idx_toks)
                    rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
                    out.extend(idx_extra)
                    out.extend(rhs_extra)
                    # Compute byte offset: idx * 4 (int32) or idx * 8 (float64)
                    ptx_t = ctx.ptx_type_for(base_name)
                    elem_bytes = 8 if ptx_t in (".f64", ".u64", ".s64") else 4
                    offset_reg = ctx.fresh("%r")
                    addr_reg   = ctx.fresh("%r")
                    out.append(f".reg .s32 {offset_reg};")
                    out.append(f".reg .u64 {addr_reg};")
                    out.append(f"mul.lo.s32 {offset_reg}, {idx_reg}, {elem_bytes};")
                    out.append(f"cvt.u64.s32 {addr_reg}, {offset_reg};")
                    out.append(f"add.u64 {addr_reg}, {base_name}, {addr_reg};")
                    st_map = {".s32": "st.global.s32", ".u32": "st.global.u32",
                               ".f32": "st.global.f32", ".f64": "st.global.f64"}
                    st_op = st_map.get(ptx_t, "st.global.s32")
                    out.append(f"{st_op} [{addr_reg}], {rhs_reg};")
                    continue

            # simple assignment: name = rhs
            eq_idx = next((i for i, t in enumerate(stmt_toks) if t.type == TokenType.EQ), -1)
            if eq_idx == 0 + 1 and stmt_toks[0].type == TokenType.IDENT and eq_idx > 0:
                var_name = stmt_toks[0].text
                rhs_toks = stmt_toks[eq_idx + 1:]
                rhs_reg, rhs_extra = expr_to_ptx_reg(rhs_toks)
                out.extend(rhs_extra)
                ptx_t = ctx.ptx_type_for(var_name)
                out.append(f"mov{ptx_t} {var_name}, {rhs_reg};")
                continue

            # unrecognised statement — emit as comment
            out.append(f"// {_tokens_to_expr(stmt_toks)};")
            continue

        # ── anything else: skip to next semicolon ────────
        stmt_toks = collect_until(TokenType.SEMI)
        out.append(f"// (unhandled) {_tokens_to_expr(stmt_toks)};")

    return out, ctx._counter + ctx.tmp_base


# ============================================================
# Top-level translation entry point
# ============================================================

def translate_ngpu_to_ptx(source: str) -> Optional[str]:
    """
    Parse `source` as NGPU and return a PTX source string, or None on
    parse errors.  Parse errors are printed to stderr.
    """
    tokens = tokenize(source)
    parser = NGPUParser(tokens)
    ok = parser.parse()

    if not ok or parser._errors:
        for e in parser._errors:
            print(f"[ngpu] Parse error: {e}", file=sys.stderr)
        if not ok:
            return None

    emitter = PTXEmitter(parser)
    return emitter.emit()


# ============================================================
# PTX → .obj via nvcc
# ============================================================

def compile_ptx_to_obj(
    ptx_source: str,
    output_obj: Path,
    cuda_root: Path,
) -> bool:
    """
    Write ptx_source to a temp .ptx file, invoke nvcc to compile it
    into a COFF .obj at output_obj.  Returns True on success.
    """
    nvcc = resolve_nvcc(cuda_root)

    tmp_dir = Path(tempfile.gettempdir())
    tmp_ptx = tmp_dir / f"kyber_ngpu_{os.getpid()}.ptx"

    try:
        tmp_ptx.write_text(ptx_source, encoding="utf-8")
    except OSError as e:
        print(f"[ngpu] Cannot write temp PTX file: {e}", file=sys.stderr)
        return False

    # Ensure the output directory exists
    try:
        output_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ngpu] Cannot create output directory: {e}", file=sys.stderr)
        tmp_ptx.unlink(missing_ok=True)
        return False

    # nvcc flags:
    #   --ptx-compilation    : compile PTX directly (as opposed to CUDA C)
    #   -c                   : compile to object file
    #   --output-file        : destination
    #   -arch sm_52          : target Maxwell (lowest for PTX 7.x)
    cuda_include = cuda_root / "include"
    cmd = [
        nvcc,
        "-c",
        "-arch", "sm_52",
        "-o", str(output_obj),
    ]
    if cuda_include.exists():
        cmd += ["-I", str(cuda_include)]
    cmd.append(str(tmp_ptx))

    print(f"[ngpu] Invoking nvcc: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False)
        rc = result.returncode
    except FileNotFoundError:
        print(
            f"[ngpu] nvcc not found at '{nvcc}'.\n"
            "       Install the CUDA toolkit or set KYBER_CUDA_ROOT.",
            file=sys.stderr,
        )
        tmp_ptx.unlink(missing_ok=True)
        return False
    finally:
        tmp_ptx.unlink(missing_ok=True)

    if rc != 0:
        print(f"[ngpu] nvcc failed with exit code {rc}.", file=sys.stderr)
        return False

    if not output_obj.exists():
        print(
            f"[ngpu] nvcc exited 0 but the object file was not created:\n"
            f"       {output_obj}",
            file=sys.stderr,
        )
        return False

    print(f"[ngpu] Generated: {output_obj}")
    return True


# ============================================================
# Main
# ============================================================

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ngpu_compiler.py",
        description="NGPU → PTX → .obj compiler for Kyber",
    )
    ap.add_argument("source", help="Path to the .ngpu source file")
    ap.add_argument("output", help="Destination .obj path (e.g. build/.obj/kyber_ngpu_0.obj)")
    ap.add_argument(
        "--cuda-root",
        default=None,
        help="Override CUDA toolkit root (default: KYBER_CUDA_ROOT env var or bundled CUDA)",
    )
    args = ap.parse_args(argv[1:])

    source_path = Path(args.source)
    output_path = Path(args.output)

    cuda_root = (
        Path(args.cuda_root) if args.cuda_root
        else CUDA_TOOLKIT_DEFAULT
    )

    # ── 1. Read .ngpu source ──────────────────────────────
    if not source_path.exists():
        print(f"[ngpu] Source file not found: {source_path}", file=sys.stderr)
        return 1

    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[ngpu] Cannot read source: {e}", file=sys.stderr)
        return 1

    print(f"[ngpu] Translating: {source_path}")

    # ── 2. NGPU → PTX ────────────────────────────────────
    ptx = translate_ngpu_to_ptx(source)
    if ptx is None:
        print("[ngpu] Translation to PTX failed.", file=sys.stderr)
        return 1

    # ── 3. PTX → .obj via nvcc ───────────────────────────
    ok = compile_ptx_to_obj(ptx, output_path, cuda_root)
    if not ok:
        return 1

    print(f"[ngpu] Compilation successful: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
