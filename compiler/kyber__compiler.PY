#!/usr/bin/env python3
"""
kyberc.py - Kyber Compiler (Python port of the original C++ kyberc.cpp)

This is a straight port of the original C++ Kyber compiler to Python.
Nothing from the original has been removed - lexer, AST, parser,
#extern / #n8n / #ninja block extraction, C-source generation for
those blocks, clang invocation, and LLVM object-file emission are all
still here (LLVM emission now goes through `llvmlite` instead of the
native LLVM C++ API).

WHAT'S NEW vs. the C++ file
----------------------------
The C++ file already supported three kinds of embedded blocks:

    #extern "c" / #extern "c++"   -> compiled directly with clang
    #n8n "name" { ... }           -> compiled into a libcurl webhook caller
    #ninja "name" { ... }         -> wrapped so it shells out to
                                      "python3 NINJA--COMPILER.py <file>"
                                      at runtime

But "ninja" itself was never added to the lexer's KEYWORDS table the
way "extern" was (extern is both a preprocessor marker *and* a real
Kyber keyword; ninja/n8n were only ever preprocessor markers). And
there was no block type at all that targeted the actual Python
interpreter this project ships with (py_ninja.py / PY_NINJA), only
the hypothetical "NINJA--COMPILER.py".

This port fixes both gaps:

  * "ninja" and "pyninja" are now real entries in KEYWORDS, just like
    "extern" already was.
  * A brand-new #pyninja block/command is added, parallel to #ninja:

        #pyninja "run_forward_pass"
        {
            VOID MAIN() {
                SCREEN.PRINT("hello from py_ninja")
            }
        }

    #pyninja is extracted, embedded, and compiled exactly like #ninja,
    except the generated runtime wrapper shells out to

        python3 py_ninja.py <temp file>

    i.e. it targets the real py_ninja.py interpreter (the PY_NINJA
    file in this project) instead of the placeholder
    "NINJA--COMPILER.py". Both #ninja and #pyninja are supported side
    by side - #ninja is left completely intact.

  * NEW: #extern "cuda" and #extern "xla" languages.

        #extern "cuda"
        {
            __global__ void addKernel(int* c, const int* a, const int* b) {
                int i = threadIdx.x;
                c[i] = a[i] + b[i];
            }
        }

        #extern "xla"
        {
            // XLA custom-call / HLO-adjacent C++ source
            ...
        }

    These behave like the existing "c"/"c++" extern languages -
    extracted the same way, with the same brace-matching / string
    handling - except:

      - "cuda" blocks are written to a .cu file and compiled with
        nvcc (NVIDIA CUDA Compiler) instead of clang/clang++, using
        the CUDA toolkit rooted at CUDA_TOOLKIT_DEFAULT (overridable
        via the KYBER_CUDA_ROOT env var). nvcc is resolved from
        <CUDA_TOOLKIT_ROOT>/bin/nvcc (falling back to "nvcc" on PATH
        if that binary isn't found), and the toolkit's include/lib
        directories are passed to nvcc so CUDA headers/libraries
        resolve without extra setup.

      - "xla" blocks are written to a .cpp file and compiled with
        clang++ (XLA's runtime/custom-call ABI is plain C++), using
        the XLA toolkit rooted at XLA_TOOLKIT_DEFAULT (overridable
        via the KYBER_XLA_ROOT env var). The toolkit's include/lib
        directories are passed to clang++ the same way.

    Object files land in the usual .obj directory as
    kyber_extern_N.obj, exactly like "c"/"c++" extern blocks - CUDA
    and XLA extern blocks are just two more entries in the same
    extern-block list, compiled with a different toolchain per
    language. kyberlink still does the final link; CUDA blocks need
    -lcudart (and the CUDA toolkit's lib dir on the linker path) at
    link time, and XLA blocks need whatever XLA runtime libraries the
    toolkit provides on the linker path.

Everything else (paths, block extraction algorithm, object file
layout, linking notes, usage text) is preserved from the original.
"""

import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from llvmlite import ir as llvm_ir
    from llvmlite import binding as llvm_binding
    LLVMLITE_AVAILABLE = True
except ImportError:
    LLVMLITE_AVAILABLE = False

# Attempt to load a project.env YAML early so settings (paths, folders, toolchain) are available as env vars.
# Loading is defensive: missing PyYAML or errors won't abort kyber.
try:
    import importlib
    _pe = None
    try:
        _pe = importlib.import_module('compiler.project_env')
    except Exception:
        try:
            _pe = importlib.import_module('project_env')
        except Exception:
            _pe = None
    if _pe and hasattr(_pe, 'load_project_env'):
        try:
            _pe.load_project_env()
        except Exception:
            pass
except Exception:
    pass

# ============================================================
# Kyber paths
# ============================================================
KYBER_ROOT = Path(os.environ.get(
    "KYBER_ROOT",
    Path(__file__).resolve().parent
))

BUILD_DIR = Path(os.environ.get(
    "KYBER_BUILD_DIR",
    "build"
))

OBJECT_DIRECTORY = BUILD_DIR / ".obj"

KYBER_OBJECT = OBJECT_DIRECTORY / "kyber.obj"


NINJA_INTERPRETER_DEFAULT = Path(os.environ.get(
    "KYBER_NINJA_ROOT",
    KYBER_ROOT / "experiments"
))

PYNINJA_INTERPRETER_DEFAULT = Path(os.environ.get(
    "KYBER_PYNINJA_ROOT",
    KYBER_ROOT / "experiments"
))

# NEW: CUDA toolkit root, used by #extern "cuda" blocks.
# Override with the KYBER_CUDA_ROOT env var.
CUDA_TOOLKIT_DEFAULT = Path(os.environ.get(
    "KYBER_CUDA_ROOT",
    r"C:\Users\Admin\kyber (7)\kyber\CUDA"
))

# NEW: XLA toolkit root, used by #extern "xla" blocks.
# Override with the KYBER_XLA_ROOT env var.
XLA_TOOLKIT_DEFAULT = Path(os.environ.get(
    "KYBER_XLA_ROOT",
    r"C:\Users\Admin\kyber (7)\kyber\XLA"
))

# NEW: NGPU source output directory.  #extern "ngpu" blocks are written
# here as .ngpu files before being passed to ngpu_compiler.py.
# Override with the KYBER_NGPU_DIR env var.
NGPU_SOURCE_DIR = Path(os.environ.get(
    "KYBER_NGPU_DIR",
    r"C:\Users\Admin\kyber (7)\kyber\ngpu"
))

# Path to the NGPU compiler script, resolved relative to this file.
NGPU_COMPILER = Path(os.environ.get(
    "KYBER_NGPU_COMPILER",
    Path(__file__).resolve().parent / "ngpu_compiler.py"
))

# NEW: XASM source output directory. #extern "xasm" blocks are written
# here as .xasm files before being passed to xasm_compiler.py.
# Override with the KYBER_XASM_DIR env var.
XASM_SOURCE_DIR = Path(os.environ.get(
    "KYBER_XASM_DIR",
    r"C:\Users\Admin\kyber (7)\kyber\xasm"
))

# NEW: path to the XASM compiler script.
XASM_COMPILER = Path(os.environ.get(
    "KYBER_XASM_COMPILER",
    Path(__file__).resolve().parent / "xasm_compiler.py"
))

# Python interpreter used to invoke sub-compilers (ngpu, xasm, npu).
# Override with KYBER_PYTHON env var.
KYBER_PYTHON = os.environ.get(
    "KYBER_PYTHON",
    r"C:/Users/Admin/PyCharmMiscProject/.venv/Scripts/python.exe"
)

# NEW: NPU source output directory. #extern "npu" blocks are written
# here as .npu files before being passed to npu_compiler.py.
# Override with the KYBER_NPU_DIR env var.
NPU_SOURCE_DIR = Path(os.environ.get(
    "KYBER_NPU_DIR",
    r"C:\Users\Admin\kyber (7)\kyber\npu"
))

# NEW: path to the NPU compiler script.
NPU_COMPILER = Path(os.environ.get(
    "KYBER_NPU_COMPILER",
    Path(__file__).resolve().parent / "npu_compiler.py"
))

# ============================================================
# Token system
# ============================================================

class TokenKind(Enum):
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    KEYWORD = auto()
    OPERATOR = auto()

    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    OPEN_BRACKET = auto()
    CLOSE_BRACKET = auto()
    OPEN_BRACE = auto()
    CLOSE_BRACE = auto()

    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    DOT = auto()
    ARROW = auto()

    NEWLINE = auto()
    END = auto()


@dataclass
class Token:
    kind: TokenKind
    text: str
    line: int
    column: int


# ============================================================
# Lexer
# ============================================================

class KyberLexer:

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    # ── low level helpers ────────────────────────────────

    def _eof(self) -> bool:
        return self.pos >= len(self.source)

    def _peek(self, n: int = 0) -> str:
        idx = self.pos + n
        if idx >= len(self.source):
            return "\0"
        return self.source[idx]

    def _advance(self) -> str:
        if self._eof():
            return "\0"
        c = self.source[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return c

    def _starts_with(self, marker: str) -> bool:
        return self.source.startswith(marker, self.pos)

    def _skip_spaces(self):
        """
        Skip whitespace and Kyber comments.

        #extern / #n8n / #ninja / #pyninja are handled (extracted)
        before lexing ever runs. If one of those markers somehow
        reaches the lexer (extraction was skipped or failed), stop
        rather than silently eating it as a comment.
        """
        while not self._eof():
            c = self._peek()

            if c in (" ", "\t", "\r"):
                self._advance()
                continue

            if c == "#":
                if (self._starts_with("#extern") or
                        self._starts_with("#n8n") or
                        self._starts_with("#ninja") or
                        self._starts_with("#pyninja")):
                    break

                # plain Kyber comment
                while not self._eof() and self._peek() != "\n":
                    self._advance()
                continue

            break

    # ── keywords ─────────────────────────────────────────

    KEYWORDS = {
        "fn",
        "let",
        "class",
        "new",
        "this",
        "self",
        "tensor",
        "matrix",
        "transpose",
        "determinant",
        "inverse",
        "dot",
        "multiply",
        "add",
        "sub",
        "div",
        "layer",
        "target",

        "cpu",
        "gpu",

        # I/O
        "print",
        "input",

        # File handling
        "file",
        "open",
        "read",
        "write",
        "append",
        "close",

        # Error handling
        "try",
        "catch",
        "throw",
        "error",

        # C/C++ interop
        "extern",
        "cpp",
        "c",
        "include",
        "call",

        # NEW: reserved the same way "extern" already was, so that
        # bare `ninja` / `pyninja` identifiers in Kyber source are
        # treated as keywords, matching the #ninja / #pyninja block
        # markers extracted before lexing.
        "ninja",
        "pyninja",

        # NEW: reserved for #extern "ngpu" blocks.
        "ngpu",

        # NEW: reserved for #extern "npu" blocks (KNPU/DirectML).
        "npu",

        "return",
        "if",
        "else",
        "for",
        "in",

        "true",
        "false",

        "Dense",
        "ReLU",
        "Conv",
        "Sigmoid",
        "Softmax",
    }

    def _identifier(self) -> Token:
        start_line = self.line
        start_column = self.column
        text = []

        while not self._eof():
            c = self._peek()
            if c.isalnum() or c == "_":
                text.append(self._advance())
            else:
                break

        text = "".join(text)
        kind = TokenKind.KEYWORD if text in self.KEYWORDS else TokenKind.IDENTIFIER
        return Token(kind, text, start_line, start_column)

    def _number(self) -> Token:
        start_line = self.line
        start_column = self.column
        text = []
        decimal = False

        while not self._eof():
            c = self._peek()
            if c.isdigit():
                text.append(self._advance())
                continue
            if c == "." and not decimal:
                decimal = True
                text.append(self._advance())
                continue
            break

        return Token(TokenKind.NUMBER, "".join(text), start_line, start_column)

    def _string_literal(self) -> Token:
        start_line = self.line
        start_column = self.column

        self._advance()  # opening quote

        text = []
        while not self._eof() and self._peek() != '"':
            c = self._advance()
            if c == "\\" and not self._eof():
                nxt = self._advance()
                if nxt == "n":
                    text.append("\n")
                elif nxt == "t":
                    text.append("\t")
                else:
                    text.append(nxt)
            else:
                text.append(c)

        if not self._eof():
            self._advance()  # closing quote

        return Token(TokenKind.STRING, "".join(text), start_line, start_column)

    # ── main entry point ─────────────────────────────────

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while not self._eof():
            self._skip_spaces()

            if self._eof():
                break

            c = self._peek()
            token_line = self.line
            token_column = self.column

            if c == "\n":
                self._advance()
                tokens.append(Token(TokenKind.NEWLINE, "\\n", token_line, token_column))
                continue

            if c.isalpha() or c == "_":
                tokens.append(self._identifier())
                continue

            if c.isdigit():
                tokens.append(self._number())
                continue

            if c == '"':
                tokens.append(self._string_literal())
                continue

            if c == "-" and self._peek(1) == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenKind.ARROW, "->", token_line, token_column))
                continue

            simple = {
                "(": TokenKind.OPEN_PAREN,
                ")": TokenKind.CLOSE_PAREN,
                "[": TokenKind.OPEN_BRACKET,
                "]": TokenKind.CLOSE_BRACKET,
                "{": TokenKind.OPEN_BRACE,
                "}": TokenKind.CLOSE_BRACE,
                ",": TokenKind.COMMA,
                ":": TokenKind.COLON,
                ";": TokenKind.SEMI,
                ".": TokenKind.DOT,
            }

            kind = simple.get(c, TokenKind.OPERATOR)
            text = self._advance()

            if kind == TokenKind.OPERATOR:
                if text in ("=", "!", "<", ">") and self._peek() == "=":
                    text += self._advance()

            tokens.append(Token(kind, text, token_line, token_column))

        tokens.append(Token(TokenKind.END, "", self.line, self.column))
        return tokens


# ============================================================
# Type system
# ============================================================

class KyberType(Enum):
    """
    Every value in Kyber has one of these types.  The parser infers
    the type from the initializer expression and stores it on the
    AST node so that codegen (and future passes like FCORE) can make
    informed decisions without re-analysing the tree.
    """
    UNKNOWN   = auto()   # not yet resolved
    INT       = auto()   # integer literal  e.g.  let x = 5
    FLOAT     = auto()   # float literal    e.g.  let x = 3.14
    BOOL      = auto()   # true / false
    STRING    = auto()   # string literal   e.g.  let x = "hello"
    TENSOR    = auto()   # tensor [...]
    MATRIX    = auto()   # matrix [[...]]
    LAYER     = auto()   # Dense(...) / ReLU(...) etc.
    OBJECT    = auto()   # new ClassName(...)
    VOID      = auto()   # functions that return nothing
    FUNCTION  = auto()   # fn declarations
    # hardware-specific
    NGPU      = auto()   # value originating from an ngpu block
    NPU       = auto()   # value originating from an npu block
    XASM      = auto()   # value originating from an xasm block


# Human-readable names used in error messages.
_TYPE_NAMES: dict = {
    KyberType.UNKNOWN:  "unknown",
    KyberType.INT:      "int",
    KyberType.FLOAT:    "float",
    KyberType.BOOL:     "bool",
    KyberType.STRING:   "string",
    KyberType.TENSOR:   "tensor",
    KyberType.MATRIX:   "matrix",
    KyberType.LAYER:    "layer",
    KyberType.OBJECT:   "object",
    KyberType.VOID:     "void",
    KyberType.FUNCTION: "function",
    KyberType.NGPU:     "ngpu",
    KyberType.NPU:      "npu",
    KyberType.XASM:     "xasm",
}

# Layer type names recognised by the parser.
_LAYER_TYPES = {"Dense", "ReLU", "Conv", "Sigmoid", "Softmax"}


def type_name(t: KyberType) -> str:
    return _TYPE_NAMES.get(t, "unknown")


# ============================================================
# AST
# ============================================================

class ASTNode:
    # Every node carries its resolved Kyber type.
    kyber_type: KyberType = KyberType.UNKNOWN

    def codegen(self, module):
        raise NotImplementedError


class TensorNode(ASTNode):
    kyber_type = KyberType.TENSOR

    def __init__(self, name: str, values: List[float]):
        self.name = name
        self.values = values

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE or not self.values:
            return

        elem_ty = llvm_ir.DoubleType()
        arr_ty = llvm_ir.ArrayType(elem_ty, len(self.values))
        constants = [llvm_ir.Constant(elem_ty, v) for v in self.values]
        initializer = llvm_ir.Constant(arr_ty, constants)

        gv = llvm_ir.GlobalVariable(module, arr_ty, name=self.name)
        gv.global_constant = True
        gv.linkage = "internal"
        gv.initializer = initializer


class MatrixNode(ASTNode):
    kyber_type = KyberType.MATRIX

    def __init__(self, name: str, values: List[List[float]]):
        self.name = name
        self.values = values

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE or not self.values:
            return

        rows = len(self.values)
        cols = len(self.values[0])
        if cols == 0:
            return

        elem_ty = llvm_ir.DoubleType()
        row_ty = llvm_ir.ArrayType(elem_ty, cols)
        matrix_ty = llvm_ir.ArrayType(row_ty, rows)

        row_constants = []
        for row in self.values:
            if len(row) != cols:
                return
            elements = [llvm_ir.Constant(elem_ty, v) for v in row]
            row_constants.append(llvm_ir.Constant(row_ty, elements))

        initializer = llvm_ir.Constant(matrix_ty, row_constants)

        gv = llvm_ir.GlobalVariable(module, matrix_ty, name=self.name)
        gv.global_constant = True
        gv.linkage = "internal"
        gv.initializer = initializer


class LayerNode(ASTNode):
    kyber_type = KyberType.LAYER

    def __init__(self, name: str, layer_type: str, arguments: List[float]):
        self.name = name
        self.layer_type = layer_type
        self.arguments = arguments

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE:
            return

        elem_ty = llvm_ir.DoubleType()
        fields = [elem_ty for _ in self.arguments]
        if not fields:
            fields = [llvm_ir.IntType(32)]

        struct_ty = llvm_ir.LiteralStructType(fields)
        initializer = struct_ty(None)  # zero-initializer

        gv = llvm_ir.GlobalVariable(module, struct_ty, name=self.name)
        gv.linkage = "internal"
        gv.initializer = initializer


class MathNode(ASTNode):
    kyber_type = KyberType.FLOAT   # math ops produce numeric results

    def __init__(self, name: str, operation: str, operands: List[str]):
        self.name = name
        self.operation = operation
        self.operands = operands

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE:
            return

        main_fn = None
        for fn in module.functions:
            if fn.name == "main":
                main_fn = fn
                break

        if main_fn is None or not main_fn.blocks:
            return

        block = main_fn.blocks[0]
        builder = llvm_ir.IRBuilder(block)
        double_ty = llvm_ir.DoubleType()

        if self.operation == "add" and len(self.operands) == 2:
            a = llvm_ir.Constant(double_ty, 0.0)
            b = llvm_ir.Constant(double_ty, 0.0)
            builder.fadd(a, b, name=self.name)


class ClassFieldNode(ASTNode):
    kyber_type = KyberType.UNKNOWN

    def __init__(self, name: str, field_type: Optional[str] = None, default: Optional[str] = None):
        self.name = name
        self.field_type = field_type
        self.default = default
        # Resolve field type from annotation if provided
        if field_type is not None:
            self.kyber_type = _annotation_to_kyber_type(field_type)

    def codegen(self, module):
        return None


class ClassNode(ASTNode):
    kyber_type = KyberType.OBJECT

    def __init__(self, name: str, fields: Optional[List[ClassFieldNode]] = None, methods: Optional[List[str]] = None):
        self.name = name
        self.fields = fields or []
        self.methods = methods or []

    def codegen(self, module):
        return None


class VariableDeclNode(ASTNode):
    """
    A variable declaration.  The `kyber_type` is set to UNKNOWN by
    default and updated by the parser once it sees the initializer.
    """
    def __init__(self, name: str, initializer: Optional[ASTNode] = None,
                 kyber_type: KyberType = KyberType.UNKNOWN):
        self.name = name
        self.initializer = initializer
        self.kyber_type = kyber_type
        # Inherit type from initializer when not explicitly supplied.
        if self.kyber_type == KyberType.UNKNOWN and initializer is not None:
            self.kyber_type = initializer.kyber_type

    def codegen(self, module):
        return None


class NewNode(ASTNode):
    kyber_type = KyberType.OBJECT

    def __init__(self, class_name: str, arguments: Optional[List[ASTNode]] = None):
        self.class_name = class_name
        self.arguments = arguments or []

    def codegen(self, module):
        return None


class MemberAccessNode(ASTNode):
    kyber_type = KyberType.UNKNOWN

    def __init__(self, target: str, member: str):
        self.target = target
        self.member = member

    def codegen(self, module):
        return None


class MethodCallNode(ASTNode):
    kyber_type = KyberType.UNKNOWN

    def __init__(self, target: str, method: str, arguments: Optional[List[ASTNode]] = None):
        self.target = target
        self.method = method
        self.arguments = arguments or []

    def codegen(self, module):
        return None


class FunctionNode(ASTNode):
    kyber_type = KyberType.FUNCTION

    def __init__(self, name: str, params: Optional[List[str]] = None,
                 body: Optional[List[ASTNode]] = None,
                 return_type: KyberType = KyberType.VOID):
        self.name = name
        self.params = params or []
        self.body = body or []
        self.return_type = return_type

    def codegen(self, module):
        return None


class MatrixValueNode(ASTNode):
    kyber_type = KyberType.MATRIX

    def __init__(self, name: str, values: Optional[List[List[float]]] = None):
        self.name = name
        self.values = values or []

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE:
            return
        if not self.values:
            return
        rows = len(self.values)
        cols = len(self.values[0]) if rows else 0
        if cols == 0:
            return

        elem_ty = llvm_ir.DoubleType()
        row_ty = llvm_ir.ArrayType(elem_ty, cols)
        matrix_ty = llvm_ir.ArrayType(row_ty, rows)
        row_constants = []
        for row in self.values:
            if len(row) != cols:
                return
            row_constants.append(llvm_ir.Constant(row_ty, [llvm_ir.Constant(elem_ty, v) for v in row]))

        gv = llvm_ir.GlobalVariable(module, matrix_ty, name=self.name)
        gv.global_constant = True
        gv.linkage = "internal"
        gv.initializer = llvm_ir.Constant(matrix_ty, row_constants)


class MatrixOperationNode(ASTNode):
    kyber_type = KyberType.MATRIX
    MATRIX_OPS = {"add", "sub", "mul", "div", "dot", "transpose", "determinant", "inverse"}

    def __init__(self, name: str, operation: str, operands: Optional[List[str]] = None):
        self.name = name
        self.operation = operation.lower()
        self.operands = operands or []

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE:
            return
        return None


class ProgramNode(ASTNode):
    kyber_type = KyberType.VOID

    def __init__(self):
        self.items: List[ASTNode] = []

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE:
            return

        fn_ty = llvm_ir.FunctionType(llvm_ir.IntType(32), [])
        main_fn = llvm_ir.Function(module, fn_ty, name="main")
        block = main_fn.append_basic_block(name="entry")
        builder = llvm_ir.IRBuilder(block)

        for item in self.items:
            item.codegen(module)

        if not block.is_terminated:
            builder.position_at_end(block)
            builder.ret(llvm_ir.Constant(llvm_ir.IntType(32), 0))


# ── Helper: map a type-annotation string to KyberType ────────────────

def _annotation_to_kyber_type(annotation: str) -> KyberType:
    """Convert a textual type annotation (from `let x: int = ...`) to a KyberType."""
    _map = {
        "int":     KyberType.INT,
        "float":   KyberType.FLOAT,
        "bool":    KyberType.BOOL,
        "string":  KyberType.STRING,
        "str":     KyberType.STRING,
        "tensor":  KyberType.TENSOR,
        "matrix":  KyberType.MATRIX,
        "layer":   KyberType.LAYER,
        "void":    KyberType.VOID,
        "ngpu":    KyberType.NGPU,
        "npu":     KyberType.NPU,
        "xasm":    KyberType.XASM,
    }
    return _map.get(annotation.lower(), KyberType.UNKNOWN)


# ── Additional typed AST nodes ────────────────────────────────────────

class StringNode(ASTNode):
    """A string literal:  let x = "hello" """
    kyber_type = KyberType.STRING

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def codegen(self, module):
        # String constants are emitted as global i8 arrays (null-terminated).
        if not LLVMLITE_AVAILABLE or not self.name:
            return
        data = (self.value + "\0").encode("utf-8")
        arr_ty = llvm_ir.ArrayType(llvm_ir.IntType(8), len(data))
        initializer = llvm_ir.Constant(arr_ty, bytearray(data))
        gv = llvm_ir.GlobalVariable(module, arr_ty, name=self.name)
        gv.global_constant = True
        gv.linkage = "internal"
        gv.initializer = initializer


class BoolNode(ASTNode):
    """A boolean literal:  let flag = true """
    kyber_type = KyberType.BOOL

    def __init__(self, name: str, value: bool):
        self.name = name
        self.value = value

    def codegen(self, module):
        if not LLVMLITE_AVAILABLE or not self.name:
            return
        bool_ty = llvm_ir.IntType(1)
        const = llvm_ir.Constant(bool_ty, int(self.value))
        gv = llvm_ir.GlobalVariable(module, bool_ty, name=self.name)
        gv.global_constant = True
        gv.linkage = "internal"
        gv.initializer = const


class ReturnNode(ASTNode):
    """A return statement:  return <expr> """
    kyber_type = KyberType.VOID

    def __init__(self, value: Optional[ASTNode] = None):
        self.value = value
        if value is not None:
            self.kyber_type = value.kyber_type

    def codegen(self, module):
        return None


class IfNode(ASTNode):
    """An if/else statement."""
    kyber_type = KyberType.VOID

    def __init__(self, condition: ASTNode,
                 then_body: List[ASTNode],
                 else_body: Optional[List[ASTNode]] = None):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body or []

    def codegen(self, module):
        return None


class ForNode(ASTNode):
    """A for-in loop:  for item in collection { ... } """
    kyber_type = KyberType.VOID

    def __init__(self, variable: str, iterable: str, body: List[ASTNode]):
        self.variable = variable
        self.iterable = iterable
        self.body = body

    def codegen(self, module):
        return None


# ============================================================
# Parser
# ============================================================

class KyberParser:

    def __init__(self, tokens: List[Token], source_lines: Optional[List[str]] = None):
        self.tokens = tokens
        self.current = 0
        # Original source lines for rich error display (optional).
        self._source_lines: List[str] = source_lines or []

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _is(self, kind: TokenKind) -> bool:
        return self._peek().kind == kind

    def _is_text(self, text: str) -> bool:
        return self._peek().text == text

    def _advance(self) -> Token:
        tok = self.tokens[self.current]
        self.current += 1
        return tok

    def _match(self, kind: TokenKind) -> bool:
        if not self._is(kind):
            return False
        self._advance()
        return True

    def _match_text(self, text: str) -> bool:
        if not self._is_text(text):
            return False
        self._advance()
        return True

    def _skip_newlines(self):
        while self._is(TokenKind.NEWLINE):
            self._advance()

    # ── Rich error reporting ──────────────────────────────────────────

    def _error(self, tok: Token, message: str) -> None:
        """
        Print a clang-style error pointing at the offending token:

            error: line 12: expected ')'
             12 |     let x = add(a b)
                |                  ^
        """
        line_no = tok.line
        col_no  = tok.column

        header = f"error: line {line_no}: {message}"
        print(header, file=sys.stderr)

        if self._source_lines and 1 <= line_no <= len(self._source_lines):
            src_line = self._source_lines[line_no - 1]
            gutter   = f" {line_no} | "
            print(gutter + src_line, file=sys.stderr)
            # Place the caret under the token start column (1-indexed).
            pad = len(gutter) + max(0, col_no - 1)
            # Underline the full token text when possible.
            width = max(1, len(tok.text)) if tok.kind != TokenKind.END else 1
            print(" " * pad + "^" + "~" * (width - 1), file=sys.stderr)

    def _expect(self, kind: TokenKind, message: str) -> bool:
        if self._is(kind):
            self._advance()
            return True
        self._error(self._peek(), message)
        return False

    def _parse_number_list(self) -> List[float]:
        values: List[float] = []

        if not self._expect(TokenKind.OPEN_BRACKET, "expected '['"):
            return values

        while not self._is(TokenKind.CLOSE_BRACKET) and not self._is(TokenKind.END):
            if not self._is(TokenKind.NUMBER):
                self._error(self._peek(), "expected a number inside '[...]'")
                return []
            values.append(float(self._advance().text))
            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.CLOSE_BRACKET, "expected ']'")
        return values

    def _parse_matrix(self) -> List[List[float]]:
        matrix: List[List[float]] = []

        if not self._expect(TokenKind.OPEN_BRACKET, "expected '['"):
            return matrix

        while not self._is(TokenKind.CLOSE_BRACKET) and not self._is(TokenKind.END):
            matrix.append(self._parse_number_list())
            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.CLOSE_BRACKET, "expected ']'")
        return matrix

    def _parse_call_arguments(self) -> List[float]:
        args: List[float] = []

        if not self._expect(TokenKind.OPEN_PAREN, "expected '('"):
            return args

        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
            if not self._is(TokenKind.NUMBER):
                self._error(self._peek(), "expected a number in argument list")
                return []
            args.append(float(self._advance().text))
            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.CLOSE_PAREN, "expected ')'")
        return args

    def _parse_tensor(self) -> Optional[ASTNode]:
        self._advance()  # 'tensor'

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected tensor name after 'tensor'")
            return None

        name = self._advance().text

        if not self._match_text("="):
            self._error(self._peek(), "expected '=' after tensor name")
            return None

        values = self._parse_number_list()
        return TensorNode(name, values)

    def _parse_matrix_node(self) -> Optional[ASTNode]:
        self._advance()  # 'matrix'

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected matrix name after 'matrix'")
            return None

        name = self._advance().text

        if not self._match_text("="):
            self._error(self._peek(), "expected '=' after matrix name")
            return None

        values = self._parse_matrix()
        return MatrixValueNode(name, values)

    def _parse_matrix_operation(self, target_name: str) -> Optional[ASTNode]:
        if not self._match_text("="):
            return None

        if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
            op_name = self._advance().text.lower()
            if op_name in MatrixOperationNode.MATRIX_OPS:
                if self._match(TokenKind.OPEN_PAREN):
                    operands: List[str] = []
                    while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                        if self._is(TokenKind.IDENTIFIER):
                            operands.append(self._advance().text)
                        else:
                            self._advance()
                        if not self._match(TokenKind.COMMA):
                            break
                    self._match(TokenKind.CLOSE_PAREN)
                    return MatrixOperationNode(target_name, op_name, operands)
                return MatrixOperationNode(target_name, op_name, [])

        if self._is(TokenKind.OPEN_BRACKET):
            values = self._parse_matrix()
            return MatrixValueNode(target_name, values)

        return MatrixValueNode(target_name)

    def _parse_layer(self) -> Optional[ASTNode]:
        self._advance()  # 'layer'

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected layer name after 'layer'")
            return None

        name = self._advance().text

        if not self._match_text("="):
            self._error(self._peek(), "expected '=' after layer name")
            return None

        if not self._is(TokenKind.IDENTIFIER) and not self._is(TokenKind.KEYWORD):
            self._error(self._peek(), "expected layer type (e.g. Dense, ReLU)")
            return None

        layer_type = self._advance().text
        args = self._parse_call_arguments()
        return LayerNode(name, layer_type, args)

    def _parse_math(self) -> Optional[ASTNode]:
        if not self._is(TokenKind.IDENTIFIER):
            return None

        name = self._advance().text

        if not self._match_text("="):
            return None

        if not self._is(TokenKind.IDENTIFIER) and not self._is(TokenKind.KEYWORD):
            return None

        operation = self._advance().text

        if not self._expect(TokenKind.OPEN_PAREN, "expected '('"):
            return None

        operands: List[str] = []
        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
            if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.NUMBER):
                operands.append(self._advance().text)
            else:
                return None
            if not self._match(TokenKind.COMMA):
                break

        if not self._expect(TokenKind.CLOSE_PAREN, "expected ')'"):
            return None

        return MathNode(name, operation, operands)

    def _parse_class(self) -> Optional[ASTNode]:
        self._advance()  # class

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected class name after 'class'")
            return None

        name = self._advance().text
        if not self._expect(TokenKind.OPEN_BRACE, "expected '{' after class name"):
            return None

        fields: List[ClassFieldNode] = []
        methods: List[str] = []

        while not self._is(TokenKind.CLOSE_BRACE) and not self._is(TokenKind.END):
            self._skip_newlines()
            if self._is(TokenKind.CLOSE_BRACE):
                break

            if self._is_text("let"):
                self._advance()
                if self._is(TokenKind.IDENTIFIER):
                    field_name = self._advance().text
                    fields.append(ClassFieldNode(field_name))
                else:
                    self._error(self._peek(), "expected field name after 'let' in class body")
                    return None
                while self._is(TokenKind.SEMI):
                    self._advance()
                continue

            if self._is_text("fn"):
                self._advance()
                if self._is(TokenKind.IDENTIFIER):
                    methods.append(self._advance().text)
                else:
                    self._error(self._peek(), "expected method name after 'fn' in class body")
                    return None

                while not self._is(TokenKind.OPEN_PAREN) and not self._is(TokenKind.END):
                    self._advance()
                if self._match(TokenKind.OPEN_PAREN):
                    while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                        self._advance()
                    self._match(TokenKind.CLOSE_PAREN)
                if self._match(TokenKind.OPEN_BRACE):
                    depth = 1
                    while not self._is(TokenKind.END) and depth > 0:
                        if self._match(TokenKind.OPEN_BRACE):
                            depth += 1
                        elif self._match(TokenKind.CLOSE_BRACE):
                            depth -= 1
                        else:
                            self._advance()
                continue

            if self._is(TokenKind.IDENTIFIER):
                field_name = self._advance().text
                fields.append(ClassFieldNode(field_name))
                continue

            self._advance()

        self._expect(TokenKind.CLOSE_BRACE, "expected '}' to close class body")
        return ClassNode(name, fields, methods)

    def _parse_let_statement(self) -> Optional[ASTNode]:
        self._advance()  # let
        if not self._is(TokenKind.IDENTIFIER):
            return None

        name = self._advance().text

        # Optional type annotation:  let x: int = ...
        explicit_type = KyberType.UNKNOWN
        if self._match(TokenKind.COLON):
            if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
                explicit_type = _annotation_to_kyber_type(self._advance().text)

        init = None
        if self._match_text("="):
            if self._is_text("new"):
                self._advance()
                class_name = self._advance().text if self._is(TokenKind.IDENTIFIER) else ""
                if self._match(TokenKind.OPEN_PAREN):
                    while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                        self._advance()
                    self._match(TokenKind.CLOSE_PAREN)
                init = NewNode(class_name)

            elif self._is_text("true") or self._is_text("false"):
                value = self._advance().text == "true"
                init = BoolNode(name, value)

            elif self._is(TokenKind.STRING):
                value = self._advance().text
                init = StringNode(name, value)

            elif self._is(TokenKind.OPEN_BRACKET):
                values = self._parse_matrix()
                # Single-row bracket → tensor, multi-row → matrix
                if values and isinstance(values[0], list):
                    init = MatrixValueNode(name, values)
                else:
                    # _parse_matrix always returns List[List[float]]; flatten for tensor
                    flat = [v for row in values for v in row]
                    init = TensorNode(name, flat)

            elif self._is(TokenKind.NUMBER):
                raw = self._advance().text
                num = float(raw)
                if "." in raw:
                    t = TensorNode(name, [num])
                    t.kyber_type = KyberType.FLOAT
                    init = t
                else:
                    t = TensorNode(name, [num])
                    t.kyber_type = KyberType.INT
                    init = t

            elif self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
                op_name = self._advance().text
                op_lower = op_name.lower()
                if op_lower in MatrixOperationNode.MATRIX_OPS:
                    operands: List[str] = []
                    if self._match(TokenKind.OPEN_PAREN):
                        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                            if self._is(TokenKind.IDENTIFIER):
                                operands.append(self._advance().text)
                            else:
                                self._advance()
                            if not self._match(TokenKind.COMMA):
                                break
                        self._match(TokenKind.CLOSE_PAREN)
                    init = MatrixOperationNode(name, op_lower, operands)
                elif op_name in _LAYER_TYPES:
                    # e.g.  let fc1 = Dense(128)
                    args: List[float] = []
                    if self._match(TokenKind.OPEN_PAREN):
                        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                            if self._is(TokenKind.NUMBER):
                                args.append(float(self._advance().text))
                            else:
                                self._advance()
                            if not self._match(TokenKind.COMMA):
                                break
                        self._match(TokenKind.CLOSE_PAREN)
                    init = LayerNode(name, op_name, args)
                else:
                    init = VariableDeclNode(name)

        if self._is(TokenKind.SEMI):
            self._advance()

        if init is None:
            return VariableDeclNode(name, kyber_type=explicit_type)

        # Apply explicit annotation if present (overrides inference)
        if explicit_type != KyberType.UNKNOWN:
            init.kyber_type = explicit_type

        return init

    def _parse_member_access(self, target: str) -> Optional[ASTNode]:
        if not self._match(TokenKind.DOT):
            return None
        if not self._is(TokenKind.IDENTIFIER):
            return None
        member = self._advance().text
        if self._match(TokenKind.OPEN_PAREN):
            args: List[ASTNode] = []
            while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                if self._is(TokenKind.NUMBER):
                    args.append(TensorNode("", [float(self._advance().text)]))
                else:
                    self._advance()
            self._match(TokenKind.CLOSE_PAREN)
            return MethodCallNode(target, member, args)
        return MemberAccessNode(target, member)

    # ── Block parser (shared by fn / if / for) ────────────────────────

    def _parse_block(self) -> List[ASTNode]:
        """Parse a brace-enclosed list of statements."""
        body: List[ASTNode] = []
        if not self._expect(TokenKind.OPEN_BRACE, "expected '{'"):
            return body
        self._skip_newlines()
        while not self._is(TokenKind.CLOSE_BRACE) and not self._is(TokenKind.END):
            node = self._parse_statement()
            if node is not None:
                body.append(node)
            self._skip_newlines()
        self._expect(TokenKind.CLOSE_BRACE, "expected '}'")
        return body

    # ── fn declaration ────────────────────────────────────────────────

    def _parse_fn(self) -> Optional[ASTNode]:
        """
        fn name(param1, param2) -> ReturnType { ... }
        The -> ReturnType annotation is optional.
        """
        self._advance()  # 'fn'

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected function name after 'fn'")
            return None
        name = self._advance().text

        # Parameter list
        params: List[str] = []
        if not self._expect(TokenKind.OPEN_PAREN, "expected '(' after function name"):
            return None
        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
            if self._is(TokenKind.IDENTIFIER):
                params.append(self._advance().text)
            else:
                self._advance()
            if not self._match(TokenKind.COMMA):
                break
        self._expect(TokenKind.CLOSE_PAREN, "expected ')' after parameter list")

        # Optional return-type annotation:  -> int
        return_type = KyberType.VOID
        if self._match(TokenKind.ARROW):
            if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
                return_type = _annotation_to_kyber_type(self._advance().text)

        # Body
        body = self._parse_block()
        return FunctionNode(name, params, body, return_type)

    # ── return statement ──────────────────────────────────────────────

    def _parse_return(self) -> Optional[ASTNode]:
        self._advance()  # 'return'
        # Optional expression on the same line (before newline / '}')
        if self._is(TokenKind.NEWLINE) or self._is(TokenKind.SEMI) or self._is(TokenKind.CLOSE_BRACE):
            return ReturnNode(None)
        value = self._parse_statement()
        return ReturnNode(value)

    # ── if / else ─────────────────────────────────────────────────────

    def _parse_if(self) -> Optional[ASTNode]:
        self._advance()  # 'if'

        # Condition: everything up to the opening '{', parsed as a
        # single statement.  For now we grab one expression-level node.
        condition = self._parse_primary_expr()
        if condition is None:
            condition = VariableDeclNode("__cond__")

        then_body = self._parse_block()

        else_body: List[ASTNode] = []
        self._skip_newlines()
        if self._is_text("else"):
            self._advance()
            self._skip_newlines()
            if self._is_text("if"):
                nested = self._parse_if()
                if nested:
                    else_body = [nested]
            else:
                else_body = self._parse_block()

        return IfNode(condition, then_body, else_body)

    # ── for-in loop ───────────────────────────────────────────────────

    def _parse_for(self) -> Optional[ASTNode]:
        self._advance()  # 'for'

        if not self._is(TokenKind.IDENTIFIER):
            self._error(self._peek(), "expected loop variable name after 'for'")
            return None
        variable = self._advance().text

        if not self._match_text("in"):
            self._error(self._peek(), "expected 'in' after loop variable")
            return None

        iterable = ""
        if self._is(TokenKind.IDENTIFIER):
            iterable = self._advance().text

        body = self._parse_block()
        return ForNode(variable, iterable, body)

    # ── primary expression (for condition parsing) ────────────────────

    def _parse_primary_expr(self) -> Optional[ASTNode]:
        """
        Parse the simplest expression that can appear as an `if` condition:
        an identifier, a number, a string, true/false, or a comparison.
        """
        if self._is(TokenKind.NUMBER):
            raw = self._advance().text
            num = float(raw)
            n = TensorNode("", [num])
            n.kyber_type = KyberType.FLOAT if "." in raw else KyberType.INT
            return n

        if self._is(TokenKind.STRING):
            val = self._advance().text
            return StringNode("", val)

        if self._is_text("true") or self._is_text("false"):
            val = self._advance().text == "true"
            return BoolNode("", val)

        if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
            name = self._advance().text
            # comparison:  x == y  /  x != y  /  x < y  etc.
            if self._is(TokenKind.OPERATOR):
                op = self._peek().text
                if op in ("==", "!=", "<", ">", "<=", ">="):
                    self._advance()
                    rhs = self._parse_primary_expr()
                    return MathNode(name, op, [name, rhs.name if rhs else ""])
            return VariableDeclNode(name)

        return None

    def _parse_statement(self) -> Optional[ASTNode]:
        if self._is_text("fn"):
            return self._parse_fn()

        if self._is_text("return"):
            return self._parse_return()

        if self._is_text("if"):
            return self._parse_if()

        if self._is_text("for"):
            return self._parse_for()

        if self._is_text("tensor"):
            return self._parse_tensor()

        if self._is_text("matrix"):
            return self._parse_matrix_node()

        if self._is_text("layer"):
            return self._parse_layer()

        if self._is_text("add") or self._is_text("sub") or self._is_text("mul") or self._is_text("div") or self._is_text("dot") or self._is_text("transpose") or self._is_text("determinant") or self._is_text("inverse"):
            op_name = self._advance().text.lower()
            target = ""
            if self._is(TokenKind.IDENTIFIER):
                target = self._advance().text
            if self._match(TokenKind.OPEN_PAREN):
                args: List[str] = []
                while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                    if self._is(TokenKind.IDENTIFIER):
                        args.append(self._advance().text)
                    else:
                        self._advance()
                    if not self._match(TokenKind.COMMA):
                        break
                self._match(TokenKind.CLOSE_PAREN)
            return MatrixOperationNode(target or op_name, op_name, args if 'args' in locals() else [])

        if self._is_text("class"):
            return self._parse_class()

        if self._is_text("let"):
            return self._parse_let_statement()

        if self._is_text("new"):
            self._advance()
            if self._is(TokenKind.IDENTIFIER):
                class_name = self._advance().text
                if self._match(TokenKind.OPEN_PAREN):
                    while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                        self._advance()
                    self._match(TokenKind.CLOSE_PAREN)
                return NewNode(class_name)
            return None

        if self._is(TokenKind.IDENTIFIER):
            ident = self._advance().text
            if self._match(TokenKind.DOT):
                if self._is(TokenKind.IDENTIFIER):
                    member = self._advance().text
                    if self._match(TokenKind.OPEN_PAREN):
                        while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                            self._advance()
                        self._match(TokenKind.CLOSE_PAREN)
                    return MethodCallNode(ident, member)
                return MemberAccessNode(ident, "")
            return self._parse_math_from_identifier(ident)

        # target cpu / target gpu
        if self._is_text("target"):
            self._advance()
            if self._is(TokenKind.IDENTIFIER) or self._is(TokenKind.KEYWORD):
                self._advance()
            return None

        self._advance()
        return None

    def _parse_math_from_identifier(self, name: str) -> Optional[ASTNode]:
        if not self._match_text("="):
            return None
        if self._is_text("new"):
            self._advance()
            class_name = self._advance().text if self._is(TokenKind.IDENTIFIER) else ""
            if self._match(TokenKind.OPEN_PAREN):
                while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                    self._advance()
                self._match(TokenKind.CLOSE_PAREN)
            return VariableDeclNode(name, NewNode(class_name))
        if self._is(TokenKind.IDENTIFIER):
            operation = self._advance().text
            if self._match(TokenKind.OPEN_PAREN):
                while not self._is(TokenKind.CLOSE_PAREN) and not self._is(TokenKind.END):
                    self._advance()
                self._match(TokenKind.CLOSE_PAREN)
            return MathNode(name, operation, [])
        return VariableDeclNode(name)

    def parse(self) -> ProgramNode:
        program = ProgramNode()

        while not self._is(TokenKind.END):
            self._skip_newlines()

            if self._is(TokenKind.END):
                break

            node = self._parse_statement()
            if node is not None:
                program.items.append(node)

            self._skip_newlines()

        return program


# ============================================================
# Extern blocks
# ============================================================

@dataclass
class ExternBlock:
    language: str
    source: str


# ============================================================
# n8n blocks
#
# Example:
#
# #n8n "send_alert"
# {
#     url: "https://your-n8n-host/webhook/abc123"
#     method: "POST"
#     payload: "{\"status\":\"alert\"}"
# }
#
# Each #n8n block is compiled, like an #extern block, into its own
# object file exposing a C function of the same name:
#
#     int send_alert(const char* payload);
#
# Calling that function performs an HTTP request to the configured
# n8n webhook URL, so Kyber programs (or any code linked against
# kyber.obj) can trigger n8n workflows. Passing NULL for payload
# falls back to the payload literal given in the block, if any.
# ============================================================

@dataclass
class N8nBlock:
    name: str
    url: str = ""
    method: str = "POST"
    payload: str = ""


# ============================================================
# Ninja blocks
#
# Ninja (NINJA--COMPILER.py) is treated as a separate, standalone
# interpreter - it tokenizes, parses, and directly executes a .ninja
# program; it does not emit machine code or object files. That means
# it can't be lowered into LLVM IR and linked the way Kyber's own
# tensor/matrix/layer nodes are.
#
# What IS possible: treat a #ninja block like an #extern block, but
# instead of compiling the embedded source with clang, embed it as a
# string constant and generate a tiny C wrapper function that, at
# RUNTIME, writes that string to a temp .ninja file and shells out to
# the Ninja interpreter via "python3 NINJA--COMPILER.py <temp file>".
# That wrapper still compiles to a normal .obj (kyber_ninja_N.obj)
# that kyberlink can link in like any other extern symbol.
#
# Caveats worth knowing:
#   - This requires python3 and NINJA--COMPILER.py to be present on
#     the machine that RUNS the final executable, not just the
#     machine that compiles it.
#   - The Kyber program only gets the Ninja process's exit code back,
#     not structured return values, since Ninja has no notion of
#     returning a value to a caller.
#
# Example:
#
# #ninja "run_forward_pass"
# {
#     VOID MAIN() {
#         SCREEN.PRINT("hello from ninja")
#     }
# }
# ============================================================

@dataclass
class NinjaBlock:
    name: str
    source: str


# ============================================================
# NEW: PyNinja blocks
#
# Exactly the same idea as #ninja, but targets the real Python
# interpreter shipped with this project (py_ninja.py / PY_NINJA)
# instead of the placeholder "NINJA--COMPILER.py". The generated C
# wrapper shells out to:
#
#     python3 py_ninja.py <temp file>
#
# Example:
#
# #pyninja "run_forward_pass"
# {
#     VOID MAIN() {
#         SCREEN.PRINT("hello from py_ninja")
#     }
# }
# ============================================================

@dataclass
class PyNinjaBlock:
    name: str
    source: str


# ============================================================
# Generic helpers shared by block extraction
# ============================================================

# ============================================================
# IMPORT() resolution
#
# Kyber source files can declare NForce package dependencies:
#
#     IMPORT(kyber.math)
#     IMPORT(kyber.io)
#
# The compiler scans for these before lexing, resolves each
# package from ~/.nforce/packages/<name>/, finds its pre-built
# .obj files (or C source to compile), and adds them to the
# object directory so kyberlink can link them automatically.
#
# Package lookup order:
#   1. ~/.nforce/packages/<name>/          (installed via nforce)
#   2. <KYBER_ROOT>/packages/<name>/       (bundled packages)
#   3. KYBER_PACKAGES env var paths        (custom override)
# ============================================================

# Directories where NForce packages are searched
_NFORCE_PACKAGE_DIR = Path.home() / ".nforce" / "packages"
_KYBER_PACKAGE_DIR  = KYBER_ROOT / "packages"


def _find_package_dir(name: str) -> Optional[Path]:
    """Return the directory for an installed NForce package, or None."""
    candidates = [
        _NFORCE_PACKAGE_DIR / name,
        _KYBER_PACKAGE_DIR / name,
    ]
    # Extra paths from env var (colon/semicolon separated)
    extra = os.environ.get("KYBER_PACKAGES", "")
    for p in extra.replace(";", ":").split(":"):
        p = p.strip()
        if p:
            candidates.append(Path(p) / name)

    for c in candidates:
        if c.exists() and (c / "pkg_kyber.nforce").exists():
            return c
    return None


def extract_imports(source: str) -> Tuple[str, List[str]]:
    """
    Strip all  IMPORT(package.name)  lines from source and return
    (remaining_source, [package_name, ...]).
    """
    import_names: List[str] = []
    remaining_lines: List[str] = []

    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("IMPORT(") and stripped.endswith(")"):
            pkg = stripped[7:-1].strip()
            if pkg:
                import_names.append(pkg)
        else:
            remaining_lines.append(line)

    return "".join(remaining_lines), import_names


def resolve_imports(
    import_names: List[str],
    index_offset: int = 0,
) -> bool:
    """
    For each imported package:
      - Find the package directory.
      - If it contains pre-built .obj files, copy them into OBJECT_DIRECTORY.
      - If it contains .c source files (and no .obj), compile them with clang.
    Returns True if all imports resolved successfully, False otherwise.
    """
    if not import_names:
        return True

    print(f"\n[Kyber IMPORT] Resolving {len(import_names)} package(s)...")

    for i, name in enumerate(import_names):
        pkg_dir = _find_package_dir(name)
        if pkg_dir is None:
            print(
                f"[Kyber IMPORT] Package not found: {name}\n"
                f"  Run:  nforce register {name} <path>  then  nforce install {name}",
                file=sys.stderr,
            )
            return False

        print(f"[Kyber IMPORT] {name}  ->  {pkg_dir}")

        # Check for pre-built .obj files first
        obj_files = list(pkg_dir.glob("*.obj"))
        if obj_files:
            for obj in obj_files:
                dest = OBJECT_DIRECTORY / f"kyber_pkg_{name.replace('.', '_')}_{obj.stem}.obj"
                import shutil
                shutil.copy2(obj, dest)
                print(f"  Copied: {obj.name} -> {dest.name}")
            continue

        # No .obj — compile .c source files with clang
        c_files = list(pkg_dir.glob("*.c"))
        if not c_files:
            print(f"[Kyber IMPORT] Warning: package {name} has no .obj or .c files.", file=sys.stderr)
            continue

        for j, c_file in enumerate(c_files):
            obj_name = f"kyber_pkg_{name.replace('.', '_')}_{c_file.stem}.obj"
            obj_path = OBJECT_DIRECTORY / obj_name
            print(f"  Compiling: {c_file.name} -> {obj_name}")
            success = compile_source_to_object(
                source_text=c_file.read_text(encoding="utf-8"),
                extension=".c",
                compiler="clang",
                object_path=obj_path,
                temp_file_tag=f"pkg_{name.replace('.', '_')}",
                index=index_offset + i * 100 + j,
            )
            if not success:
                print(
                    f"[Kyber IMPORT] Failed to compile {c_file.name} from package {name}.",
                    file=sys.stderr,
                )
                return False

    print("[Kyber IMPORT] All packages resolved.")
    return True


def trim_whitespace(text: str) -> str:
    return text.strip(" \t\r\n")


def unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def escape_for_c(text: str) -> str:
    result = []
    for c in text:
        if c in ('"', "\\"):
            result.append("\\")
        result.append(c)
    return "".join(result)


def escape_for_c_string_literal(text: str) -> str:
    result = []
    for c in text:
        if c == '"':
            result.append('\\"')
        elif c == "\\":
            result.append("\\\\")
        elif c == "\n":
            result.append("\\n")
        elif c == "\r":
            result.append("\\r")
        else:
            result.append(c)
    return "".join(result)


def find_block_end(source: str, open_brace: int) -> int:
    """
    Finds the matching '}' for the '{' found at `open_brace`, respecting
    nested braces and quoted strings. Returns the index one past the
    closing '}', or -1 if the block is unterminated.
    """
    depth = 1
    i = open_brace + 1
    in_string = False
    string_char = "\0"

    n = len(source)
    while i < n and depth > 0:
        c = source[i]

        if c in ('"', "'") and (i == 0 or source[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = c
            elif string_char == c:
                in_string = False

        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1

        i += 1

    if depth != 0:
        return -1

    return i


def _extract_named_brace_blocks(source: str, marker: str):
    """
    Shared scanner for "#marker \"name\" { ... }" style blocks used by
    #extern, #n8n, #ninja, and #pyninja. Returns (remaining_source,
    list_of_(name, body)) or (None, None) on a malformed block.
    """
    remaining = []
    blocks = []

    pos = 0
    while pos < len(source):
        idx = source.find(marker, pos)
        if idx == -1:
            remaining.append(source[pos:])
            break

        remaining.append(source[pos:idx])

        quote1 = source.find('"', idx)
        if quote1 == -1:
            return None, None

        quote2 = source.find('"', quote1 + 1)
        if quote2 == -1:
            return None, None

        name = source[quote1 + 1:quote2]

        open_brace = source.find("{", quote2)
        if open_brace == -1:
            return None, None

        i = find_block_end(source, open_brace)
        if i == -1:
            print(f"Unclosed {marker} block", file=sys.stderr)
            return None, None

        body = source[open_brace + 1:i - 1]
        blocks.append((name, body))

        pos = i

    return "".join(remaining), blocks


# ============================================================
# Extract #extern blocks
#
# Example:
#
# #extern "c"
# {
#     int hello() { return 1; }
# }
#
# #extern "c++"
# {
#     int world() { return 2; }
# }
#
# #extern "cuda"     (NEW)
# {
#     __global__ void addKernel(int* c, const int* a, const int* b) {
#         int i = threadIdx.x;
#         c[i] = a[i] + b[i];
#     }
# }
#
# #extern "xla"       (NEW)
# {
#     // XLA custom-call C++ source
#     ...
# }
# ============================================================

# Languages accepted by #extern, mapped to (file_extension, compiler).
# "cuda" and "xla" are new: cuda compiles with nvcc, xla with clang++,
# each against its own toolkit root (see compile_extern_block).
# "ngpu" is handled entirely by the dedicated NGPU compiler pipeline
# (write .ngpu source → ngpu_compiler.py → nvcc PTX → .obj) and is
# therefore NOT in this table; extract_extern_blocks rejects "ngpu"
# so it never reaches compile_extern_block.
def _resolve_clang(variant: str = "clang++") -> str:
    """
    Resolve clang or clang++ — checks the bundled bin/ directory first,
    then falls back to PATH. Works on Windows where the exe must be explicit.
    """
    bin_dir = KYBER_ROOT / "bin"
    for name in (f"{variant}.exe", variant):
        candidate = bin_dir / name
        if candidate.exists():
            return str(candidate)
    return variant


EXTERN_LANGUAGES = {
    "c":    (".c",   _resolve_clang("clang")),
    "c++":  (".cpp", _resolve_clang("clang++")),
    "cuda": (".cu",  "nvcc"),
    "xla":  (".cpp", _resolve_clang("clang++")),
}

# Languages that are reserved #extern names but handled by their own
# dedicated pipeline (not compile_extern_block).
_EXTERN_DEDICATED = {"ngpu", "xasm", "npu"}


def extract_extern_blocks(source: str) -> Tuple[Optional[str], Optional[List[ExternBlock]]]:
    remaining, raw_blocks = _extract_named_brace_blocks(source, "#extern")
    if remaining is None:
        return None, None

    blocks: List[ExternBlock] = []
    for language, body in raw_blocks:
        if language in _EXTERN_DEDICATED:
            # These languages have their own dedicated pipeline.
            # Re-inject the block into `remaining` so extract_ngpu_blocks
            # can find it, by appending back a synthetic marker.
            # Instead of re-injecting, we rely on extract_ngpu_blocks
            # running on the ORIGINAL source before extern extraction,
            # so nothing needs to happen here - we simply skip it.
            # (See main(): ngpu blocks are extracted from the original
            # source before any other extraction runs.)
            continue
        if language not in EXTERN_LANGUAGES:
            print(f"Unsupported #extern language: {language}", file=sys.stderr)
            return None, None
        blocks.append(ExternBlock(language, body))

    return remaining, blocks


# ============================================================
# Extract #n8n blocks
# ============================================================

def _parse_n8n_body(name: str, body: str) -> Optional[N8nBlock]:
    block = N8nBlock(name=name)

    for line in body.splitlines():
        colon = line.find(":")
        if colon == -1:
            continue

        key = trim_whitespace(line[:colon])
        value = unquote(trim_whitespace(line[colon + 1:]))

        if key == "url":
            block.url = value
        elif key == "method":
            block.method = value
        elif key == "payload":
            block.payload = value

    if not block.url:
        print(f"#n8n block '{block.name}' is missing a 'url' field", file=sys.stderr)
        return None

    if not block.method:
        block.method = "POST"

    return block


def extract_n8n_blocks(source: str) -> Tuple[Optional[str], Optional[List[N8nBlock]]]:
    remaining, raw_blocks = _extract_named_brace_blocks(source, "#n8n")
    if remaining is None:
        return None, None

    blocks: List[N8nBlock] = []
    for name, body in raw_blocks:
        block = _parse_n8n_body(name, body)
        if block is None:
            return None, None
        blocks.append(block)

    return remaining, blocks


# ============================================================
# Extract #ninja blocks
#
# Unlike #n8n, the block body is kept verbatim as raw Ninja source
# rather than parsed into key/value fields - same shape as an
# #extern block.
# ============================================================

def extract_ninja_blocks(source: str) -> Tuple[Optional[str], Optional[List[NinjaBlock]]]:
    remaining, raw_blocks = _extract_named_brace_blocks(source, "#ninja")
    if remaining is None:
        return None, None

    blocks = [NinjaBlock(name, body) for name, body in raw_blocks]
    return remaining, blocks


# ============================================================
# NEW: extract #pyninja blocks
#
# Same shape as #ninja - the block body is kept verbatim as raw
# PY_NINJA source.
# ============================================================

def extract_pyninja_blocks(source: str) -> Tuple[Optional[str], Optional[List[PyNinjaBlock]]]:
    remaining, raw_blocks = _extract_named_brace_blocks(source, "#pyninja")
    if remaining is None:
        return None, None

    blocks = [PyNinjaBlock(name, body) for name, body in raw_blocks]
    return remaining, blocks


# ============================================================
# NEW: NgpuBlock and #extern "ngpu" extraction
#
# #extern "ngpu" { ... } is extracted separately from the generic
# #extern pipeline.  The embedded NGPU source is:
#
#   1. Written to NGPU_SOURCE_DIR as kyber_ngpu_N.ngpu.
#   2. Passed to ngpu_compiler.py, which translates it to PTX and
#      invokes nvcc to produce a COFF .obj.
#   3. The resulting .obj lands in OBJECT_DIRECTORY as
#      kyber_ngpu_N.obj, exactly like every other extern block.
#
# kyberlink then links kyber_ngpu_N.obj in the final binary alongside
# all other extern, n8n, ninja, and pyninja objects.
# ============================================================

@dataclass
class NgpuBlock:
    source: str


def _extract_dedicated_extern_blocks(source: str, language: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Shared helper for dedicated #extern pipelines (ngpu, xasm, npu).

    Scans `source` for  #extern "<language>" { ... }  blocks and
    returns (remaining_source_with_those_blocks_stripped, [body, ...]).
    Non-matching #extern blocks are left in the remaining source so
    that extract_extern_blocks (which runs on the same original text)
    can still handle them.  Returns (None, None) on a malformed block.
    """
    remaining: List[str] = []
    bodies: List[str] = []
    pos = 0
    marker = '#extern'
    marker_len = len(marker)

    while pos < len(source):
        idx = source.find(marker, pos)
        if idx == -1:
            remaining.append(source[pos:])
            break

        quote1 = source.find('"', idx)
        if quote1 == -1:
            remaining.append(source[pos:])
            break
        quote2 = source.find('"', quote1 + 1)
        if quote2 == -1:
            remaining.append(source[pos:])
            break

        found_language = source[quote1 + 1:quote2]

        if found_language != language:
            # Not our language — keep this text and advance past the marker
            # so we don't re-match it on the next iteration.
            remaining.append(source[pos:idx + marker_len])
            pos = idx + marker_len
            continue

        # Our language — strip this block.
        remaining.append(source[pos:idx])

        open_brace = source.find("{", quote2)
        if open_brace == -1:
            print(f'Malformed #extern "{language}" block: missing \'{{\'', file=sys.stderr)
            return None, None

        end = find_block_end(source, open_brace)
        if end == -1:
            print(f'Unclosed #extern "{language}" block', file=sys.stderr)
            return None, None

        body = source[open_brace + 1:end - 1]
        bodies.append(body)
        pos = end

    return "".join(remaining), bodies


def extract_ngpu_blocks(source: str) -> Tuple[Optional[str], Optional[List[NgpuBlock]]]:
    """
    Scan `source` for #extern "ngpu" { ... } blocks.  Each block's
    body is kept verbatim as raw NGPU source.  Returns the source
    with all #extern "ngpu" blocks removed, plus the list of blocks.
    """
    remaining, bodies = _extract_dedicated_extern_blocks(source, "ngpu")
    if bodies is None:
        return None, None
    return remaining, [NgpuBlock(b) for b in bodies]


# ============================================================
# NEW: XasmBlock and #extern "xasm" extraction
#
# #extern "xasm" { ... } is extracted separately from the generic
# #extern pipeline. The embedded XASM source is:
#
#   1. Written to XASM_SOURCE_DIR as kyber_xasm_N.xasm.
#   2. Passed to xasm_compiler.py, which translates it to NASM
#      and invokes nasm to produce a COFF .obj.
#   3. The resulting .obj lands in OBJECT_DIRECTORY as
#      kyber_xasm_N.obj, exactly like every other extern block.
#
# kyberlink links kyber_xasm_N.obj in the final binary alongside
# all other extern objects.
# ============================================================

@dataclass
class XasmBlock:
    source: str


def extract_xasm_blocks(source: str) -> Tuple[Optional[str], Optional[List[XasmBlock]]]:
    """
    Scan `source` for #extern "xasm" { ... } blocks.
    Returns (remaining_source_with_xasm_stripped, [XasmBlock, ...])
    or (None, None) on a malformed block.
    """
    remaining, bodies = _extract_dedicated_extern_blocks(source, "xasm")
    if bodies is None:
        return None, None
    return remaining, [XasmBlock(b) for b in bodies]


# ============================================================
# NEW: NpuBlock and #extern "npu" extraction
#
# #extern "npu" { ... } contains KNPU source which is:
#   1. Written to NPU_SOURCE_DIR as kyber_npu_N.npu
#   2. Compiled by npu_compiler.py to DirectML C++
#   3. Compiled by clang++ to a COFF .obj
#   4. Linked by kyberlink (needs d3d12.lib dxgi.lib DirectML.lib)
# ============================================================

@dataclass
class NpuBlock:
    source: str


def extract_npu_blocks(source: str) -> Tuple[Optional[str], Optional[List[NpuBlock]]]:
    """
    Scan source for #extern "npu" { ... } blocks.
    Returns (remaining_source, [NpuBlock, ...]) or (None, None).
    """
    remaining, bodies = _extract_dedicated_extern_blocks(source, "npu")
    if bodies is None:
        return None, None
    return remaining, [NpuBlock(b) for b in bodies]


def compile_npu_block(block: NpuBlock, object_path: Path, index: int) -> bool:
    """
    1. Write block.source to NPU_SOURCE_DIR/kyber_npu_<index>.npu
    2. Invoke npu_compiler.py <source.npu> <output.obj>
    3. Verify the .obj was produced.
    """
    try:
        NPU_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create NPU source directory: {e}", file=sys.stderr)
        return False

    try:
        OBJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create .obj directory: {e}", file=sys.stderr)
        return False

    npu_source_path = NPU_SOURCE_DIR / f"kyber_npu_{index}.npu"
    if not write_file(npu_source_path, block.source):
        print(f"Could not write NPU source file: {npu_source_path}", file=sys.stderr)
        return False

    print(f"\n[Kyber NPU]\nSource : {npu_source_path}\nOutput : {object_path}")

    cmd = [
        KYBER_PYTHON,
        str(NPU_COMPILER),
        str(npu_source_path),
        str(object_path),
    ]

    try:
        result = subprocess.run(cmd)
        rc = result.returncode
    except FileNotFoundError as e:
        print(f"Could not invoke NPU compiler: {e}", file=sys.stderr)
        return False

    if rc != 0:
        print(f"NPU compiler failed (exit {rc}) for block {index}.", file=sys.stderr)
        return False

    if not object_path.exists():
        print(
            f"NPU compiler returned success but object file was not created:\n{object_path}",
            file=sys.stderr,
        )
        return False

    print(f"Generated: {object_path}")
    return True


def compile_xasm_block(block: XasmBlock, object_path: Path, index: int) -> bool:
    """
    1. Ensure XASM_SOURCE_DIR exists.
    2. Write block.source to XASM_SOURCE_DIR/kyber_xasm_<index>.xasm.
    3. Invoke xasm_compiler.py <source.xasm> <output.obj>.
    4. Verify the .obj was produced.
    Returns True on success, False on any failure.
    """
    try:
        XASM_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create XASM source directory: {e}", file=sys.stderr)
        return False

    try:
        OBJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create .obj directory: {e}", file=sys.stderr)
        return False

    xasm_source_path = XASM_SOURCE_DIR / f"kyber_xasm_{index}.xasm"
    if not write_file(xasm_source_path, block.source):
        print(f"Could not write XASM source file: {xasm_source_path}", file=sys.stderr)
        return False

    print(f"\n[Kyber XASM]\nSource : {xasm_source_path}\nOutput : {object_path}")

    target = os.environ.get("KYBER_TARGET", "cpu").strip().lower()
    cmd = [
        KYBER_PYTHON,
        str(XASM_COMPILER),
        str(xasm_source_path),
        str(object_path),
        "--target",
        target,
    ]

    try:
        result = subprocess.run(cmd)
        rc = result.returncode
    except FileNotFoundError as e:
        print(f"Could not invoke XASM compiler: {e}", file=sys.stderr)
        return False

    if rc != 0:
        print(
            f"XASM compiler failed (exit {rc}) for block {index}.",
            file=sys.stderr,
        )
        return False

    if not object_path.exists():
        print(
            f"XASM compiler returned success but object file was not created:\n"
            f"{object_path}",
            file=sys.stderr,
        )
        return False

    print(f"Generated: {object_path}")
    return True


def compile_ngpu_block(block: NgpuBlock, object_path: Path, index: int) -> bool:
    """
    1. Ensure NGPU_SOURCE_DIR exists.
    2. Write block.source to NGPU_SOURCE_DIR/kyber_ngpu_<index>.ngpu.
    3. Invoke ngpu_compiler.py <source.ngpu> <object_path>.
    4. Verify the .obj was produced.
    Returns True on success, False on any failure.
    """
    # Ensure output directories exist
    try:
        NGPU_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create NGPU source directory: {e}", file=sys.stderr)
        return False

    try:
        OBJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create .obj directory: {e}", file=sys.stderr)
        return False

    # Write the .ngpu source file
    ngpu_source_path = NGPU_SOURCE_DIR / f"kyber_ngpu_{index}.ngpu"
    if not write_file(ngpu_source_path, block.source):
        print(f"Could not write NGPU source file: {ngpu_source_path}", file=sys.stderr)
        return False

    print(f"\n[Kyber NGPU]\nSource : {ngpu_source_path}\nOutput : {object_path}")

    # Invoke ngpu_compiler.py
    cmd = [
        KYBER_PYTHON,
        str(NGPU_COMPILER),
        str(ngpu_source_path),
        str(object_path),
    ]

    try:
        result = subprocess.run(cmd)
        rc = result.returncode
    except FileNotFoundError as e:
        print(f"Could not invoke NGPU compiler: {e}", file=sys.stderr)
        return False

    if rc != 0:
        print(
            f"NGPU compiler failed with exit code {rc} for block {index}.",
            file=sys.stderr,
        )
        return False

    if not object_path.exists():
        print(
            f"NGPU compiler returned success, but object file was not created:\n"
            f"{object_path}",
            file=sys.stderr,
        )
        return False

    print(f"Generated: {object_path}")
    return True


# ============================================================
# Generate the C source for a compiled #n8n block
# ============================================================

def generate_n8n_runtime_source(block: N8nBlock) -> str:
    payload_literal = "NULL" if not block.payload else f'"{escape_for_c(block.payload)}"'

    return f"""#include <stddef.h>
#include <curl/curl.h>

// Auto-generated by Kyber for #n8n "{block.name}"
int {block.name}(const char* payload) {{

    CURL* curl = curl_easy_init();
    if (!curl) return -1;

    const char* effectivePayload = payload;
    if (!effectivePayload) {{
        effectivePayload = {payload_literal};
    }}

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(
        headers,
        "Content-Type: application/json"
    );

    curl_easy_setopt(curl, CURLOPT_URL, "{escape_for_c(block.url)}");
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "{escape_for_c(block.method)}");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    if (effectivePayload) {{
        curl_easy_setopt(
            curl,
            CURLOPT_POSTFIELDS,
            effectivePayload
        );
    }}

    CURLcode result = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return result == CURLE_OK ? 0 : -1;
}}
"""


# ============================================================
# Generate the C source for a compiled #ninja block
# ============================================================

def generate_ninja_runtime_source(block: NinjaBlock) -> str:
    return _generate_shellout_runtime_source(
        func_name=block.name,
        source=block.source,
        marker="ninja",
        env_var="KYBER_NINJA_INTERPRETER",
        default_path=NINJA_INTERPRETER_DEFAULT,
    )


# ============================================================
# NEW: generate the C source for a compiled #pyninja block
#
# Identical wrapper shape to #ninja, but targets py_ninja.py /
# KYBER_PYNINJA_INTERPRETER instead of NINJA--COMPILER.py /
# KYBER_NINJA_INTERPRETER.
# ============================================================

def generate_pyninja_runtime_source(block: PyNinjaBlock) -> str:
    return _generate_shellout_runtime_source(
        func_name=block.name,
        source=block.source,
        marker="pyninja",
        env_var="KYBER_PYNINJA_INTERPRETER",
        default_path=PYNINJA_INTERPRETER_DEFAULT,
    )


def _generate_shellout_runtime_source(
    func_name: str,
    source: str,
    marker: str,
    env_var: str,
    default_path: Path,
) -> str:
    escaped_source = escape_for_c_string_literal(source)
    escaped_default_path = escape_for_c_string_literal(str(default_path))

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define KYBER_GETPID _getpid
#else
#include <unistd.h>
#define KYBER_GETPID getpid
#endif

// Auto-generated by Kyber for #{marker} "{func_name}"
// Embeds the source below and runs it through the {marker} interpreter
// via python3 at runtime. Requires python3 and the interpreter to be
// available on the machine that runs this program (see {env_var}).

static const char* KYBER_{marker.upper()}_SOURCE_{func_name} =
    "{escaped_source}";

int {func_name}(void) {{

    const char* interpreter =
        getenv("{env_var}");

    if (!interpreter || !interpreter[0]) {{
        interpreter = "{escaped_default_path}";
    }}

    const char* tempDir = getenv("TEMP");
    if (!tempDir || !tempDir[0]) {{
        tempDir = getenv("TMPDIR");
    }}
    if (!tempDir || !tempDir[0]) {{
        tempDir = "/tmp";
    }}

    char tempPath[1024];
    snprintf(
        tempPath,
        sizeof(tempPath),
        "%s/kyber_{marker}_{func_name}_%d.{marker}",
        tempDir,
        (int)KYBER_GETPID()
    );

    FILE* f = fopen(tempPath, "w");
    if (!f) return -1;
    fputs(KYBER_{marker.upper()}_SOURCE_{func_name}, f);
    fclose(f);

    char command[2048];
    snprintf(
        command,
        sizeof(command),
        "python3 \\"%s\\" \\"%s\\"",
        interpreter,
        tempPath
    );

    int result = system(command);

    remove(tempPath);

    return result;
}}
"""


# ============================================================
# File utilities
# ============================================================

def write_file(path: Path, data: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        return True
    except OSError:
        return False


def read_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# ============================================================
# NEW: toolkit-aware compiler/include/lib resolution for the CUDA
# and XLA #extern languages.
#
# Both toolkits are laid out the same conventional way:
#
#     <root>/bin/<compiler>
#     <root>/include
#     <root>/lib
#
# so this is one small helper shared by both, rather than two
# separate near-duplicate code paths.
# ============================================================

def _resolve_toolkit_compiler(
    toolkit_root: Path,
    binary_name: str,
    fallback: str,
) -> Tuple[str, List[str]]:
    """
    Returns (compiler_executable, extra_flags) for a block that needs
    to build against `toolkit_root`. Prefers <root>/bin/<binary_name>
    if it exists, otherwise falls back to `fallback` on PATH. Adds
    -I<root>/include and -L<root>/lib when those directories exist.
    """
    candidate = toolkit_root / "bin" / binary_name
    compiler = str(candidate) if candidate.exists() else fallback

    extra_flags: List[str] = []

    include_dir = toolkit_root / "include"
    if include_dir.exists():
        extra_flags += ["-I", str(include_dir)]

    lib_dir = toolkit_root / "lib"
    if lib_dir.exists():
        extra_flags += ["-L", str(lib_dir)]

    return compiler, extra_flags


# ============================================================
# Compile a C/C++ source file with Clang into an object file.
#
# Shared by #extern, #n8n, #ninja, and #pyninja compilation, since
# all of them are "write a temp source file, invoke clang -c, keep
# the .obj" pipelines. NEW: accepts optional extra_flags so the CUDA
# / XLA extern languages can pass toolkit include/lib directories
# through to nvcc / clang++ without duplicating this function.
# ============================================================

def compile_source_to_object(
    source_text: str,
    extension: str,
    compiler: str,
    object_path: Path,
    temp_file_tag: str,
    index: int,
    extra_flags: Optional[List[str]] = None,
) -> bool:
    temp_dir = Path(tempfile.gettempdir())
    temp_source = temp_dir / f"kyber_{temp_file_tag}_{index}{extension}"

    if not write_file(temp_source, source_text):
        print(f"Could not create temporary source:\n{temp_source}", file=sys.stderr)
        return False

    try:
        OBJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create Kyber .obj directory: {e}", file=sys.stderr)
        temp_source.unlink(missing_ok=True)
        return False

    command = [compiler, "-c", "-O2"]
    if extra_flags:
        command += extra_flags
    command += [str(temp_source), "-o", str(object_path)]

    print(f"\n[Kyber Clang]\nCompiler: {compiler}\nOutput: {object_path}")

    try:
        result = subprocess.run(command)
        return_code = result.returncode
    except FileNotFoundError:
        print(f"Could not find compiler '{compiler}' on PATH.", file=sys.stderr)
        temp_source.unlink(missing_ok=True)
        return False

    temp_source.unlink(missing_ok=True)

    if return_code != 0:
        print(f"Clang failed to compile {temp_file_tag} block.", file=sys.stderr)
        return False

    if not object_path.exists():
        print(
            f"Clang returned success, but object file was not created:\n{object_path}",
            file=sys.stderr,
        )
        return False

    print(f"Generated: {object_path}")
    return True


# ============================================================
# Compile C/C++/CUDA/XLA extern block
#
# "c" and "c++" behave exactly as before (clang / clang++, no extra
# flags). NEW: "cuda" resolves nvcc (and -I/-L flags) against
# CUDA_TOOLKIT_DEFAULT / KYBER_CUDA_ROOT; "xla" resolves clang++
# (and -I/-L flags) against XLA_TOOLKIT_DEFAULT / KYBER_XLA_ROOT.
# ============================================================

def compile_extern_block(block: ExternBlock, object_path: Path, index: int) -> bool:
    if block.language not in EXTERN_LANGUAGES:
        print(f"Unsupported extern language: {block.language}", file=sys.stderr)
        return False

    extension, default_compiler = EXTERN_LANGUAGES[block.language]

    if block.language == "cuda":
        compiler, extra_flags = _resolve_toolkit_compiler(
            CUDA_TOOLKIT_DEFAULT, "nvcc", default_compiler
        )
        return compile_source_to_object(
            block.source, extension, compiler, object_path, "extern", index, extra_flags
        )

    if block.language == "xla":
        compiler, extra_flags = _resolve_toolkit_compiler(
            XLA_TOOLKIT_DEFAULT, "clang++", default_compiler
        )
        return compile_source_to_object(
            block.source, extension, compiler, object_path, "extern", index, extra_flags
        )

    return compile_source_to_object(
        block.source, extension, default_compiler, object_path, "extern", index
    )


# ============================================================
# Compile a #n8n block into its own object file.
#
# The generated function calls into libcurl, so the FINAL LINK step
# (kyberlink.exe) needs "-lcurl" alongside the usual NForce libraries
# whenever any #n8n blocks were compiled.
# ============================================================

def compile_n8n_block(block: N8nBlock, object_path: Path, index: int) -> bool:
    source = generate_n8n_runtime_source(block)
    return compile_source_to_object(source, ".c", _resolve_clang("clang"), object_path, "n8n", index)


# ============================================================
# Compile a #ninja block into its own object file. The generated
# wrapper shells out to python3 at runtime, so nothing Ninja-specific
# is needed at COMPILE time beyond clang itself.
# ============================================================

def compile_ninja_block(block: NinjaBlock, object_path: Path, index: int) -> bool:
    source = generate_ninja_runtime_source(block)
    return compile_source_to_object(source, ".c", _resolve_clang("clang"), object_path, "ninja", index)


# ============================================================
# NEW: compile a #pyninja block into its own object file. Same idea
# as #ninja, but the wrapper shells out to python3 py_ninja.py.
# ============================================================

def compile_pyninja_block(block: PyNinjaBlock, object_path: Path, index: int) -> bool:
    source = generate_pyninja_runtime_source(block)
    return compile_source_to_object(source, ".c", _resolve_clang("clang"), object_path, "pyninja", index)


# ============================================================
# LLVM lowering / object generation (via llvmlite)
# ============================================================

def lower_to_llvm(ast: ProgramNode, module_name: str):
    if not LLVMLITE_AVAILABLE:
        return None

    module = llvm_ir.Module(name=module_name)
    module.triple = llvm_binding.get_default_triple()

    if ast is not None:
        ast.codegen(module)

    return module


def emit_object_file(module, output_path: Path) -> bool:
    if not LLVMLITE_AVAILABLE:
        print(
            "llvmlite is not installed - install it with "
            "'pip install llvmlite' to enable LLVM object emission.",
            file=sys.stderr,
        )
        return False

    # llvmlite now initializes LLVM automatically; calling the deprecated
    # initialize() entry point raises an exception in newer versions.
    try:
        llvm_binding.initialize_native_target()
    except (AttributeError, RuntimeError):
        pass
    try:
        llvm_binding.initialize_native_asmprinter()
    except (AttributeError, RuntimeError):
        pass

    triple = llvm_binding.get_default_triple()
    module.triple = triple

    try:
        target = llvm_binding.Target.from_triple(triple)
    except RuntimeError as e:
        print(f"LLVM target lookup failed: {e}", file=sys.stderr)
        return False

    target_machine = target.create_target_machine()

    try:
        llvm_mod = llvm_binding.parse_assembly(str(module))
        llvm_mod.verify()
    except RuntimeError as e:
        print(f"LLVM verification failed: {e}", file=sys.stderr)
        return False

    llvm_mod.data_layout = str(target_machine.target_data)

    try:
        object_data = target_machine.emit_object(llvm_mod)
    except Exception as e:  # noqa: BLE001 - mirror the original's broad failure handling
        print(f"LLVM cannot emit object file: {e}", file=sys.stderr)
        return False

    try:
        with open(output_path, "wb") as f:
            f.write(object_data)
    except OSError as e:
        print(f"Could not open object output: {output_path}\n{e}", file=sys.stderr)
        return False

    return True


# ============================================================
# Usage
# ============================================================

def print_usage():
    print(f"""Kyber Compiler
==============

Usage:
  kyberc.py <source.kyber>

Output:
  {KYBER_OBJECT}

Extern output:
  {OBJECT_DIRECTORY}\\kyber_extern_N.obj

extern languages:
  #extern "c"      -> compiled with clang
  #extern "c++"    -> compiled with clang++
  #extern "cuda"   (NEW) -> compiled with nvcc from the CUDA
                     toolkit at:
                       {CUDA_TOOLKIT_DEFAULT}
                     override with the KYBER_CUDA_ROOT env var.
                     Needs -lcudart (and the toolkit's lib dir)
                     at link time.
  #extern "xla"    (NEW) -> compiled with clang++ against the XLA
                     toolkit at:
                       {XLA_TOOLKIT_DEFAULT}
                     override with the KYBER_XLA_ROOT env var.
                     Needs the toolkit's XLA runtime libraries
                     at link time.

  #extern "ngpu"   (NEW) -> NGPU source compiled to PTX by
                     ngpu_compiler.py, then assembled to COFF .obj
                     by nvcc. Source is written to:
                       {NGPU_SOURCE_DIR}
                     override with KYBER_NGPU_DIR.
                     Requires nvcc (CUDA toolkit) at compile time
                     and -lcudart at link time (same as "cuda").
                     NGPU compiler: {NGPU_COMPILER}

n8n output:
  {OBJECT_DIRECTORY}\\kyber_n8n_N.obj

n8n blocks:
  #n8n "workflow_name"
  {{
      url: "https://your-n8n-host/webhook/..."
      method: "POST"
      payload: "{{\\"status\\":\\"ok\\"}}"
  }}

  Compiles to an int workflow_name(const char* payload)
  function that POSTs to the webhook via libcurl.
  Requires libcurl at compile and link time.

Ninja output:
  {OBJECT_DIRECTORY}\\kyber_ninja_N.obj

ninja blocks:
  #ninja "workflow_name"
  {{
      <raw Ninja source>
  }}

  Compiles to an int workflow_name(void) function that
  writes the embedded Ninja source to a temp file and
  runs it via "python3 NINJA--COMPILER.py <file>".
  Requires python3 and NINJA--COMPILER.py at RUNTIME,
  default path:
    {NINJA_INTERPRETER_DEFAULT}
  override with the KYBER_NINJA_INTERPRETER env var.

PyNinja output (NEW):
  {OBJECT_DIRECTORY}\\kyber_pyninja_N.obj

pyninja blocks (NEW):
  #pyninja "workflow_name"
  {{
      <raw PY_NINJA source>
  }}

  Compiles to an int workflow_name(void) function that
  writes the embedded PY_NINJA source to a temp file and
  runs it via "python3 py_ninja.py <file>".
  Requires python3 and py_ninja.py at RUNTIME,
  default path:
    {PYNINJA_INTERPRETER_DEFAULT}
  override with the KYBER_PYNINJA_INTERPRETER env var.

NGPU output (NEW):
  {OBJECT_DIRECTORY}\\kyber_ngpu_N.obj
  NGPU source: {NGPU_SOURCE_DIR}\\kyber_ngpu_N.ngpu

ngpu blocks (NEW):
  #extern "ngpu"
  {{
      kernel void myKernel(float* output, float* input) {{
          int i = threadIdx.x;
          output[i] = input[i] * 2.0;
      }}
  }}

  The NGPU source is written to the ngpu/ directory as a .ngpu
  file, then compiled to PTX by ngpu_compiler.py, then assembled
  to a native COFF .obj by nvcc.  The resulting object links like
  any CUDA object.  Requires nvcc (CUDA toolkit) at COMPILE time
  and -lcudart at LINK time.
  NGPU compiler: {NGPU_COMPILER}
""")  


# ============================================================
# Main
# ============================================================

def compile_cpp_file(input_path: Path, extra_args: List[str]) -> int:
    """
    Forward a C / C++ source file directly to clang / clang++.
    Acts as a thin branded wrapper: kyberc myfile.cpp -O2 -o myapp
    The caller can pass any extra clang flags after the file name.

    Supported extensions and the compiler used:
        .c              -> clang
        .cpp .cc .cxx   -> clang++
        .cu             -> nvcc  (CUDA)
    """
    ext = input_path.suffix.lower()

    if ext == ".c":
        compiler = "clang"
    elif ext in (".cpp", ".cc", ".cxx"):
        compiler = "clang++"
    elif ext == ".cu":
        nvcc_candidate = CUDA_TOOLKIT_DEFAULT / "bin" / "nvcc.exe"
        compiler = str(nvcc_candidate) if nvcc_candidate.exists() else "nvcc"
    else:
        print(f"kyberc: unsupported file type '{ext}'", file=sys.stderr)
        return 1

    cmd = [compiler, str(input_path)] + extra_args

    print(f"[Kyber C++ pass-through]\n  {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except FileNotFoundError:
        print(
            f"kyberc: compiler '{compiler}' not found on PATH.\n"
            "Install clang or add it to your PATH.",
            file=sys.stderr,
        )
        return 1


def run_kyber_source_file(path: Path, target: str = "auto", json_output: bool = False) -> int:
    """Execute a Kyber source file through the software hardware runtime."""
    if not path.exists():
        print(f"[kyber run] Source not found: {path}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    try:
        from runtime.kyber_runtime import HardwareRuntime
    except Exception as exc:
        print(f"[kyber run] Could not import runtime: {exc}", file=sys.stderr)
        return 1

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[kyber run] Could not read source: {path} ({exc})", file=sys.stderr)
        return 1
    if not source.strip():
        print(f"[kyber run] Source is empty: {path}", file=sys.stderr)
        return 1

    result = HardwareRuntime(target).run(source, target=target)
    payload = {
        "target": result.target,
        "status": result.status,
        "summary": result.summary,
        "metrics": result.metrics,
        "output": result.output,
    }

    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"[kyber run] target={result.target} status={result.status}")
        print(result.summary)
        print(json.dumps(payload["output"], indent=2, sort_keys=True))
    return 0


# Extensions that bypass the Kyber pipeline and go straight to clang.
_CPP_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".cu"}


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print_usage()
        return 1

    if argv[1].lower() == "run":
        target = "auto"
        json_output = False
        file_arg = None
        idx = 2
        while idx < len(argv):
            arg = argv[idx]
            if arg == "--target" and idx + 1 < len(argv):
                target = argv[idx + 1]
                idx += 2
            elif arg == "--json":
                json_output = True
                idx += 1
            elif file_arg is None:
                file_arg = arg
                idx += 1
            else:
                print(f"[kyber run] Unexpected argument: {arg}", file=sys.stderr)
                return 1
        if file_arg is None:
            file_arg = "sample.kyber"
        return run_kyber_source_file(Path(file_arg), target=target, json_output=json_output)

    # ── Load project.env.yml if present ──────────────────────────
    # This exports KYBER_* env vars and creates project folders
    # before anything else runs. Silently skipped if no file found.
    try:
        _compiler_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(_compiler_dir))
        from project_env import load_project_env
        load_project_env()
    except Exception:
        pass  # project.env is optional — never block compilation

    input_path = Path(argv[1])

    # --------------------------------------------------------
    # C / C++ / CUDA pass-through mode.
    # If the file is a plain C or C++ source, skip the entire
    # Kyber pipeline and compile it directly with clang/clang++.
    # Any extra arguments after the filename are forwarded as-is.
    #   kyberc myfile.cpp -O2 -o myapp
    #   kyberc myfile.c  -std=c11 -o mylib.obj
    # --------------------------------------------------------
    if input_path.suffix.lower() in _CPP_EXTENSIONS:
        return compile_cpp_file(input_path, argv[2:])

    # --------------------------------------------------------
    # Read source.
    # --------------------------------------------------------

    source = read_file(input_path)
    if not source:
        print(f"Could not read Kyber source:\n{input_path}", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Ensure .obj directory exists.
    # --------------------------------------------------------

    try:
        OBJECT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create:\n{OBJECT_DIRECTORY}\n{e}", file=sys.stderr)
        return 1

    # ── Resolve IMPORT() declarations ────────────────────────────
    # Strip IMPORT(pkg) lines from source and compile/copy the
    # package objects into OBJECT_DIRECTORY before any other step.
    source, import_names = extract_imports(source)
    if not resolve_imports(import_names):
        return 1

    # ── Extract #extern "xasm" blocks (dedicated pipeline) ───────
    _, xasm_blocks = extract_xasm_blocks(source)
    if xasm_blocks is None:
        print('Failed to extract #extern "xasm" blocks.', file=sys.stderr)
        return 1

    # ── Extract #extern "npu" blocks (KNPU/DirectML pipeline) ────
    _, npu_blocks = extract_npu_blocks(source)
    if npu_blocks is None:
        print('Failed to extract #extern "npu" blocks.', file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Extract #extern "ngpu" blocks FIRST (before the generic
    # extern pass) so that extract_extern_blocks can safely skip
    # the "ngpu" language without losing the source bodies.
    # Both functions scan the same original `source` string.
    # --------------------------------------------------------

    _, ngpu_blocks = extract_ngpu_blocks(source)
    if ngpu_blocks is None:
        print("Failed to extract #extern \"ngpu\" blocks.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Extract C/C++/CUDA/XLA extern blocks.
    # (ngpu blocks are silently skipped inside this function.)
    # --------------------------------------------------------

    after_extern, extern_blocks = extract_extern_blocks(source)
    if after_extern is None:
        print("Failed to extract #extern blocks.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Extract #n8n blocks.
    # --------------------------------------------------------

    after_n8n, n8n_blocks = extract_n8n_blocks(after_extern)
    if after_n8n is None:
        print("Failed to extract #n8n blocks.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Extract #ninja blocks.
    # --------------------------------------------------------

    after_ninja, ninja_blocks = extract_ninja_blocks(after_n8n)
    if after_ninja is None:
        print("Failed to extract #ninja blocks.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # NEW: extract #pyninja blocks.
    # --------------------------------------------------------

    kyber_source, pyninja_blocks = extract_pyninja_blocks(after_ninja)
    if kyber_source is None:
        print("Failed to extract #pyninja blocks.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Lex Kyber code.
    # --------------------------------------------------------

    lexer = KyberLexer(kyber_source)
    tokens = lexer.tokenize()

    # --------------------------------------------------------
    # Parse Kyber code.
    # --------------------------------------------------------

    source_lines = kyber_source.splitlines()
    parser = KyberParser(tokens, source_lines)
    ast = parser.parse()

    if ast is None:
        print("Kyber parser failed.", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # LLVM generation.
    # --------------------------------------------------------

    module = lower_to_llvm(ast, "kyber_module")

    # --------------------------------------------------------
    # Generate Kyber object.
    # --------------------------------------------------------

    print(f"\n[Kyber LLVM]\nGenerating:\n{KYBER_OBJECT}")

    if module is None or not emit_object_file(module, KYBER_OBJECT):
        print("Could not generate Kyber object file.", file=sys.stderr)
        return 1

    print(f"Generated: {KYBER_OBJECT}")

    # --------------------------------------------------------
    # Compile C/C++/CUDA/XLA extern blocks.
    # --------------------------------------------------------

    for i, block in enumerate(extern_blocks):
        extern_object = OBJECT_DIRECTORY / f"kyber_extern_{i}.obj"
        if not compile_extern_block(block, extern_object, i):
            return 1

    # --------------------------------------------------------
    # Compile #n8n blocks.
    # --------------------------------------------------------

    for i, block in enumerate(n8n_blocks):
        n8n_object = OBJECT_DIRECTORY / f"kyber_n8n_{i}.obj"
        if not compile_n8n_block(block, n8n_object, i):
            return 1

    # --------------------------------------------------------
    # Compile #ninja blocks.
    # --------------------------------------------------------

    for i, block in enumerate(ninja_blocks):
        ninja_object = OBJECT_DIRECTORY / f"kyber_ninja_{i}.obj"
        if not compile_ninja_block(block, ninja_object, i):
            return 1

    # --------------------------------------------------------
    # NEW: compile #pyninja blocks.
    # --------------------------------------------------------

    for i, block in enumerate(pyninja_blocks):
        pyninja_object = OBJECT_DIRECTORY / f"kyber_pyninja_{i}.obj"
        if not compile_pyninja_block(block, pyninja_object, i):
            return 1

    # --------------------------------------------------------
    # NEW: compile #extern "ngpu" blocks.
    # Each block's .ngpu source is written to NGPU_SOURCE_DIR and
    # then compiled to a COFF .obj via ngpu_compiler.py → nvcc.
    # --------------------------------------------------------

    for i, block in enumerate(ngpu_blocks):
        ngpu_object = OBJECT_DIRECTORY / f"kyber_ngpu_{i}.obj"
        if not compile_ngpu_block(block, ngpu_object, i):
            return 1

    # --------------------------------------------------------
    # NEW: compile #extern "xasm" blocks.
    # Each block's .xasm source is written to XASM_SOURCE_DIR and
    # then compiled to a COFF .obj via xasm_compiler.py → nasm.
    # --------------------------------------------------------

    for i, block in enumerate(xasm_blocks):
        xasm_object = OBJECT_DIRECTORY / f"kyber_xasm_{i}.obj"
        if not compile_xasm_block(block, xasm_object, i):
            return 1

    # ── Compile #extern "npu" blocks ──────────────────────────────
    for i, block in enumerate(npu_blocks):
        npu_object = OBJECT_DIRECTORY / f"kyber_npu_{i}.obj"
        if not compile_npu_block(block, npu_object, i):
            return 1

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This compiler DOES NOT LINK.
    #
    # kyberlink is responsible for:
    #
    #   kyber.obj
    #   kyber_extern_0.obj, kyber_extern_1.obj, ...
    #     (language may be c, c++, cuda, or xla per-block - NEW)
    #   kyber_n8n_0.obj, kyber_n8n_1.obj, ...
    #   kyber_ninja_0.obj, kyber_ninja_1.obj, ...
    #   kyber_pyninja_0.obj, kyber_pyninja_1.obj, ...   (NEW)
    #   kyber_ngpu_0.obj, kyber_ngpu_1.obj, ...         (NEW)
    #   NForce libraries
    #   -lcurl (only if there are #n8n blocks)
    #   -lcudart plus the CUDA toolkit lib dir (only if there are
    #     #extern "cuda" or #extern "ngpu" blocks)        (NEW)
    #   the XLA toolkit's runtime libraries plus its lib dir
    #     (only if there are #extern "xla" blocks)         (NEW)
    #
    # and producing the final executable.
    #
    # #ninja and #pyninja objects don't need anything special at LINK
    # time, but at RUN time the machine executing the program needs
    # python3 and either NINJA--COMPILER.py or py_ninja.py available
    # (see KYBER_NINJA_INTERPRETER / NINJA_INTERPRETER_DEFAULT and
    # KYBER_PYNINJA_INTERPRETER / PYNINJA_INTERPRETER_DEFAULT).
    #
    # #extern "ngpu" objects are fully native COFF .obj files produced
    # by nvcc from PTX and can be linked like any CUDA object.
    # --------------------------------------------------------

    cuda_count = sum(1 for b in extern_blocks if b.language == "cuda")
    xla_count = sum(1 for b in extern_blocks if b.language == "xla")

    print(
        "\n====================================\n"
        "Kyber compilation successful!\n"
        "====================================\n"
        f"Object directory:\n{OBJECT_DIRECTORY}\n\n"
        f"Kyber object:\n{KYBER_OBJECT}\n"
        f"Extern objects generated: {len(extern_blocks)} "
        f"(cuda: {cuda_count}, xla: {xla_count})\n"
        f"n8n objects generated: {len(n8n_blocks)}\n"
        f"ninja objects generated: {len(ninja_blocks)}\n"
        f"pyninja objects generated: {len(pyninja_blocks)}\n"
        f"ngpu objects generated: {len(ngpu_blocks)}\n"
        f"xasm objects generated: {len(xasm_blocks)}\n"
        f"npu objects generated:  {len(npu_blocks)}\n"
        f"packages imported:      {len(import_names)}"
    )

    if n8n_blocks:
        print("\nNote: #n8n blocks require libcurl.\nLink the final executable with -lcurl.")

    if cuda_count:
        print(
            "\nNote: #extern \"cuda\" blocks require the CUDA\n"
            "toolkit at COMPILE time (nvcc) and -lcudart plus\n"
            "the toolkit's lib dir at LINK time.\n"
            f"Default toolkit path:\n  {CUDA_TOOLKIT_DEFAULT}\n"
            "Override with KYBER_CUDA_ROOT if it's installed\n"
            "elsewhere."
        )

    if ngpu_blocks:
        print(
            "\nNote: #extern \"ngpu\" blocks are compiled to PTX\n"
            "by ngpu_compiler.py, then assembled to .obj by nvcc.\n"
            "NGPU source files are written to:\n"
            f"  {NGPU_SOURCE_DIR}\n"
            "The resulting objects require -lcudart (and the CUDA\n"
            "toolkit lib dir) at link time, same as #extern \"cuda\".\n"
            f"CUDA toolkit path:\n  {CUDA_TOOLKIT_DEFAULT}\n"
            "Override with KYBER_CUDA_ROOT.\n"
            f"NGPU compiler: {NGPU_COMPILER}"
        )

    if xasm_blocks:
        print(
            "\nNote: #extern \"xasm\" blocks are translated to NASM\n"
            "by xasm_compiler.py, then assembled to .obj by nasm.\n"
            "XASM source files are written to:\n"
            f"  {XASM_SOURCE_DIR}\n"
            "No extra link-time libraries required.\n"
            "Install NASM from https://nasm.us or set KYBER_NASM.\n"
            f"XASM compiler: {XASM_COMPILER}"
        )

    if npu_blocks:
        print(
            "\nNote: #extern \"npu\" (KNPU) blocks are compiled to\n"
            "DirectML C++ by npu_compiler.py, then compiled to .obj\n"
            "by clang++. NPU source files are written to:\n"
            f"  {NPU_SOURCE_DIR}\n"
            "Link with: d3d12.lib  dxgi.lib  DirectML.lib\n"
            "Override C++ compiler with KYBER_CXX env var.\n"
            f"NPU compiler: {NPU_COMPILER}"
        )

    if xla_count:
        print(
            "\nNote: #extern \"xla\" blocks require the XLA\n"
            "toolkit at COMPILE time (clang++) and its runtime\n"
            "libraries plus lib dir at LINK time.\n"
            f"Default toolkit path:\n  {XLA_TOOLKIT_DEFAULT}\n"
            "Override with KYBER_XLA_ROOT if it's installed\n"
            "elsewhere."
        )

    if ninja_blocks:
        print(
            "\nNote: #ninja blocks require python3 and\n"
            "NINJA--COMPILER.py at RUNTIME (not link time).\n"
            f"Default interpreter path:\n  {NINJA_INTERPRETER_DEFAULT}\n"
            "Override with KYBER_NINJA_INTERPRETER if it's\n"
            "installed elsewhere."
        )

    if pyninja_blocks:
        print(
            "\nNote: #pyninja blocks require python3 and\n"
            "py_ninja.py at RUNTIME (not link time).\n"
            f"Default interpreter path:\n  {PYNINJA_INTERPRETER_DEFAULT}\n"
            "Override with KYBER_PYNINJA_INTERPRETER if it's\n"
            "installed elsewhere."
        )

    print("\nNext stage:\nkyberlink")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))