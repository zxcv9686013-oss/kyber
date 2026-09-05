"""
PY_NINJA - The Ninja Programming Language Compiler/Interpreter
Original compiler source code written in Python.
Supports: VOID, MAIN, SCREEN.PRINT, SCANLN, variables (STR/INT/FLOAT/NUMB/BOOL),
          for/while loops, if/elseif/else, math functions, arrays, pointers,
          error handling (try/except), OOP (class), imports, VMA matrices,
          TENSORS (1-D/2-D with DOT, MATMUL, TRANSPOSE, RESHAPE),
          a small NETWORK/LAYER/TRAIN loop, and DEVICE(CPU|GPU|TPU|NPU)
          compute-backend selection.
"""

import sys
import re
import math
import os


# ─────────────────────────────────────────────
#  DEVICE BACKEND
# ─────────────────────────────────────────────
#
# PY_NINJA is a pure-Python interpreter, so it has no native GPU/TPU/NPU
# driver code of its own. What DEVICE() gives you is a real backend
# dispatcher: if an accelerator library is importable and reports a usable
# device, tensor/network math is routed through it; otherwise PY_NINJA
# prints a clear notice and falls back to the pure-Python implementation
# that already existed. Nothing here pretends to accelerate silently.
#
#   DEVICE(CPU)   -> always available, plain Python (default)
#   DEVICE(GPU)   -> tries torch.cuda, then torch (ROCm builds report as cuda)
#   DEVICE(TPU)   -> tries torch_xla
#   DEVICE(NPU)   -> tries torch.npu (Ascend) / torch_directml as a generic NPU shim
#
# Install `torch` (plus `torch_xla` / your vendor's NPU plugin) in your own
# environment for any of GPU/TPU/NPU to actually engage; without them,
# DEVICE(GPU/TPU/NPU) will warn once and keep running on CPU.

class NinjaDeviceError(Exception):
    pass


class DeviceBackend:
    """
    Tracks the currently selected compute device for the whole interpreter
    run and lazily probes for an accelerator library the first time a
    non-CPU device is requested.
    """
    VALID = ("CPU", "GPU", "TPU", "NPU")

    def __init__(self):
        self.requested = "CPU"     # what the .ninja program asked for
        self.active    = "CPU"     # what we actually ended up running on
        self.torch     = None      # cached torch module, if usable
        self.torch_device = None   # cached torch.device object
        self._probed   = {"CPU": True, "GPU": False, "TPU": False, "NPU": False}
        self._warned   = set()

    def set(self, requested):
        requested = requested.upper()
        if requested not in self.VALID:
            raise NinjaDeviceError(
                f"Unknown device '{requested}'. Use one of {self.VALID}."
            )
        self.requested = requested

        if requested == "CPU":
            self.active = "CPU"
            self.torch_device = None
            print("[NINJA DEVICE] Using CPU.")
            return

        ok, detail = self._probe(requested)
        if ok:
            self.active = requested
            print(f"[NINJA DEVICE] {requested} backend active ({detail}).")
        else:
            self.active = "CPU"
            if requested not in self._warned:
                print(f"[NINJA DEVICE] {requested} requested but unavailable "
                      f"({detail}). Falling back to CPU.")
                self._warned.add(requested)

    def _probe(self, requested):
        """Try to bring up a real backend for `requested`. Returns (ok, detail)."""
        try:
            import torch
        except ImportError:
            return False, "the 'torch' package is not installed"

        self.torch = torch

        if requested == "GPU":
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
                return True, f"torch cuda, device={torch.cuda.get_device_name(0)}"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.torch_device = torch.device("mps")
                return True, "torch mps (Apple Silicon GPU)"
            return False, "no CUDA/MPS device visible to torch"

        if requested == "TPU":
            try:
                import torch_xla.core.xla_model as xm
                self.torch_device = xm.xla_device()
                return True, "torch_xla xla_device"
            except ImportError:
                return False, "the 'torch_xla' package is not installed"

        if requested == "NPU":
            if hasattr(torch, "npu") and torch.npu.is_available():
                self.torch_device = torch.device("npu")
                return True, "torch.npu (Ascend)"
            try:
                import torch_directml
                self.torch_device = torch_directml.device()
                return True, "torch_directml (generic NPU/DirectML shim)"
            except ImportError:
                return False, "no torch.npu build and 'torch_directml' is not installed"

        return False, "unrecognised device"

    # ── helpers used by tensor/network math ──
    def usable(self):
        """True if a real accelerator (not plain CPU) is active."""
        return self.active != "CPU" and self.torch is not None and self.torch_device is not None

    def to_tensor(self, flat_list, shape):
        t = self.torch.tensor(flat_list, dtype=self.torch.float64, device=self.torch_device)
        return t.reshape(shape)

    def from_tensor(self, t):
        return t.detach().to("cpu").flatten().tolist()


DEVICE = DeviceBackend()


# ─────────────────────────────────────────────
#  LEXER
# ─────────────────────────────────────────────

TOKEN_PATTERNS = [
    ("COMMENT",     r"//[^\n]*"),
    ("NEWLINE",     r"\n"),
    ("WHITESPACE",  r"[ \t]+"),
    ("FLOAT_LIT",   r"\d+\.\d+"),
    ("INT_LIT",     r"\d+"),
    ("STRING_LIT",  r'"[^"]*"'),
    ("BOOL_LIT",    r"\b(TRUE|FALSE)\b"),
    ("LPAREN",      r"\("),
    ("RPAREN",      r"\)"),
    ("LBRACE",      r"\{"),
    ("RBRACE",      r"\}"),
    ("LANGLE",      r"<"),
    ("RANGLE",      r">"),
    ("LBRACKET",    r"\["),
    ("RBRACKET",    r"\]"),
    ("COMMA",       r","),
    ("SEMICOLON",   r";"),
    ("DOT",         r"\."),
    ("EQUALS",      r"=="),
    ("ASSIGN",      r"="),
    ("NEQ",         r"!="),
    ("LTE",         r"<="),
    ("GTE",         r">="),
    ("INCREMENT",   r"\+\+"),
    ("DECREMENT",   r"--"),
    ("PLUS",        r"\+"),
    ("MINUS",       r"-"),
    ("STAR",        r"\*"),
    ("SLASH",       r"/"),
    ("PERCENT",     r"%"),
    ("AMP",         r"&"),
    ("IDENT",       r"[A-Za-z_][A-Za-z0-9_]*"),
]

MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_PATTERNS)
)

KEYWORDS = {
    "VOID", "MAIN", "SCREEN", "PRINT", "SCANLN", "VAR",
    "STR", "INT", "FLOAT", "NUMB", "BOOL",
    "FOR", "WHILE", "IF", "ELSE", "ELSEIF",
    "RETURN", "IMPORT",
    "TRY", "EXCEPT", "FINALLY", "RAISE",
    "CLASS", "SELF", "NEW",
    "ADD", "SUB", "DIV", "MUL",
    "RELU", "GELU", "SIGMOID", "TANN",
    "CREATE", "VMA", "CHANGE", "ADD_ROW", "ADD_COL", "PRINT_VMA",
    "PTR", "POINT",
    "ARR", "TRUE", "FALSE", "NULL",

    # ── tensor keywords ──
    "TENSOR",
    "DOT_PRODUCT",
    "MATMUL",
    "TRANSPOSE",
    "RESHAPE",
    "PRINT_TENSOR",
    # ── training loop keywords ──
    "NETWORK",
    "LAYER",
    "TRAIN",
    "PREDICT",
    "LOSS_MSE",
    "LOSS_CE",
    "SAVE_MODEL",
    "LOAD_MODEL",
    # ── device keywords ──
    "DEVICE",
    "CPU",
    "GPU",
    "TPU",
    "NPU",
}


class Token:
    def __init__(self, type_, value, line):
        self.type  = type_
        self.value = value
        self.line  = line

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line})"


class LexerError(Exception):
    pass


def tokenize(source: str):
    tokens = []
    line   = 1
    pos    = 0
    while pos < len(source):
        m = MASTER_RE.match(source, pos)
        if not m:
            raise LexerError(f"Unexpected character {source[pos]!r} at line {line}")
        kind  = m.lastgroup
        value = m.group()
        if kind == "NEWLINE":
            line += 1
        elif kind in ("WHITESPACE", "COMMENT"):
            pass
        else:
            if kind == "IDENT":
                up = value.upper()
                if up in KEYWORDS:
                    kind  = up
                    value = up
            tokens.append(Token(kind, value, line))
        pos = m.end()
    tokens.append(Token("EOF", "", line))
    return tokens


# ─────────────────────────────────────────────
#  AST NODES
# ─────────────────────────────────────────────

class Node:
    pass

class Program(Node):
    def __init__(self, voids, main):
        self.voids = voids
        self.main  = main

class VoidDef(Node):
    def __init__(self, name, params, body, line):
        self.name   = name
        self.params = params
        self.body   = body
        self.line   = line

class MainBlock(Node):
    def __init__(self, params, body, line):
        self.params = params
        self.body   = body
        self.line   = line

class Block(Node):
    def __init__(self, stmts):
        self.stmts = stmts

# ── Statements ───────────────────────────────

class PrintStmt(Node):
    def __init__(self, expr, line):
        self.expr = expr
        self.line = line

class ScanlnStmt(Node):
    def __init__(self, var_type, var_name, line):
        self.var_type = var_type
        self.var_name = var_name
        self.line     = line

class VarDecl(Node):
    def __init__(self, var_type, var_name, line):
        self.var_type = var_type
        self.var_name = var_name
        self.line     = line

class AssignStmt(Node):
    def __init__(self, target, value, line):
        self.target = target
        self.value  = value
        self.line   = line

class CallStmt(Node):
    def __init__(self, name, args, line):
        self.name = name
        self.args = args
        self.line = line

class ForStmt(Node):
    def __init__(self, init, step_expr, cond, post, body, line):
        self.init      = init
        self.step_expr = step_expr
        self.cond      = cond
        self.post      = post
        self.body      = body
        self.line      = line

class WhileStmt(Node):
    def __init__(self, cond, body, line):
        self.cond = cond
        self.body = body
        self.line = line

class IfStmt(Node):
    def __init__(self, cond, then_block, elseifs, else_block, line):
        self.cond       = cond
        self.then_block = then_block
        self.elseifs    = elseifs
        self.else_block = else_block
        self.line       = line

class ReturnStmt(Node):
    def __init__(self, expr, line):
        self.expr = expr
        self.line = line

class ImportStmt(Node):
    def __init__(self, name, line):
        self.name = name
        self.line = line

class TryStmt(Node):
    def __init__(self, try_block, except_var, except_block, finally_block, line):
        self.try_block     = try_block
        self.except_var    = except_var
        self.except_block  = except_block
        self.finally_block = finally_block
        self.line          = line

class RaiseStmt(Node):
    def __init__(self, expr, line):
        self.expr = expr
        self.line = line

class ClassDef(Node):
    def __init__(self, name, parent, body, line):
        self.name   = name
        self.parent = parent
        self.body   = body
        self.line   = line

class ArrayDecl(Node):
    def __init__(self, name, elements, line):
        self.name     = name
        self.elements = elements
        self.line     = line

class ArrayAssign(Node):
    def __init__(self, name, index, value, line):
        self.name  = name
        self.index = index
        self.value = value
        self.line  = line

class VmaCreate(Node):
    def __init__(self, var, rows, line):
        self.var  = var
        self.rows = rows
        self.line = line

class VmaChange(Node):
    def __init__(self, var, row, col, value, line):
        self.var   = var
        self.row   = row
        self.col   = col
        self.value = value
        self.line  = line

class VmaAddRow(Node):
    def __init__(self, var, elements, line):
        self.var      = var
        self.elements = elements
        self.line     = line

class VmaAddCol(Node):
    def __init__(self, var, elements, line):
        self.var      = var
        self.elements = elements
        self.line     = line

class VmaPrint(Node):
    def __init__(self, var, line):
        self.var  = var
        self.line = line

class PointerDecl(Node):
    def __init__(self, ptr_name, target, line):
        self.ptr_name = ptr_name
        self.target   = target
        self.line     = line

class PointerAssign(Node):
    def __init__(self, ptr_name, value, line):
        self.ptr_name = ptr_name
        self.value    = value
        self.line     = line

# ── Device AST node ──────────────────────────

class DeviceStmt(Node):
    """DEVICE(CPU|GPU|TPU|NPU) — selects the compute backend for tensor/net math."""
    def __init__(self, device_name, line):
        self.device_name = device_name
        self.line         = line

# ── Tensor AST nodes ─────────────────────────

class TensorDecl(Node):
    """
    tensor(name) = <v0, v1, ..., vN>
    Flat 1-D declaration.  2-D tensors use comma-separated rows, same as VMA.
    """
    def __init__(self, name, elements, line):
        self.name     = name
        self.elements = elements   # list of expr  (1-D)  OR  list of list of expr (2-D)
        self.is_2d    = False      # set by parser when rows are detected
        self.line     = line

class TensorAssign(Node):
    """name = tensor_expr  (result of MATMUL, TRANSPOSE, etc.)"""
    def __init__(self, name, value, line):
        self.name  = name
        self.value = value
        self.line  = line

class TensorPrint(Node):
    def __init__(self, name, line):
        self.name = name
        self.line = line

class TensorOp(Node):
    """
    op          : DOT_PRODUCT | MATMUL | TRANSPOSE | RESHAPE
    operands[0] : tensor name (str Identifier)
    operands[1] : second tensor name or reshape dims
    """
    def __init__(self, op, operands):
        self.op       = op
        self.operands = operands   # list of expr nodes


# ── Internal expression helper nodes (used by parser → interpreter) ───────────

class _PredictExpr(Node):
    """RHS of:  result = PREDICT(net, X)"""
    def __init__(self, net_name, x_name):
        self.net_name = net_name
        self.x_name   = x_name

class _LoadModelExpr(Node):
    """RHS of:  net = LOAD_MODEL("path")"""
    def __init__(self, path_expr):
        self.path_expr = path_expr


# ── Training Loop AST nodes ──────────────────

class NetworkDecl(Node):
    """NETWORK(name) — declares a new neural network stored under `name`."""
    def __init__(self, name, line):
        self.name = name
        self.line = line

class LayerStmt(Node):
    """
    LAYER(net, type, in_size, out_size)
    type: DENSE | RELU | SIGMOID | TANH | SOFTMAX
    in_size / out_size are int expr nodes; None for activation-only layers.
    """
    def __init__(self, net_name, layer_type, in_size, out_size, line):
        self.net_name   = net_name
        self.layer_type = layer_type
        self.in_size    = in_size
        self.out_size   = out_size
        self.line       = line

class TrainStmt(Node):
    """TRAIN(net, X, Y, lr, epochs) — full training loop."""
    def __init__(self, net_name, x_name, y_name, lr_expr, epochs_expr, line):
        self.net_name    = net_name
        self.x_name      = x_name
        self.y_name      = y_name
        self.lr_expr     = lr_expr
        self.epochs_expr = epochs_expr
        self.line        = line

class PredictStmt(Node):
    """result = PREDICT(net, X) — inference forward pass."""
    def __init__(self, result_name, net_name, x_name, line):
        self.result_name = result_name
        self.net_name    = net_name
        self.x_name      = x_name
        self.line        = line

class LossExpr(Node):
    """LOSS_MSE(pred, target) or LOSS_CE(pred, target) — expression → scalar."""
    def __init__(self, loss_type, pred, target):
        self.loss_type = loss_type
        self.pred      = pred
        self.target    = target

class SaveModelStmt(Node):
    """SAVE_MODEL(net, "path") — serialise weights to JSON."""
    def __init__(self, net_name, filepath_expr, line):
        self.net_name      = net_name
        self.filepath_expr = filepath_expr
        self.line          = line

class LoadModelStmt(Node):
    """net = LOAD_MODEL("path") — restore weights from JSON."""
    def __init__(self, net_name, filepath_expr, line):
        self.net_name      = net_name
        self.filepath_expr = filepath_expr
        self.line          = line

# ── Expressions ──────────────────────────────

class BinOp(Node):
    def __init__(self, op, left, right):
        self.op    = op
        self.left  = left
        self.right = right

class UnaryOp(Node):
    def __init__(self, op, operand):
        self.op      = op
        self.operand = operand

class Literal(Node):
    def __init__(self, value):
        self.value = value

class Identifier(Node):
    def __init__(self, name):
        self.name = name

class FuncCall(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class ArrayIndex(Node):
    def __init__(self, name, index):
        self.name  = name
        self.index = index

class MathFunc(Node):
    def __init__(self, func, args):
        self.func = func
        self.args = args


# ─────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0

    # ── helpers ──────────────────────────────
    def peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, *types):
        tok = self.peek()
        if tok.type not in types:
            raise ParseError(
                f"Line {tok.line}: expected {types}, got {tok.type!r} ({tok.value!r})"
            )
        return self.advance()

    def match(self, *types):
        if self.peek().type in types:
            return self.advance()
        return None

    # ── top level ────────────────────────────
    def parse(self):
        voids   = []
        main    = None
        imports = []
        while self.peek().type != "EOF":
            tok = self.peek()
            if tok.type == "IMPORT":
                imports.append(self.parse_import())
            elif tok.type == "VOID":
                voids.append(self.parse_void())
            elif tok.type == "MAIN":
                main = self.parse_main()
            elif tok.type == "CLASS":
                voids.append(self.parse_class())
            elif tok.type == "DEVICE":
                # Allow a top-level DEVICE(...) before MAIN as a program-wide default.
                voids.append(self.parse_device_stmt())
            else:
                raise ParseError(
                    f"Line {tok.line}: unexpected token {tok.value!r} at top level"
                )
        return Program(voids, main), imports

    def parse_import(self):
        line = self.peek().line
        self.expect("IMPORT")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        return ImportStmt(name, line)

    def parse_void(self):
        line = self.peek().line
        self.expect("VOID")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        params = []
        if self.peek().type == "LPAREN":
            self.advance()
            while self.peek().type != "RPAREN":
                params.append(self.expect("IDENT").value)
                if self.peek().type == "COMMA":
                    self.advance()
            self.expect("RPAREN")
        self.expect("LBRACE")
        body = self.parse_block()
        self.expect("RBRACE")
        return VoidDef(name, params, body, line)

    def parse_main(self):
        line = self.peek().line
        self.expect("MAIN")
        params = []
        if self.peek().type == "LPAREN":
            self.advance()
            while self.peek().type != "RPAREN":
                params.append(self.expect("IDENT").value)
                if self.peek().type == "COMMA":
                    self.advance()
            self.expect("RPAREN")
        self.expect("LBRACE")
        body = self.parse_block()
        self.expect("RBRACE")
        return MainBlock(params, body, line)

    def parse_class(self):
        line = self.peek().line
        self.expect("CLASS")
        name = self.expect("IDENT").value
        parent = None
        if self.match("LPAREN"):
            parent = self.expect("IDENT").value
            self.expect("RPAREN")
        self.expect("LBRACE")
        body = self.parse_block()
        self.expect("RBRACE")
        return ClassDef(name, parent, body, line)

    # ── block ─────────────────────────────────
    def parse_block(self):
        stmts = []
        while self.peek().type not in ("RBRACE", "EOF"):
            s = self.parse_stmt()
            if s:
                stmts.append(s)
        return Block(stmts)

    # ── statements ────────────────────────────
    def parse_stmt(self):
        tok = self.peek()
        t   = tok.type

        if t == "SCREEN":        return self.parse_print()
        if t == "VAR":           return self.parse_var_decl()
        if t == "SCANLN":        return self.parse_scanln()
        if t == "FOR":           return self.parse_for()
        if t == "WHILE":         return self.parse_while()
        if t == "IF":            return self.parse_if()
        if t == "RETURN":        return self.parse_return()
        if t == "TRY":           return self.parse_try()
        if t == "RAISE":         return self.parse_raise()
        if t == "VOID":          return self.parse_void()
        if t == "CLASS":         return self.parse_class()

        # ── device statement ──────────────────
        if t == "DEVICE":        return self.parse_device_stmt()

        # ── tensor statements ─────────────────
        if t == "TENSOR":        return self.parse_tensor_decl()
        if t == "PRINT_TENSOR":  return self.parse_tensor_print()
        if t == "MATMUL":        return self.parse_tensor_op_stmt("MATMUL")
        if t == "TRANSPOSE":     return self.parse_tensor_op_stmt("TRANSPOSE")
        if t == "DOT_PRODUCT":   return self.parse_tensor_op_stmt("DOT_PRODUCT")
        if t == "RESHAPE":       return self.parse_tensor_op_stmt("RESHAPE")

        # ── training loop statements ───────────
        if t == "NETWORK":       return self.parse_network_decl()
        if t == "LAYER":         return self.parse_layer_stmt()
        if t == "TRAIN":         return self.parse_train_stmt()
        if t == "SAVE_MODEL":    return self.parse_save_model()
        if t == "LOAD_MODEL":    return self.parse_load_model()

        # Array declaration:  arr = <1,2,3>
        if t == "ARR" or (t == "IDENT" and self.peek(1).type == "ASSIGN"
                          and self.peek(2).type == "LANGLE"):
            return self.parse_array_decl()

        # VMA operations
        if t == "CREATE":    return self.parse_vma_create()
        if t == "CHANGE":    return self.parse_vma_change()
        if t == "ADD_ROW":   return self.parse_vma_addrow()
        if t == "ADD_COL":   return self.parse_vma_addcol()
        if t == "PRINT_VMA": return self.parse_vma_print()

        # Pointer
        if t == "POINT":     return self.parse_pointer_decl()

        # Generic identifier-led statements
        if t == "IDENT":     return self.parse_ident_stmt()

        # skip stray semicolons
        if t == "SEMICOLON":
            self.advance()
            return None

        raise ParseError(f"Line {tok.line}: unexpected statement token {tok.value!r}")

    # ────── device parsing ────────────────────

    def parse_device_stmt(self):
        """DEVICE(CPU) / DEVICE(GPU) / DEVICE(TPU) / DEVICE(NPU)"""
        line = self.peek().line
        self.expect("DEVICE")
        self.expect("LPAREN")
        tok = self.advance()
        if tok.type not in ("CPU", "GPU", "TPU", "NPU", "IDENT"):
            raise ParseError(
                f"Line {tok.line}: expected a device (CPU/GPU/TPU/NPU), got {tok.value!r}"
            )
        device_name = tok.value.upper()
        self.expect("RPAREN")
        return DeviceStmt(device_name, line)

    # ────── tensor parsing ────────────────────

    def parse_tensor_decl(self):
        """
        Syntax (1-D):  tensor(name) = <1, 2, 3>
        Syntax (2-D):  tensor(name) = <1 2 3, 4 5 6>
                       rows separated by comma, elements by space / comma-within-row
        We reuse the VMA row-grouping logic: a bare COMMA at the top level
        separates rows; elements within a row are parsed until the next top-level comma
        or RANGLE.
        """
        line = self.peek().line
        self.expect("TENSOR")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("ASSIGN")
        self.expect("LANGLE")

        rows = []
        row  = []
        while self.peek().type != "RANGLE":
            if self.peek().type == "SEMICOLON":
                # semicolon = row separator for 2-D tensors
                self.advance()
                if row:
                    rows.append(row)
                    row = []
            elif self.peek().type == "COMMA":
                # comma = element separator within a row
                self.advance()
            else:
                row.append(self.parse_additive())
        if row:
            rows.append(row)
        self.expect("RANGLE")

        node = TensorDecl(name, rows, line)
        node.is_2d = len(rows) > 1
        return node

    def parse_tensor_print(self):
        line = self.peek().line
        self.expect("PRINT_TENSOR")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        return TensorPrint(name, line)

    def parse_tensor_op_stmt(self, op):
        """
        Result stored into a named variable:
          result = MATMUL(a, b)
          result = TRANSPOSE(a)
          result = DOT_PRODUCT(a, b)
          result = RESHAPE(a, rows, cols)

        Called when the *keyword* leads the statement.
        """
        line = self.peek().line
        self.advance()   # consume keyword
        self.expect("LPAREN")
        operands = []
        while self.peek().type != "RPAREN":
            operands.append(self.parse_expr())
            if self.peek().type == "COMMA":
                self.advance()
        self.expect("RPAREN")

        tensor_op = TensorOp(op, operands)
        return AssignStmt("_result", tensor_op, line)

    # ────── end tensor parsing ────────────────

    def parse_print(self):
        line = self.peek().line
        self.expect("SCREEN")
        self.expect("DOT")
        self.advance()   # PRINT keyword
        self.expect("LPAREN")
        expr = None
        if self.peek().type != "RPAREN":
            expr = self.parse_expr()
        self.expect("RPAREN")
        return PrintStmt(expr, line)

    def parse_var_decl(self):
        line = self.peek().line
        self.expect("VAR")
        self.expect("DOT")
        var_type = self.advance().value.upper()
        self.expect("DOT")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        return VarDecl(var_type, name, line)

    def parse_scanln(self):
        line = self.peek().line
        self.expect("SCANLN")
        self.expect("DOT")
        var_type = self.advance().value.upper()
        var_name = None
        if self.peek().type == "LPAREN":
            self.advance()
            var_name = self.expect("IDENT").value
            self.expect("RPAREN")
        return ScanlnStmt(var_type, var_name, line)

    def parse_for(self):
        line = self.peek().line
        self.expect("FOR")
        self.expect("LPAREN")
        init_name = self.expect("IDENT").value
        self.expect("ASSIGN")
        init_val  = self.parse_expr()
        self.expect("COMMA")
        step_expr = self.parse_expr()
        self.expect("COMMA")
        cond      = self.parse_expr()
        self.expect("COMMA")
        self.expect("IDENT")
        post_op   = self.advance()
        self.expect("RPAREN")
        body = self.parse_body_or_block()
        init = AssignStmt(init_name, init_val, line)
        return ForStmt(init, step_expr, cond, post_op.type, body, line)

    def parse_while(self):
        line = self.peek().line
        self.expect("WHILE")
        self.expect("LPAREN")
        cond = self.parse_expr()
        self.expect("RPAREN")
        body = self.parse_body_or_block()
        return WhileStmt(cond, body, line)

    def parse_if(self):
        line = self.peek().line
        self.expect("IF")
        self.expect("LPAREN")
        cond = self.parse_expr()
        self.expect("RPAREN")
        then_block = self.parse_brace_block()
        elseifs    = []
        else_block = None
        while self.peek().type == "ELSEIF":
            self.advance()
            self.expect("LPAREN")
            ec = self.parse_expr()
            self.expect("RPAREN")
            eb = self.parse_brace_block()
            elseifs.append((ec, eb))
        if self.peek().type == "ELSE":
            self.advance()
            else_block = self.parse_brace_block()
        return IfStmt(cond, then_block, elseifs, else_block, line)

    def parse_return(self):
        line = self.peek().line
        self.expect("RETURN")
        expr = None
        if self.peek().type not in ("RBRACE", "SEMICOLON", "NEWLINE", "EOF"):
            expr = self.parse_expr()
        return ReturnStmt(expr, line)

    def parse_try(self):
        line = self.peek().line
        self.expect("TRY")
        try_block    = self.parse_brace_block()
        except_var   = None
        except_block = None
        finally_block = None
        if self.peek().type == "EXCEPT":
            self.advance()
            if self.peek().type == "LPAREN":
                self.advance()
                except_var = self.expect("IDENT").value
                self.expect("RPAREN")
            except_block = self.parse_brace_block()
        if self.peek().type == "FINALLY":
            self.advance()
            finally_block = self.parse_brace_block()
        return TryStmt(try_block, except_var, except_block, finally_block, line)

    def parse_raise(self):
        line = self.peek().line
        self.expect("RAISE")
        expr = self.parse_expr()
        return RaiseStmt(expr, line)

    def parse_array_decl(self):
        line = self.peek().line
        name = self.advance().value
        self.expect("ASSIGN")
        self.expect("LANGLE")
        elements = []
        while self.peek().type != "RANGLE":
            elements.append(self.parse_expr())
            if self.peek().type == "COMMA":
                self.advance()
        self.expect("RANGLE")
        return ArrayDecl(name, elements, line)

    def parse_vma_create(self):
        line = self.peek().line
        self.expect("CREATE")
        self.expect("DOT")
        self.expect("VMA")
        self.expect("LPAREN")
        var = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("ASSIGN")
        self.advance()   # dimension token e.g. 2D
        self.expect("LANGLE")
        rows = []
        row  = []
        while self.peek().type != "RANGLE":
            if self.peek().type == "COMMA":
                self.advance()
                if row:
                    rows.append(row)
                    row = []
            else:
                row.append(self.parse_expr())
        if row:
            rows.append(row)
        self.expect("RANGLE")
        return VmaCreate(var, rows, line)

    def parse_vma_change(self):
        line = self.peek().line
        self.expect("CHANGE")
        self.expect("DOT")
        self.expect("VMA")
        self.expect("LPAREN")
        var = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("LBRACKET")
        r = self.parse_expr()
        self.expect("RBRACKET")
        self.expect("COMMA")
        self.expect("LBRACKET")
        c = self.parse_expr()
        self.expect("RBRACKET")
        self.expect("ASSIGN")
        val = self.parse_expr()
        return VmaChange(var, r, c, val, line)

    def parse_vma_addrow(self):
        line = self.peek().line
        self.expect("ADD_ROW")
        self.expect("VMA")
        self.expect("LPAREN")
        var = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("LANGLE")
        elements = []
        while self.peek().type != "RANGLE":
            elements.append(self.parse_expr())
            if self.peek().type == "COMMA":
                self.advance()
        self.expect("RANGLE")
        return VmaAddRow(var, elements, line)

    def parse_vma_addcol(self):
        line = self.peek().line
        self.expect("ADD_COL")
        self.expect("VMA")
        self.expect("LPAREN")
        var = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("LANGLE")
        elements = []
        while self.peek().type != "RANGLE":
            elements.append(self.parse_expr())
            if self.peek().type == "COMMA":
                self.advance()
        self.expect("RANGLE")
        return VmaAddCol(var, elements, line)

    def parse_vma_print(self):
        line = self.peek().line
        self.expect("PRINT_VMA")
        self.expect("VMA")
        self.expect("LPAREN")
        var = self.expect("IDENT").value
        self.expect("RPAREN")
        return VmaPrint(var, line)

    def parse_pointer_decl(self):
        line = self.peek().line
        self.expect("POINT")
        self.expect("LPAREN")
        ptr = self.expect("IDENT").value
        self.expect("RPAREN")
        self.expect("ASSIGN")
        self.expect("AMP")
        target = self.expect("IDENT").value
        return PointerDecl(ptr, target, line)

    def parse_ident_stmt(self):
        line = self.peek().line
        name = self.advance().value

        # Array index assign:  arr 1 = 3
        if self.peek().type == "INT_LIT":
            idx = int(self.advance().value)
            self.expect("ASSIGN")
            val = self.parse_expr()
            return ArrayAssign(name, Literal(idx), val, line)

        # Void / function call:  (name)
        if self.peek().type == "LPAREN":
            self.advance()
            self.expect("RPAREN")
            return CallStmt(name, [], line)

        # Assignment:  x = expr  (also catches  x = MATMUL(...) etc.)
        if self.peek().type == "ASSIGN":
            self.advance()
            # Tensor operation on right-hand side?
            rhs_type = self.peek().type
            if rhs_type in ("MATMUL", "TRANSPOSE", "DOT_PRODUCT", "RESHAPE"):
                val = self.parse_tensor_op_expr()
            elif rhs_type == "PREDICT":
                val = self.parse_predict_expr()
            elif rhs_type == "LOAD_MODEL":
                val = self.parse_load_model_expr()
            elif rhs_type in ("LOSS_MSE", "LOSS_CE"):
                val = self.parse_loss_expr()
            else:
                val = self.parse_expr()
            return AssignStmt(name, val, line)

        # compound assign  x+1
        if self.peek().type in ("PLUS", "MINUS", "STAR", "SLASH"):
            op  = self.advance().type
            rhs = self.parse_expr()
            return AssignStmt(name, BinOp(op, Identifier(name), rhs), line)

        return AssignStmt(name, self.parse_expr(), line)

    def parse_tensor_op_expr(self):
        """Parse a tensor operation that appears on the RHS of an assignment."""
        op_tok = self.advance()   # consume MATMUL / TRANSPOSE / DOT_PRODUCT / RESHAPE
        op     = op_tok.type
        self.expect("LPAREN")
        operands = []
        while self.peek().type != "RPAREN":
            operands.append(self.parse_expr())
            if self.peek().type == "COMMA":
                self.advance()
        self.expect("RPAREN")
        return TensorOp(op, operands)

    # ── training loop parsers ────────────────

    def parse_network_decl(self):
        """NETWORK(name)"""
        line = self.peek().line
        self.expect("NETWORK")
        self.expect("LPAREN")
        name = self.expect("IDENT").value
        self.expect("RPAREN")
        return NetworkDecl(name, line)

    def parse_layer_stmt(self):
        """
        LAYER(net, DENSE, in_size, out_size)
        LAYER(net, RELU)
        LAYER(net, SIGMOID)
        LAYER(net, TANH)
        LAYER(net, SOFTMAX)
        """
        line = self.peek().line
        self.expect("LAYER")
        self.expect("LPAREN")
        net_name   = self.expect("IDENT").value
        self.expect("COMMA")
        layer_type = self.advance().value.upper()   # DENSE/RELU/SIGMOID/TANH/SOFTMAX
        in_size    = None
        out_size   = None
        if self.peek().type == "COMMA":
            self.advance()
            in_size = self.parse_expr()
            self.expect("COMMA")
            out_size = self.parse_expr()
        self.expect("RPAREN")
        return LayerStmt(net_name, layer_type, in_size, out_size, line)

    def parse_train_stmt(self):
        """TRAIN(net, X, Y, lr, epochs)"""
        line = self.peek().line
        self.expect("TRAIN")
        self.expect("LPAREN")
        net_name = self.expect("IDENT").value
        self.expect("COMMA")
        x_name   = self.expect("IDENT").value
        self.expect("COMMA")
        y_name   = self.expect("IDENT").value
        self.expect("COMMA")
        lr_expr      = self.parse_expr()
        self.expect("COMMA")
        epochs_expr  = self.parse_expr()
        self.expect("RPAREN")
        return TrainStmt(net_name, x_name, y_name, lr_expr, epochs_expr, line)

    def parse_predict_expr(self):
        """PREDICT(net, X)  — used on the RHS of an assignment."""
        self.expect("PREDICT")
        self.expect("LPAREN")
        net_name = self.expect("IDENT").value
        self.expect("COMMA")
        x_name   = self.expect("IDENT").value
        self.expect("RPAREN")
        return _PredictExpr(net_name, x_name)

    def parse_loss_expr(self):
        """LOSS_MSE(pred, target) or LOSS_CE(pred, target)"""
        loss_type = self.advance().type   # LOSS_MSE or LOSS_CE
        self.expect("LPAREN")
        pred   = self.parse_expr()
        self.expect("COMMA")
        target = self.parse_expr()
        self.expect("RPAREN")
        kind = "MSE" if loss_type == "LOSS_MSE" else "CE"
        return LossExpr(kind, pred, target)

    def parse_load_model_expr(self):
        """LOAD_MODEL("path")  — on RHS: net = LOAD_MODEL("path")"""
        self.expect("LOAD_MODEL")
        self.expect("LPAREN")
        path_expr = self.parse_expr()
        self.expect("RPAREN")
        return _LoadModelExpr(path_expr)

    def parse_save_model(self):
        """SAVE_MODEL(net, "path")"""
        line = self.peek().line
        self.expect("SAVE_MODEL")
        self.expect("LPAREN")
        net_name   = self.expect("IDENT").value
        self.expect("COMMA")
        path_expr  = self.parse_expr()
        self.expect("RPAREN")
        return SaveModelStmt(net_name, path_expr, line)

    def parse_load_model(self):
        """LOAD_MODEL("path")  — as standalone statement (result stored to _net)"""
        line = self.peek().line
        self.expect("LOAD_MODEL")
        self.expect("LPAREN")
        path_expr = self.parse_expr()
        self.expect("RPAREN")
        return LoadModelStmt("_net", path_expr, line)

    # ── helpers ──────────────────────────────
    def parse_body_or_block(self):
        if self.peek().type == "LBRACE":
            return self.parse_brace_block()
        s = self.parse_stmt()
        return Block([s] if s else [])

    def parse_brace_block(self):
        self.expect("LBRACE")
        b = self.parse_block()
        self.expect("RBRACE")
        return b

    # ── expressions ──────────────────────────
    def parse_expr(self):
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek().type in ("LANGLE", "RANGLE", "LTE", "GTE", "EQUALS", "NEQ", "ASSIGN"):
            op    = self.advance().type
            right = self.parse_additive()
            left  = BinOp(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek().type in ("PLUS", "MINUS"):
            op    = self.advance().type
            right = self.parse_multiplicative()
            left  = BinOp(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek().type in ("STAR", "SLASH", "PERCENT"):
            op    = self.advance().type
            right = self.parse_unary()
            left  = BinOp(op, left, right)
        return left

    def parse_unary(self):
        if self.peek().type == "MINUS":
            self.advance()
            return UnaryOp("MINUS", self.parse_primary())
        return self.parse_primary()

    MATH_FUNCS = {"ADD", "SUB", "DIV", "MUL", "RELU", "GELU", "SIGMOID", "TANN"}

    def parse_primary(self):
        tok = self.peek()

        if tok.type == "FLOAT_LIT":
            self.advance(); return Literal(float(tok.value))
        if tok.type == "INT_LIT":
            self.advance(); return Literal(int(tok.value))
        if tok.type == "STRING_LIT":
            self.advance(); return Literal(tok.value[1:-1])
        if tok.type == "BOOL_LIT":
            self.advance(); return Literal(tok.value == "TRUE")
        if tok.type == "NULL":
            self.advance(); return Literal(None)

        if tok.type in self.MATH_FUNCS:
            func = self.advance().value.upper()
            self.expect("LPAREN")
            args = []
            while self.peek().type != "RPAREN":
                args.append(self.parse_expr())
                if self.peek().type == "COMMA":
                    self.advance()
            self.expect("RPAREN")
            return MathFunc(func, args)

        if tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expr()
            self.expect("RPAREN")
            return expr

        # Tensor ops can also appear inside expressions
        if tok.type in ("MATMUL", "TRANSPOSE", "DOT_PRODUCT", "RESHAPE"):
            return self.parse_tensor_op_expr()

        # Training ops that return a value
        if tok.type in ("LOSS_MSE", "LOSS_CE"):
            return self.parse_loss_expr()

        if tok.type == "PREDICT":
            return self.parse_predict_expr()

        if tok.type == "IDENT":
            name = self.advance().value
            if self.peek().type == "LBRACKET":
                self.advance()
                idx = self.parse_expr()
                self.expect("RBRACKET")
                return ArrayIndex(name, idx)
            if self.peek().type == "LPAREN":
                self.advance()
                args = []
                while self.peek().type != "RPAREN":
                    args.append(self.parse_expr())
                    if self.peek().type == "COMMA":
                        self.advance()
                self.expect("RPAREN")
                return FuncCall(name, args)
            return Identifier(name)

        raise ParseError(f"Line {tok.line}: unexpected token in expression: {tok.value!r}")


# ─────────────────────────────────────────────
#  TENSOR RUNTIME HELPERS
# ─────────────────────────────────────────────

class NinjaTensor:
    """
    Lightweight pure-Python tensor backed by a flat list + shape.
    Supports 1-D (shape=(n,)) and 2-D (shape=(rows, cols)).

    matmul()/transpose() will transparently route through the active
    DEVICE backend (torch on GPU/TPU/NPU) when one is engaged; otherwise
    they use the original pure-Python loops below.
    """
    def __init__(self, data, shape):
        self.data  = list(data)   # flat list of numbers
        self.shape = tuple(shape)

    # ── factory helpers ──────────────────────
    @classmethod
    def from_flat(cls, flat):
        return cls(flat, (len(flat),))

    @classmethod
    def from_rows(cls, rows):
        r = len(rows)
        c = len(rows[0])
        flat = [v for row in rows for v in row]
        return cls(flat, (r, c))

    # ── element access ───────────────────────
    def _idx(self, *indices):
        if len(self.shape) == 1:
            return indices[0]
        return indices[0] * self.shape[1] + indices[1]

    def get(self, *indices):
        return self.data[self._idx(*indices)]

    # ── operations ───────────────────────────
    def dot(self, other):
        """1-D dot product → scalar."""
        if self.shape != other.shape or len(self.shape) != 1:
            raise NinjaError(
                f"DOT_PRODUCT requires two 1-D tensors of equal length, "
                f"got {self.shape} and {other.shape}"
            )
        if DEVICE.usable():
            ta = DEVICE.to_tensor(self.data, self.shape)
            tb = DEVICE.to_tensor(other.data, other.shape)
            return float(DEVICE.torch.dot(ta, tb).item())
        return sum(a * b for a, b in zip(self.data, other.data))

    def matmul(self, other):
        """2-D matrix multiplication → NinjaTensor."""
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise NinjaError(
                f"MATMUL requires two 2-D tensors, got {self.shape} and {other.shape}"
            )
        r1, c1 = self.shape
        r2, c2 = other.shape
        if c1 != r2:
            raise NinjaError(
                f"MATMUL shape mismatch: ({r1},{c1}) x ({r2},{c2})"
            )
        if DEVICE.usable():
            ta = DEVICE.to_tensor(self.data, self.shape)
            tb = DEVICE.to_tensor(other.data, other.shape)
            out = DEVICE.torch.matmul(ta, tb)
            return NinjaTensor(DEVICE.from_tensor(out), (r1, c2))

        result = []
        for i in range(r1):
            for j in range(c2):
                s = sum(self.get(i, k) * other.get(k, j) for k in range(c1))
                result.append(s)
        return NinjaTensor(result, (r1, c2))

    def transpose(self):
        """Transpose a 2-D tensor → NinjaTensor."""
        if len(self.shape) != 2:
            raise NinjaError(
                f"TRANSPOSE requires a 2-D tensor, got shape {self.shape}"
            )
        r, c = self.shape
        if DEVICE.usable():
            ta  = DEVICE.to_tensor(self.data, self.shape)
            out = ta.t().contiguous()
            return NinjaTensor(DEVICE.from_tensor(out), (c, r))

        result = [self.get(j, i) for i in range(c) for j in range(r)]
        return NinjaTensor(result, (c, r))

    def reshape(self, new_rows, new_cols):
        """Reshape to (new_rows, new_cols) — total elements must match."""
        total = new_rows * new_cols
        if total != len(self.data):
            raise NinjaError(
                f"RESHAPE: cannot reshape {len(self.data)} elements into "
                f"({new_rows}, {new_cols})"
            )
        return NinjaTensor(self.data[:], (new_rows, new_cols))

    # ── display ──────────────────────────────
    def __str__(self):
        def fmt(v):
            return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)

        if len(self.shape) == 1:
            return "[" + "  ".join(fmt(v) for v in self.data) + "]"

        r, c = self.shape
        rows = []
        for i in range(r):
            row = "  ".join(fmt(self.data[i * c + j]) for j in range(c))
            rows.append(f"[ {row} ]")
        return "\n".join(rows)

    def __repr__(self):
        return f"NinjaTensor(shape={self.shape}, data={self.data})"


# ─────────────────────────────────────────────
#  NEURAL NETWORK RUNTIME
# ─────────────────────────────────────────────

import random as _random
import json   as _json

def _rand_weight(fan_in):
    """Xavier-uniform initialisation."""
    limit = (6.0 / fan_in) ** 0.5
    return _random.uniform(-limit, limit)


class _DenseLayer:
    """
    Fully-connected layer with optional bias. Stores gradients for SGD.
    forward()/backward()/update() route through the active DEVICE backend
    (torch on GPU/TPU/NPU) when engaged, matching the CPU math exactly.
    """
    def __init__(self, in_size, out_size):
        self.in_size  = in_size
        self.out_size = out_size
        # weights: out_size × in_size  (row = output neuron)
        self.W = [[_rand_weight(in_size) for _ in range(in_size)]
                  for _ in range(out_size)]
        self.b = [0.0] * out_size
        # caches for backward
        self._input   = None
        self._pre_act = None
        # gradients
        self.dW = [[0.0]*in_size for _ in range(out_size)]
        self.db = [0.0] * out_size

    def _flat_W(self):
        return [w for row in self.W for w in row]

    def forward(self, x):
        """x: list of floats (in_size,)  →  list of floats (out_size,)"""
        self._input = x[:]

        if DEVICE.usable():
            tw = DEVICE.to_tensor(self._flat_W(), (self.out_size, self.in_size))
            tb = DEVICE.to_tensor(self.b, (self.out_size,))
            tx = DEVICE.to_tensor(x, (self.in_size,))
            out_t = DEVICE.torch.matmul(tw, tx) + tb
            out = DEVICE.from_tensor(out_t)
            self._pre_act = out[:]
            return out

        out = []
        for i in range(self.out_size):
            s = sum(self.W[i][j] * x[j] for j in range(self.in_size)) + self.b[i]
            out.append(s)
        self._pre_act = out[:]
        return out

    def backward(self, grad_out):
        """grad_out: (out_size,)  →  grad_in: (in_size,)"""
        if DEVICE.usable():
            tg  = DEVICE.to_tensor(grad_out, (self.out_size,))
            tx  = DEVICE.to_tensor(self._input, (self.in_size,))
            tw  = DEVICE.to_tensor(self._flat_W(), (self.out_size, self.in_size))
            dW  = DEVICE.torch.outer(tg, tx)
            grad_in = DEVICE.torch.matmul(tg, tw)
            flat_dW = DEVICE.from_tensor(dW)
            for i in range(self.out_size):
                for j in range(self.in_size):
                    self.dW[i][j] += flat_dW[i * self.in_size + j]
                self.db[i] += grad_out[i]
            return DEVICE.from_tensor(grad_in)

        grad_in = [0.0] * self.in_size
        for i in range(self.out_size):
            for j in range(self.in_size):
                self.dW[i][j] += grad_out[i] * self._input[j]
                grad_in[j]    += grad_out[i] * self.W[i][j]
            self.db[i] += grad_out[i]
        return grad_in

    def update(self, lr):
        for i in range(self.out_size):
            for j in range(self.in_size):
                self.W[i][j] -= lr * self.dW[i][j]
                self.dW[i][j] = 0.0
            self.b[i] -= lr * self.db[i]
            self.db[i] = 0.0

    def to_dict(self):
        return {"type": "dense", "in": self.in_size, "out": self.out_size,
                "W": self.W, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        layer = cls(d["in"], d["out"])
        layer.W = d["W"]
        layer.b = d["b"]
        return layer


class _ActivationLayer:
    """Stateless activation: RELU | SIGMOID | TANH | SOFTMAX."""
    def __init__(self, kind):
        self.kind   = kind.upper()
        self._input = None

    def _apply(self, x, v):
        import math as _m
        if self.kind == "RELU":    return max(0.0, v)
        if self.kind == "SIGMOID": return 1.0 / (1.0 + _m.exp(-max(-500, min(500, v))))
        if self.kind == "TANH":    return _m.tanh(v)
        return v  # SOFTMAX handled at vector level

    def forward(self, x):
        import math as _m
        self._input = x[:]
        if self.kind == "SOFTMAX":
            m = max(x)
            exps = [_m.exp(v - m) for v in x]
            s    = sum(exps)
            out  = [e / s for e in exps]
        else:
            out = [self._apply(x, v) for v in x]
        self._output = out[:]
        return out

    def backward(self, grad_out):
        import math as _m
        if self.kind == "RELU":
            return [g if self._input[i] > 0 else 0.0
                    for i, g in enumerate(grad_out)]
        if self.kind == "SIGMOID":
            return [g * self._output[i] * (1 - self._output[i])
                    for i, g in enumerate(grad_out)]
        if self.kind == "TANH":
            return [g * (1 - self._output[i] ** 2)
                    for i, g in enumerate(grad_out)]
        if self.kind == "SOFTMAX":
            # simple element-wise passthrough (used with CE loss whose gradient
            # already incorporates the softmax derivative)
            return grad_out
        return grad_out

    def update(self, lr):
        pass   # no learnable parameters

    def to_dict(self):
        return {"type": "activation", "kind": self.kind}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"])


class NinjaNetwork:
    """
    A sequential neural network.
    Layers are added with LAYER(); training runs with TRAIN().
    """
    def __init__(self, name):
        self.name   = name
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    # ── forward ──────────────────────────────
    def forward(self, x):
        """x: list of floats → list of floats (output of last layer)."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ── loss + gradients ─────────────────────
    @staticmethod
    def loss_mse(pred, target):
        n = len(pred)
        return sum((p - t) ** 2 for p, t in zip(pred, target)) / n

    @staticmethod
    def grad_mse(pred, target):
        n = len(pred)
        return [2 * (p - t) / n for p, t in zip(pred, target)]

    @staticmethod
    def loss_ce(pred, target):
        """Cross-entropy (expects softmax output and one-hot / class-index target)."""
        import math as _m
        eps = 1e-12
        if isinstance(target, (int, float)):
            # class index
            return -_m.log(max(pred[int(target)], eps))
        # one-hot vector
        return -sum(t * _m.log(max(p, eps)) for p, t in zip(pred, target))

    @staticmethod
    def grad_ce_softmax(pred, target):
        """Combined softmax + CE gradient (pred - target)."""
        if isinstance(target, (int, float)):
            one_hot = [0.0] * len(pred)
            one_hot[int(target)] = 1.0
            target = one_hot
        return [p - t for p, t in zip(pred, target)]

    # ── backward + update ────────────────────
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    # ── serialise ────────────────────────────
    def to_dict(self):
        return {"name": self.name,
                "layers": [l.to_dict() for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        net = cls(d["name"])
        for ld in d["layers"]:
            if ld["type"] == "dense":
                net.layers.append(_DenseLayer.from_dict(ld))
            else:
                net.layers.append(_ActivationLayer.from_dict(ld))
        return net


# ─────────────────────────────────────────────
#  RUNTIME
# ─────────────────────────────────────────────

class NinjaReturn(Exception):
    def __init__(self, value):
        self.value = value

class NinjaRaise(Exception):
    def __init__(self, value):
        self.value = value

class NinjaError(Exception):
    pass


class Environment:
    def __init__(self, parent=None):
        self.vars   = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NinjaError(f"Undefined variable '{name}'")

    def set(self, name, value):
        if name in self.vars:
            self.vars[name] = value
            return
        if self.parent and self.parent.has(name):
            self.parent.set(name, value)
            return
        self.vars[name] = value

    def has(self, name):
        if name in self.vars:
            return True
        if self.parent:
            return self.parent.has(name)
        return False

    def define(self, name, value=None):
        self.vars[name] = value


class NinjaClass:
    def __init__(self, name, parent_class, body, env):
        self.name         = name
        self.parent_class = parent_class
        self.body         = body
        self.env          = env

class NinjaInstance:
    def __init__(self, klass):
        self.klass  = klass
        self.fields = {}

    def get(self, name):
        if name in self.fields:
            return self.fields[name]
        raise NinjaError(f"No attribute '{name}' on {self.klass.name}")

    def set(self, name, value):
        self.fields[name] = value


class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self.voids      = {}
        self.classes    = {}
        self.pointers   = {}
        self.vmas       = {}
        self.tensors    = {}   # name -> NinjaTensor
        self.networks   = {}   # name -> NinjaNetwork

    # ── run ──────────────────────────────────
    def run(self, program: Program):
        for v in program.voids:
            if isinstance(v, VoidDef):
                self.voids[v.name] = v
            elif isinstance(v, ClassDef):
                self.classes[v.name] = v
            elif isinstance(v, DeviceStmt):
                DEVICE.set(v.device_name)
        if program.main is None:
            raise NinjaError("No MAIN block found — program cannot execute.")
        self.exec_block(program.main.body, self.global_env)

    # ── imports ───────────────────────────────
    def load_imports(self, imports):
        PACKAGE_DIR = os.path.join(os.path.expanduser("~"), ".nforce", "packages")

        for imp in imports:
            package_name = imp.name
            package_path = os.path.join(PACKAGE_DIR, package_name)

            if not os.path.isdir(package_path):
                raise NinjaError(f"Package '{package_name}' is not installed.")

            for root, dirs, files in os.walk(package_path):
                for file in files:
                    if not file.endswith(".ninja"):
                        continue
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tokens = tokenize(source)
                    parser = Parser(tokens)
                    pkg_prog, _ = parser.parse()
                    for item in pkg_prog.voids:
                        if isinstance(item, VoidDef):
                            self.voids[item.name] = item
                        elif isinstance(item, ClassDef):
                            self.classes[item.name] = item

    # ── block / stmt ─────────────────────────
    def exec_block(self, block: Block, env: Environment):
        for stmt in block.stmts:
            self.exec_stmt(stmt, env)

    def exec_stmt(self, stmt, env):
        if isinstance(stmt, PrintStmt):
            val = self.eval_expr(stmt.expr, env) if stmt.expr is not None else ""
            print(self.to_str(val))

        elif isinstance(stmt, VarDecl):
            env.define(stmt.var_name, None)

        elif isinstance(stmt, ScanlnStmt):
            raw = input()
            val = self.coerce(raw, stmt.var_type)
            if stmt.var_name:
                env.set(stmt.var_name, val)
            else:
                env.define("_input", val)

        elif isinstance(stmt, AssignStmt):
            val = self.eval_expr(stmt.value, env)
            # If the RHS produced a NinjaTensor, store it in the tensor registry too
            if isinstance(val, NinjaTensor):
                self.tensors[stmt.target] = val
            env.set(stmt.target, val)

        elif isinstance(stmt, CallStmt):
            self.call_void(stmt.name, [], env)

        elif isinstance(stmt, ForStmt):
            self.exec_for(stmt, env)

        elif isinstance(stmt, WhileStmt):
            self.exec_while(stmt, env)

        elif isinstance(stmt, IfStmt):
            self.exec_if(stmt, env)

        elif isinstance(stmt, ReturnStmt):
            val = self.eval_expr(stmt.expr, env) if stmt.expr else None
            raise NinjaReturn(val)

        elif isinstance(stmt, TryStmt):
            self.exec_try(stmt, env)

        elif isinstance(stmt, RaiseStmt):
            val = self.eval_expr(stmt.expr, env)
            raise NinjaRaise(val)

        elif isinstance(stmt, VoidDef):
            self.voids[stmt.name] = stmt

        elif isinstance(stmt, ClassDef):
            self.classes[stmt.name] = stmt

        elif isinstance(stmt, ArrayDecl):
            elements = [self.eval_expr(e, env) for e in stmt.elements]
            env.define(stmt.name, elements)

        elif isinstance(stmt, ArrayAssign):
            arr = env.get(stmt.name)
            idx = int(self.eval_expr(stmt.index, env))
            val = self.eval_expr(stmt.value, env)
            arr[idx] = val

        # ── device statement ──────────────────
        elif isinstance(stmt, DeviceStmt):
            DEVICE.set(stmt.device_name)

        # ── tensor statements ─────────────────
        elif isinstance(stmt, TensorDecl):
            tensor = self._build_tensor(stmt, env)
            self.tensors[stmt.name] = tensor
            env.define(stmt.name, tensor)

        elif isinstance(stmt, TensorPrint):
            t = self._resolve_tensor(stmt.name, env)
            print(str(t))

        # ── VMA statements ────────────────────
        elif isinstance(stmt, VmaCreate):
            matrix = [[self.eval_expr(c, env) for c in row] for row in stmt.rows]
            self.vmas[stmt.var] = matrix

        elif isinstance(stmt, VmaChange):
            r   = int(self.eval_expr(stmt.row, env)) - 1
            c   = int(self.eval_expr(stmt.col, env)) - 1
            val = self.eval_expr(stmt.value, env)
            self.vmas[stmt.var][r][c] = val

        elif isinstance(stmt, VmaAddRow):
            row = [self.eval_expr(e, env) for e in stmt.elements]
            self.vmas[stmt.var].append(row)

        elif isinstance(stmt, VmaAddCol):
            col = [self.eval_expr(e, env) for e in stmt.elements]
            for i, row in enumerate(self.vmas[stmt.var]):
                row.append(col[i] if i < len(col) else None)

        elif isinstance(stmt, VmaPrint):
            m = self.vmas.get(stmt.var, [])
            for row in m:
                print("  ".join(str(v) for v in row))

        elif isinstance(stmt, PointerDecl):
            self.pointers[stmt.ptr_name] = stmt.target
            env.define(stmt.ptr_name, env.get(stmt.target))

        elif isinstance(stmt, PointerAssign):
            val = self.eval_expr(stmt.value, env)
            env.set(stmt.ptr_name, val)
            if stmt.ptr_name in self.pointers:
                env.set(self.pointers[stmt.ptr_name], val)

        elif isinstance(stmt, ImportStmt):
            self.do_import(stmt.name)

        # ── training loop statements ──────────
        elif isinstance(stmt, NetworkDecl):
            net = NinjaNetwork(stmt.name)
            self.networks[stmt.name] = net
            env.define(stmt.name, net)

        elif isinstance(stmt, LayerStmt):
            net = self._resolve_network(stmt.net_name, env)
            lt  = stmt.layer_type
            if lt == "DENSE":
                in_s  = int(self._to_num(self.eval_expr(stmt.in_size,  env)))
                out_s = int(self._to_num(self.eval_expr(stmt.out_size, env)))
                net.add_layer(_DenseLayer(in_s, out_s))
            elif lt in ("RELU", "SIGMOID", "TANH", "SOFTMAX"):
                net.add_layer(_ActivationLayer(lt))
            else:
                raise NinjaError(f"Unknown layer type '{lt}'. Use DENSE/RELU/SIGMOID/TANH/SOFTMAX.")

        elif isinstance(stmt, TrainStmt):
            self._exec_train(stmt, env)

        elif isinstance(stmt, SaveModelStmt):
            net  = self._resolve_network(stmt.net_name, env)
            path = str(self.eval_expr(stmt.filepath_expr, env))
            with open(path, "w") as _f:
                _json.dump(net.to_dict(), _f, indent=2)
            print(f"[NINJA] Model '{net.name}' saved to '{path}'")

        elif isinstance(stmt, LoadModelStmt):
            path = str(self.eval_expr(stmt.filepath_expr, env))
            with open(path) as _f:
                d = _json.load(_f)
            net = NinjaNetwork.from_dict(d)
            self.networks[stmt.net_name] = net
            env.define(stmt.net_name, net)

    # ── tensor helpers ───────────────────────

    def _build_tensor(self, node: TensorDecl, env: Environment) -> NinjaTensor:
        """Evaluate a TensorDecl's element expressions and build a NinjaTensor."""
        if node.is_2d:
            rows = [[self._to_num(self.eval_expr(e, env)) for e in row]
                    for row in node.elements]
            return NinjaTensor.from_rows(rows)
        else:
            flat_exprs = node.elements[0] if node.elements else []
            flat = [self._to_num(self.eval_expr(e, env)) for e in flat_exprs]
            return NinjaTensor.from_flat(flat)

    def _resolve_tensor(self, name, env) -> NinjaTensor:
        """Get a NinjaTensor by name — checks env first, then tensor registry."""
        if env.has(name):
            val = env.get(name)
            if isinstance(val, NinjaTensor):
                return val
        if name in self.tensors:
            return self.tensors[name]
        raise NinjaError(f"No tensor named '{name}'")

    def _to_num(self, val):
        if isinstance(val, (int, float)):
            return val
        try:
            return float(val)
        except (TypeError, ValueError):
            raise NinjaError(f"Tensor element must be numeric, got {val!r}")

    def _eval_tensor_op(self, node: TensorOp, env) -> NinjaTensor:
        op = node.op
        ops = node.operands

        if op == "DOT_PRODUCT":
            a = self._resolve_tensor(self._ident_name(ops[0]), env)
            b = self._resolve_tensor(self._ident_name(ops[1]), env)
            result = a.dot(b)           # returns a scalar
            return result               # scalar; caller stores it

        if op == "MATMUL":
            a = self._resolve_tensor(self._ident_name(ops[0]), env)
            b = self._resolve_tensor(self._ident_name(ops[1]), env)
            return a.matmul(b)

        if op == "TRANSPOSE":
            a = self._resolve_tensor(self._ident_name(ops[0]), env)
            return a.transpose()

        if op == "RESHAPE":
            a   = self._resolve_tensor(self._ident_name(ops[0]), env)
            r   = int(self._to_num(self.eval_expr(ops[1], env)))
            c   = int(self._to_num(self.eval_expr(ops[2], env)))
            return a.reshape(r, c)

        raise NinjaError(f"Unknown tensor operation: {op}")

    def _ident_name(self, node) -> str:
        if isinstance(node, Identifier):
            return node.name
        raise NinjaError(f"Expected tensor name (identifier), got {type(node).__name__}")

    # ── training helpers ─────────────────────

    def _resolve_network(self, name, env) -> "NinjaNetwork":
        if env.has(name):
            val = env.get(name)
            if isinstance(val, NinjaNetwork):
                return val
        if name in self.networks:
            return self.networks[name]
        raise NinjaError(f"No network named '{name}'. Did you forget NETWORK({name})?")

    def _resolve_input(self, name, env):
        """Return a flat list of floats from a variable (NinjaTensor, list, or scalar)."""
        val = env.get(name)
        if isinstance(val, NinjaTensor):
            return [float(v) for v in val.data]
        if isinstance(val, list):
            return [float(v) for v in val]
        return [float(val)]

    def _resolve_targets(self, name, env):
        """Return a list of target vectors (list of list of float) for a dataset."""
        val = env.get(name)
        if isinstance(val, NinjaTensor):
            if len(val.shape) == 2:
                r, c = val.shape
                return [[float(val.data[i*c+j]) for j in range(c)] for i in range(r)]
            return [[float(v) for v in val.data]]
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                return [[float(v) for v in row] for row in val]
            return [[float(v) for v in val]]
        return [[float(val)]]

    def _resolve_dataset(self, name, env):
        """Return list of input vectors from a NinjaTensor or list."""
        val = env.get(name)
        if isinstance(val, NinjaTensor):
            if len(val.shape) == 2:
                r, c = val.shape
                return [[float(val.data[i*c+j]) for j in range(c)] for i in range(r)]
            return [[float(v) for v in val.data]]
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                return [[float(v) for v in row] for row in val]
            return [[float(v) for v in val]]
        return [[float(val)]]

    def _exec_train(self, stmt: "TrainStmt", env):
        net    = self._resolve_network(stmt.net_name, env)
        X      = self._resolve_dataset(stmt.x_name,  env)
        Y      = self._resolve_targets(stmt.y_name,  env)
        lr     = float(self._to_num(self.eval_expr(stmt.lr_expr,     env)))
        epochs = int(self._to_num(self.eval_expr(stmt.epochs_expr, env)))

        if len(X) != len(Y):
            raise NinjaError(
                f"TRAIN: X has {len(X)} samples but Y has {len(Y)} targets.")

        use_ce = (net.layers and
                  isinstance(net.layers[-1], _ActivationLayer) and
                  net.layers[-1].kind == "SOFTMAX")

        n_samples = len(X)
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x_i, y_i in zip(X, Y):
                pred = net.forward(x_i)

                if use_ce:
                    total_loss += NinjaNetwork.loss_ce(pred, y_i)
                    grad = NinjaNetwork.grad_ce_softmax(pred, y_i)
                else:
                    total_loss += NinjaNetwork.loss_mse(pred, y_i)
                    grad = NinjaNetwork.grad_mse(pred, y_i)

                net.backward(grad)
                net.update(lr)

            avg_loss = total_loss / n_samples
            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"[NINJA TRAIN] epoch {epoch:>4}/{epochs}  loss={avg_loss:.6f}"
                      f"  device={DEVICE.active}")

    # ── control flow ─────────────────────────
    def exec_for(self, stmt: ForStmt, env: Environment):
        local = Environment(env)
        self.exec_stmt(stmt.init, local)
        var_name = stmt.init.target
        while self.is_truthy(self.eval_expr(stmt.cond, local)):
            try:
                self.exec_block(stmt.body, Environment(local))
            except NinjaReturn:
                raise
            cur = local.get(var_name)
            if stmt.post == "INCREMENT":
                local.set(var_name, cur + 1)
            else:
                local.set(var_name, cur - 1)

    def exec_while(self, stmt: WhileStmt, env: Environment):
        while self.is_truthy(self.eval_expr(stmt.cond, env)):
            try:
                self.exec_block(stmt.body, Environment(env))
            except NinjaReturn:
                raise

    def exec_if(self, stmt: IfStmt, env: Environment):
        if self.is_truthy(self.eval_expr(stmt.cond, env)):
            self.exec_block(stmt.then_block, Environment(env))
            return
        for cond, block in stmt.elseifs:
            if self.is_truthy(self.eval_expr(cond, env)):
                self.exec_block(block, Environment(env))
                return
        if stmt.else_block:
            self.exec_block(stmt.else_block, Environment(env))

    def exec_try(self, stmt: TryStmt, env: Environment):
        try:
            self.exec_block(stmt.try_block, Environment(env))
        except NinjaRaise as e:
            if stmt.except_block:
                exc_env = Environment(env)
                if stmt.except_var:
                    exc_env.define(stmt.except_var, e.value)
                self.exec_block(stmt.except_block, exc_env)
        except Exception as e:
            if stmt.except_block:
                exc_env = Environment(env)
                if stmt.except_var:
                    exc_env.define(stmt.except_var, str(e))
                self.exec_block(stmt.except_block, exc_env)
        finally:
            if stmt.finally_block:
                self.exec_block(stmt.finally_block, Environment(env))

    # ── void calls ───────────────────────────
    def call_void(self, name, args, caller_env):
        if name not in self.voids:
            raise NinjaError(f"Undefined void '{name}'")
        vd  = self.voids[name]
        env = Environment(self.global_env)
        for p, a in zip(vd.params, args):
            env.define(p, a)
        try:
            self.exec_block(vd.body, env)
        except NinjaReturn as r:
            return r.value
        return None

    # ── expressions ──────────────────────────
    def eval_expr(self, node, env):
        if isinstance(node, Literal):
            return node.value

        if isinstance(node, Identifier):
            return env.get(node.name)

        if isinstance(node, BinOp):
            return self.eval_binop(node, env)

        if isinstance(node, UnaryOp):
            v = self.eval_expr(node.operand, env)
            if node.op == "MINUS":
                return -v
            return not v

        if isinstance(node, FuncCall):
            args = [self.eval_expr(a, env) for a in node.args]
            return self.call_void(node.name, args, env)

        if isinstance(node, MathFunc):
            return self.eval_math(node, env)

        if isinstance(node, ArrayIndex):
            arr = env.get(node.name)
            idx = int(self.eval_expr(node.index, env))
            return arr[idx]

        if isinstance(node, TensorOp):
            return self._eval_tensor_op(node, env)

        if isinstance(node, LossExpr):
            pred   = self.eval_expr(node.pred,   env)
            target = self.eval_expr(node.target, env)
            if isinstance(pred,   NinjaTensor): pred   = pred.data
            if isinstance(target, NinjaTensor): target = target.data
            if node.loss_type == "MSE":
                return NinjaNetwork.loss_mse(pred, target)
            else:
                return NinjaNetwork.loss_ce(pred, target)

        if isinstance(node, _PredictExpr):
            net = self._resolve_network(node.net_name, env)
            x   = self._resolve_input(node.x_name, env)
            out = net.forward(x)
            return NinjaTensor.from_flat(out)

        if isinstance(node, _LoadModelExpr):
            path = str(self.eval_expr(node.path_expr, env))
            with open(path) as _f:
                d = _json.load(_f)
            net = NinjaNetwork.from_dict(d)
            return net

        raise NinjaError(f"Unknown expression node: {type(node)}")

    def eval_binop(self, node: BinOp, env):
        l  = self.eval_expr(node.left,  env)
        r  = self.eval_expr(node.right, env)
        op = node.op
        if op == "PLUS":    return l + r
        if op == "MINUS":   return l - r
        if op == "STAR":    return l * r
        if op == "SLASH":
            if r == 0:
                raise NinjaError("Division by zero")
            return l / r
        if op == "PERCENT": return l % r
        if op == "LANGLE":  return l < r
        if op == "RANGLE":  return l > r
        if op == "LTE":     return l <= r
        if op == "GTE":     return l >= r
        if op == "EQUALS":  return l == r
        if op == "ASSIGN":  return l == r
        if op == "NEQ":     return l != r
        raise NinjaError(f"Unknown operator: {op}")

    def eval_math(self, node: MathFunc, env):
        args = [self.eval_expr(a, env) for a in node.args]
        f    = node.func
        if f == "ADD":     return sum(args)
        if f == "SUB":     return args[0] - args[1]
        if f == "DIV":     return args[0] / args[1]
        if f == "MUL":     return args[0] * args[1]
        if f == "RELU":    return max(0, args[0])
        if f == "GELU":
            x = args[0]
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        if f == "SIGMOID": return 1 / (1 + math.exp(-args[0]))
        if f == "TANN":    return math.tanh(args[0])
        raise NinjaError(f"Unknown math function: {f}")

    # ── helpers ──────────────────────────────
    def is_truthy(self, val):
        if val is None:       return False
        if isinstance(val, bool):         return val
        if isinstance(val, (int, float)): return val != 0
        if isinstance(val, str):          return len(val) > 0
        return True

    def to_str(self, val):
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, NinjaTensor):
            return str(val)
        return str(val)

    def coerce(self, raw, var_type):
        try:
            if var_type == "INT":   return int(raw)
            if var_type == "FLOAT": return float(raw)
            if var_type == "NUMB":  return int(raw)
            if var_type == "BOOL":  return raw.strip().upper() == "TRUE"
        except ValueError:
            pass
        return raw

    def do_import(self, name):
        path = f"{name}.ninja"
        if not os.path.exists(path):
            path = os.path.join("src", f"{name}.ninja")
        if not os.path.exists(path):
            raise NinjaError(f"Cannot find package '{name}'")
        with open(path) as f:
            source = f.read()
        tokens  = tokenize(source)
        parser  = Parser(tokens)
        prog, _ = parser.parse()
        for v in prog.voids:
            if isinstance(v, VoidDef):
                self.voids[v.name] = v
            elif isinstance(v, ClassDef):
                self.classes[v.name] = v


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def compile_and_run(source: str):
    try:
        tokens         = tokenize(source)
        parser         = Parser(tokens)
        program, _     = parser.parse()
        interp         = Interpreter()
        interp.run(program)
    except LexerError as e:
        print(f"[NINJA LEXER ERROR] {e}",   file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        print(f"[NINJA PARSE ERROR] {e}",   file=sys.stderr)
        sys.exit(1)
    except NinjaDeviceError as e:
        print(f"[NINJA DEVICE ERROR] {e}",  file=sys.stderr)
        sys.exit(1)
    except NinjaError as e:
        print(f"[NINJA RUNTIME ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except NinjaRaise as e:
        print(f"[NINJA UNCAUGHT RAISE] {e.value}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python py_ninja.py <file.ninja>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath, "r") as f:
        source = f.read()
    compile_and_run(source)


if __name__ == "__main__":
    main()