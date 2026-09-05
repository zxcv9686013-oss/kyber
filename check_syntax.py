import py_compile
files = [
    r"C:\Users\Admin\kyber (7)\kyber\compiler\kyber__compiler.PY",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\ngpu_compiler.py",
]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"ERROR: {f}\n  {e}")
