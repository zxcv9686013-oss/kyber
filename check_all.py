import py_compile, sys

files = [
    r"C:\Users\Admin\kyber (7)\kyber\compiler\kyber__compiler.PY",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\kyber_linker.py",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\ngpu_compiler.py",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\xasm_compiler.py",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\npu_compiler.py",
    r"C:\Users\Admin\kyber (7)\kyber\compiler\project_env.py",
    r"C:\Users\Admin\kyber (7)\kyber\nforce\nforce.py",
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f.split(chr(92))[-1]}")
    except py_compile.PyCompileError as e:
        print(f"ERROR: {f.split(chr(92))[-1]}")
        print(f"  {e}")
        all_ok = False

print()
print("ALL OK" if all_ok else "SOME FILES HAVE ERRORS")
sys.exit(0 if all_ok else 1)
