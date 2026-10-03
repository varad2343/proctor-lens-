"""Fixture-free test runner for machines without pytest: `python tests/run.py [pattern]`.
Tests are plain `test_*` functions with bare asserts and no fixtures, so pytest runs them too."""
import importlib.util
import pathlib
import sys
import traceback

root = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(root.parent / "src"), str(root.parent)]
pat = sys.argv[1] if len(sys.argv) > 1 else ""
ok = fail = 0
for f in sorted(root.rglob("test_*.py")):
    if pat not in f.as_posix():  # forward slashes work on Windows too: tests/unit/test_clock
        continue
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        fail += 1
        print(f"IMPORT FAIL {f.relative_to(root)}\n{traceback.format_exc()}")
        continue
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)()
            ok += 1
        except Exception:
            fail += 1
            print(f"FAIL {f.relative_to(root)}::{name}\n{traceback.format_exc()}")
print(f"{ok} passed, {fail} failed")
sys.exit(1 if fail or not ok else 0)  # a pattern that matches nothing is an error, not a pass
