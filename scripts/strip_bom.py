"""Strip UTF-8 BOMs and normalise line endings in the working tree.

PowerShell's `Set-Content -Encoding UTF8` writes a BOM on Windows PowerShell
5.1. A BOM is legal in a Python source file but is untidy, and it breaks
byte-comparison checks that compare a file against its staged copy.
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\PLEXTEK\Desktop\Onn-Detect-")
SKIP_DIRS = {".git", ".venv", "build", "scratch", ".gradle", "node_modules"}


def main() -> int:
    changed = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in (".py", ".kt", ".kts", ".json", ".md", ".yaml", ".yml"):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if not raw.startswith(b"\xef\xbb\xbf"):
            continue
        path.write_bytes(raw[3:])
        changed.append(path.relative_to(ROOT).as_posix())
    for name in changed:
        print(f"stripped BOM: {name}")
    print(f"{len(changed)} file(s) cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
