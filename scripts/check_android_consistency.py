"""Static consistency checker for the Android + ML cross-platform contract.
Run from repo root:  python scripts/check_android_consistency.py
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "app" / "src" / "main"
JAVA = APP / "java"

errors = []
warnings = []

# ---------- collect kotlin files ----------
kt_files = {}
for p in JAVA.rglob("*.kt"):
    kt_files[p] = p.read_text(encoding="utf-8", errors="replace")

print(f"Kotlin sources found: {len(kt_files)}")

# class/interface/object/enum declarations + top-level functions per file
declared = {}   # simple-name -> path
for p, src in kt_files.items():
    m = re.findall(r"\b(?:class|interface|object|enum class)\s+(\w+)", src)
    funs = re.findall(r"^fun\s+(\w+)|^\s+fun\s+(?:\w+\.)?(\w+)\s*\(", src, re.M)
    declared[p.stem] = p
    for name in m:
        declared.setdefault(name, p)
    for a, b in funs:
        declared.setdefault(a or b, p)

# ---------- package declarations match directories ----------
for p, src in kt_files.items():
    m = re.search(r"^package\s+([\w.]+)", src, re.M)
    expected = ".".join(p.parent.relative_to(JAVA).parts)
    if not m:
        if src.strip():
            errors.append(f"{p}: no package declaration")
    elif m.group(1) != expected:
        errors.append(f"{p}: package '{m.group(1)}' != dir '{expected}'")

# ---------- internal imports resolve ----------
for p, src in kt_files.items():
    for imp in re.findall(r"^import\s+(com\.onndetect\.[\w.]+)", src, re.M):
        simple = imp.split(".")[-1]
        if simple not in declared:
            errors.append(f"{p}: import {imp} does not resolve")

# ---------- brace balance ----------
for p, src in kt_files.items():
    depth = 0
    for ch in re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|//[^\n]*|/\*.*?\*/', "", src, flags=re.S):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"{p}: unbalanced braces"); break
    else:
        if depth != 0:
            errors.append(f"{p}: unbalanced braces ({depth} open)")

# ---------- references to still-empty source files ----------
empty_kt = [p for p, s in kt_files.items() if not s.strip()]
for p, src in kt_files.items():
    if not src.strip():
        continue
    for q in empty_kt:
        # e.g. "QualityGate(" or ": QualityGate" style usages
        if re.search(rf"\b{q.stem}\s*[(<:.]", src):
            errors.append(f"{p}: references EMPTY class {q.stem}")

# ---------- manifest resources ----------
manifest = (APP / "AndroidManifest.xml").read_text(encoding="utf-8")
res_dir = APP / "res"
def res_exists(res_type: str, name: str) -> bool:
    hits = list(res_dir.rglob(f"*.{ 'xml' if res_type in ('xml','mipmap','drawable','style','string','color') else 'xml'}"))
    for h in hits:
        if h.stem == name and (
            res_type == "xml" and h.parent.name == "xml"
            or res_type == "drawable" and (h.parent.name == "drawable" or "anydpi" in h.parent.name)
            or res_type == "mipmap" and "anydpi" in h.parent.name
            or res_type in ("style",) and h.name in ("themes.xml",)
            or res_type == "string" and h.name == "strings.xml"
            or res_type == "color" and h.name == "colors.xml"
        ):
            return True
    return False

for ref_type, ref_name in re.findall(r"@(\w+)/([\w.]+)", manifest):
    if ref_type == "xml":
        if not res_exists("xml", ref_name):
            errors.append(f"Manifest resource missing: @xml/{ref_name}")
    elif ref_type == "mipmap":
        if not res_exists("mipmap", ref_name):
            errors.append(f"Manifest resource missing: @mipmap/{ref_name}")
    elif ref_type == "style":
        themes = (res_dir / "values" / "themes.xml")
        if not themes.exists() or f'"{ref_name}"' not in themes.read_text(encoding="utf-8"):
            errors.append(f"Manifest style missing: {ref_name}")

# theme referenced by app must exist
themes = (res_dir / "values" / "themes.xml")
if not themes.exists() or '"Theme.OnnDetect"' not in themes.read_text(encoding="utf-8"):
    errors.append("Missing Theme.OnnDetect")

# ---------- cross-platform feature contract ----------
py_fe = ROOT / "ml" / "src" / "meta_classifier" / "feature_extractor.py"
kt_meta = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "onndetect" / "detection" / "fusion" / "XGBoostMetaClassifier.kt"
if py_fe.exists() and kt_meta.exists():
    py_src = py_fe.read_text(encoding="utf-8")
    kt_src = kt_meta.read_text(encoding="utf-8")
    py_feats = re.search(r"FEATURE_NAMES\s*=\s*\[(.*?)\]", py_src, re.S)
    feats = re.findall(r'"(\w+)"', py_feats.group(1)) if py_feats else []
    doc_ok = all(f in kt_src for f in feats[:4])
    print(f"Feature contract: python features={feats}")
    print(f"Kotlin export documents first 4 features: {doc_ok}")
    if "\\n" in kt_src:
        errors.append("Kotlin meta classifier contains literal \\n escapes!")

# ---------- assets sanity ----------
models_dir = APP / "assets" / "models"
const_dir = APP / "assets" / "constants"
zero_assets = [f.name for f in list(models_dir.glob("*")) + list(const_dir.glob("*")) if f.stat().st_size == 0]
if zero_assets:
    errors.append(f"Zero-byte assets: {zero_assets}")
else:
    print("Assets: models + constants all non-empty")

print("\n=== WARNINGS ===" if warnings else "")
for w in warnings:
    print("W:", w)
print("\n=== ERRORS ===" if errors else "\nNo errors found.")
for e in errors:
    print("E:", e)

sys.exit(1 if errors else 0)
