#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = ROOT / "package.json"
PY_VERSION = ROOT / "py" / "autoevals" / "version.py"
JS_VERSION = ROOT / "js" / "version.ts"

package_version = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["version"]
match = re.search(
    r'^VERSION\s*=\s*["\']([^"\']+)["\']\s*$',
    PY_VERSION.read_text(encoding="utf-8"),
    re.MULTILINE,
)

if not match:
    print(f"Could not parse VERSION from {PY_VERSION}", file=sys.stderr)
    sys.exit(1)

python_version = match.group(1)

js_match = re.search(
    r'^export const VERSION = ["\']([^"\']+)["\'];?\s*$', JS_VERSION.read_text(encoding="utf-8"), re.MULTILINE
)

if not js_match:
    print(f"Could not parse VERSION from {JS_VERSION}", file=sys.stderr)
    sys.exit(1)

if js_match.group(1) != package_version:
    print(
        "Version mismatch detected:\n" f"- package.json: {package_version}\n" f"- js/version.ts: {js_match.group(1)}",
        file=sys.stderr,
    )
    sys.exit(1)

if package_version != python_version:
    print(
        "Version mismatch detected:\n"
        f"- package.json: {package_version}\n"
        f"- py/autoevals/version.py: {python_version}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Versions are in sync: {package_version}")
