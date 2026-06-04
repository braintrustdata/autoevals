#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = ROOT / "package.json"
PY_VERSION = ROOT / "py" / "autoevals" / "version.py"

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

if package_version != python_version:
    print(
        "Version mismatch detected:\n"
        f"- package.json: {package_version}\n"
        f"- py/autoevals/version.py: {python_version}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Versions are in sync: {package_version}")
