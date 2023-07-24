#!/usr/bin/env python3

import json
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "py"))
from autoevals.version import VERSION


if __name__ == "__main__":
    package_file = os.path.join(SCRIPT_DIR, "..", "package.json")
    with open(package_file, "r") as f:
        package_json = json.load(f)

    package_json["version"] = VERSION

    with open(package_file, "w") as f:
        json.dump(package_json, f, indent=2)

    subprocess.call([os.path.join(SCRIPT_DIR, "prepare_readme.py"), "js"])

    subprocess.call(["npm", "run", "build"], cwd=os.path.join(SCRIPT_DIR, ".."))
