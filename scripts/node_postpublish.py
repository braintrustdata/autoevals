#!/usr/bin/env python3

import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    subprocess.call(["git", "checkout", os.path.join(SCRIPT_DIR, "..", "package.json")])
    subprocess.call(["git", "checkout", os.path.join(SCRIPT_DIR, "..", "README.md")])
