#!/usr/bin/env python3

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
README_FILE = os.path.join(SCRIPT_DIR, "..", "README.md")

if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ["py", "js"], mode

    with open(README_FILE, "r") as f:
        readme = f.read()

    remove_section = "Python" if mode == "js" else "Node.js"

    # Remove the whole section
    readme = re.sub(
        r"\#+\s*" + remove_section + r"\s*\n.*?((^\#\#+)|\Z)",
        r"\1",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Remove the "Python" or "Node.js" header
    remove_header = "Python" if mode == "py" else "Node.js"
    readme = re.sub(r"\#+\s*" + remove_header + r"\s*\n", "", readme)

    readme = readme.strip()

    with open(README_FILE, "w") as f:
        f.write(readme)
