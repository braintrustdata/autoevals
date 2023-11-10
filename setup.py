import os

import setuptools

dir_name = os.path.abspath(os.path.dirname(__file__))

version_contents = {}
with open(os.path.join(dir_name, "py", "autoevals", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version_contents)

with open(os.path.join(dir_name, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = ["chevron", "levenshtein", "pyyaml"]

extras_require = {
    "dev": [
        "black",
        "build",
        "flake8",
        "flake8-isort",
        "IPython",
        "isort==5.12.0",
        "pre-commit",
        "pytest",
        "twine",
    ],
    "doc": ["pydoc-markdown"],
}

extras_require["all"] = sorted({package for packages in extras_require.values() for package in packages})

setuptools.setup(
    name="autoevals",
    version=version_contents["VERSION"],
    author="BrainTrust",
    author_email="info@braintrustdata.com",
    description="Universal library for evaluating AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.braintrustdata.com",
    project_urls={
        "Bug Tracker": "https://github.com/braintrustdata/autoevals",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "py"},
    include_package_data=True,
    packages=setuptools.find_packages(where="py"),
    python_requires=">=3.9.0",
    entry_points={},
    install_requires=install_requires,
    extras_require=extras_require,
)
