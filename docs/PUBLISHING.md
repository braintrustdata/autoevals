# Publishing

This repository contains both JavaScript and Python packages, both published as `autoevals`:

- npm package: `autoevals`
- PyPI package: `autoevals`

Publishing is handled via GitHub Actions trusted publishing.

## Workflows

Publishing workflows:

- `.github/workflows/publish.yaml` — manual dispatcher that triggers both package publish workflows
- `.github/workflows/publish-js.yaml` — npm publish workflow
- `.github/workflows/publish-py.yaml` — PyPI publish workflow
- `.github/workflows/version-sync.yaml` — CI check that JS/Python versions stay in sync

## Versioning policy

JavaScript and Python package versions must always match.

The canonical version files are:

- `package.json`
- `py/autoevals/version.py`

CI enforces this with:

- `.github/workflows/version-sync.yaml`
- `.github/scripts/check_version_sync.py`

If these versions do not match, CI fails and publish workflows fail early.

## Recommended publish flow

Use the top-level `publish` workflow for normal releases.

In GitHub Actions, manually run:

- `publish`

Inputs:

- `release_type=stable` or `prerelease`
- `branch=main` (or another branch to publish from)

This workflow dispatches both:

- `publish-js.yaml`
- `publish-py.yaml`

For prereleases, the dispatcher passes a shared prerelease suffix to both workflows so the releases stay aligned:

- npm: `<version>-rc.<suffix>`
- PyPI: `<version>rc<suffix>`

Example for base version `0.2.0` and suffix `123`:

- npm: `0.2.0-rc.123`
- PyPI: `0.2.0rc123`

## JavaScript npm publishing

The JavaScript publish workflow lives at:

- `.github/workflows/publish-js.yaml`

It supports two release types:

- `stable`: publishes the exact version in `package.json`
- `prerelease`: publishes `<package.json version>-rc.<suffix>` with the `rc` dist-tag

For stable releases, the workflow also:

- creates and pushes a git tag named `js-<version>`
- creates a GitHub Release named `autoevals JavaScript v<version>`

### npm trusted publishing setup

Configure trusted publishing for the `autoevals` package in npm with these values:

- Package: `autoevals`
- Provider: `GitHub Actions`
- Repository owner: `braintrustdata`
- Repository name: `autoevals`
- Workflow file: `.github/workflows/publish-js.yaml`

Notes:

- The workflow uses GitHub OIDC, so no `NPM_TOKEN` is required.
- The workflow publishes with provenance enabled via `npm publish --provenance`.

## Python PyPI publishing

The Python publish workflow lives at:

- `.github/workflows/publish-py.yaml`

It supports two release types:

- `stable`: publishes the exact version in `py/autoevals/version.py`
- `prerelease`: publishes a PEP 440 prerelease version `<python version>rc<suffix>`

For stable releases, the workflow also:

- creates and pushes a git tag named `py-<version>`
- creates a GitHub Release named `autoevals Python v<version>`

### PyPI trusted publishing setup

Configure trusted publishing for the `autoevals` project in PyPI with these values:

- Project name: `autoevals`
- Owner: `braintrustdata`
- Repository name: `autoevals`
- Workflow file: `.github/workflows/publish-py.yaml`

Notes:

- The workflow uses GitHub OIDC, so no PyPI API token is required.
- The workflow publishes via `pypa/gh-action-pypi-publish`.
- The workflow must have `id-token: write` permission for trusted publishing.

## How to publish a stable release

1. Bump both versions together:
   - `package.json`
   - `py/autoevals/version.py`
2. Merge the change to `main`.
3. In GitHub Actions, run the `publish` workflow.
4. Choose:
   - `release_type=stable`
   - `branch=main`

Expected outcome:

- npm package `autoevals@<version>` is published
- PyPI package `autoevals==<version>` is published
- git tag `js-<version>` is created and pushed
- git tag `py-<version>` is created and pushed
- GitHub Release `autoevals JavaScript v<version>` is created
- GitHub Release `autoevals Python v<version>` is created

## How to publish a prerelease

1. Make sure both version files contain the same base version:
   - `package.json`
   - `py/autoevals/version.py`
2. In GitHub Actions, run the `publish` workflow.
3. Choose:
   - `release_type=prerelease`
   - `branch=main`

Expected outcome:

- npm package `autoevals@<version>-rc.<suffix>` is published
- npm dist-tag `rc` is updated
- PyPI package `autoevals==<version>rc<suffix>` is published
- no stable git tags are created
- no GitHub Releases are created

## Publishing package-specific workflows directly

If needed, you can manually trigger either workflow directly:

- `publish-js`
- `publish-py`

Both accept:

- `release_type`
- `branch`
- `prerelease_suffix` (optional)

Normally you should prefer the top-level `publish` workflow so JS and Python prereleases use the same suffix.

## Safeguards in the workflows

The workflows fail early if:

- `package.json` and `py/autoevals/version.py` do not match
- the stable JS tag `js-<version>` already exists on `origin`
- the stable Python tag `py-<version>` already exists on `origin`
- the npm version being published already exists
- the PyPI version being published already exists

## Local validation

Useful commands before triggering a release:

```bash
python3 .github/scripts/check_version_sync.py
pnpm install --frozen-lockfile
pnpm run build
npm publish --dry-run --access public
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine check dist/*
```

## Historical releases and source mapping

Older npm releases may not be traceable back to an exact git commit from npm alone because they were published before trusted publishing and provenance attestations were enabled. In particular:

- npm metadata for older releases may not include `gitHead`
- those releases do not have OIDC/provenance attestations tying the package to a workflow run and commit

For those historical versions, the best commit mapping may need to be inferred from repository history, publish timestamps, and version bumps. New npm releases published through `.github/workflows/publish-js.yaml` are easier to trace because they use trusted publishing with provenance.

Python releases published through `.github/workflows/publish-py.yaml` are similarly expected to be easier to trace because they use PyPI trusted publishing via GitHub Actions OIDC.
