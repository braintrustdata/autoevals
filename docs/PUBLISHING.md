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
- `sha=<commit>` — the full 40-character commit SHA to release

You declare the exact commit to release (rather than a branch) so the release is
pinned to a specific, reviewed commit, and JS and Python publish the *same* commit.
Merge the version bump first, then dispatch `publish` with that commit's SHA.

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

It runs the shared, centrally-maintained release actions from
[`braintrustdata/sdk-actions`](https://github.com/braintrustdata/sdk-actions),
pinned by SHA (`braintrustdata/sdk-actions/actions/release/...@<sha>`). Bumping that
SHA pulls in upstream release-tooling improvements. autoevals keeps only its own glue
(version-sync, version/channel computation, the prerelease `package.json` patch) in
dedicated jobs/steps; the shared jobs contain only the pinned action.

It supports two release types:

- `stable`: publishes the exact version in `package.json`
- `prerelease`: publishes `<package.json version>-rc.<suffix>` with the `rc` dist-tag

It also takes a `dry_run` input (default `false`): when `true`, it builds and packs
(`npm publish --dry-run`) without publishing, tagging, or creating a release, and runs
under the `publish-dry-run` environment.

For stable releases, the workflow also:

- creates and pushes a git tag named `js-<version>`
- creates a GitHub Release named `autoevals JavaScript v<version>`

The flow is `compute-metadata → validate → prepare → notify-pending → publish`. The
`publish` job is gated by a GitHub environment (see below); Slack notifications are sent
before approval (pending) and after completion.

### Approval gate

The `publish` job uses GitHub Environments to require manual approval:

- `publish` — real publishes; required reviewers + `main`-only deployment branches
- `publish-dry-run` — used when `dry_run=true`

Create both under repo Settings → Environments and add the reviewers.

### npm trusted publishing setup

Configure trusted publishing for the `autoevals` package in npm with these values:

- Package: `autoevals`
- Provider: `GitHub Actions`
- Repository owner: `braintrustdata`
- Repository name: `autoevals`
- Workflow file: `.github/workflows/publish-js.yaml`
- Environment: `publish`

Notes:

- The workflow uses GitHub OIDC, so no `NPM_TOKEN` is required.
- **The trusted publisher must include the `publish` environment.** The gated job's
  OIDC token carries an `environment` claim; if the publisher isn't configured for it,
  the publish is rejected (`ENEEDAUTH`). A `dry_run` does *not* exercise this — the first
  real publish (use a prerelease as the canary) is the first true test.
- The workflow publishes with provenance enabled via `npm publish --provenance`.

### Slack notifications (optional)

The JS workflow posts a pending notification (before approval) and a completion
notification. These self-guard and no-op if unconfigured. To enable:

- Repository variable `SLACK_SDK_RELEASE_CHANNEL` — the channel ID
- Repository/org secret `SLACK_BOT_TOKEN` — the Brainbot token (invite it to the channel)

Python publishing currently has neither the approval gate nor Slack (JS only, for now).

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
   - `sha=<the merged commit's SHA>`
5. Approve the `publish` environment when the JS `publish` job requests it.

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
   - `sha=<the commit's SHA>`
4. Approve the `publish` environment when the JS `publish` job requests it.

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
- `sha` — the full commit SHA to release
- `prerelease_suffix` (optional)

`publish-js` additionally accepts `dry_run` (default `false`) — build and pack without
publishing, under the `publish-dry-run` environment.

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
uv sync --extra dev
uv run python -m build
uv run python -m twine check dist/*
```

## Historical releases and source mapping

Older npm releases may not be traceable back to an exact git commit from npm alone because they were published before trusted publishing and provenance attestations were enabled. In particular:

- npm metadata for older releases may not include `gitHead`
- those releases do not have OIDC/provenance attestations tying the package to a workflow run and commit

For those historical versions, the best commit mapping may need to be inferred from repository history, publish timestamps, and version bumps. New npm releases published through `.github/workflows/publish-js.yaml` are easier to trace because they use trusted publishing with provenance.

Python releases published through `.github/workflows/publish-py.yaml` are similarly expected to be easier to trace because they use PyPI trusted publishing via GitHub Actions OIDC.
