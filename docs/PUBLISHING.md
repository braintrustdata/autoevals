# Publishing

This repository contains both JavaScript and Python packages. The JavaScript package (`autoevals`) is published to npm via GitHub Actions trusted publishing with provenance attestations.

## JavaScript npm publishing

The JavaScript publish workflow lives at:

- `.github/workflows/publish-js.yaml`

It supports two release types:

- `stable`: publishes the exact version in `package.json`
- `prerelease`: publishes `<package.json version>-rc.<github run number>` with the `rc` dist-tag

For stable releases, the workflow also:

- creates and pushes a git tag named `js-<version>`
- creates a GitHub Release named `autoevals v<version>`

## npm trusted publishing setup

Configure trusted publishing for the `autoevals` package in npm with these values:

- Package: `autoevals`
- Provider: `GitHub Actions`
- Repository owner: `braintrustdata`
- Repository name: `autoevals`
- Workflow file: `.github/workflows/publish-js.yaml`
- Environment: `npm-publish`

Notes:

- The workflow uses GitHub OIDC, so no `NPM_TOKEN` is required.
- The workflow publishes with provenance enabled via `npm publish --provenance`.

## GitHub environment setup

Create a GitHub Actions environment named:

- `npm-publish`

Recommended configuration:

- restrict deployments to `main`
- add required reviewers if you want manual approval before publish

The workflow already references this environment:

```yaml
environment: npm-publish
```

## How to publish a stable release

1. Bump the JavaScript package version in `package.json`.
2. Merge the change to `main`.
3. In GitHub Actions, run the `publish-js` workflow.
4. Choose:
   - `release_type=stable`
   - `branch=main`

Expected outcome:

- npm package `autoevals@<version>` is published
- git tag `js-<version>` is created and pushed
- GitHub Release `autoevals v<version>` is created

## How to publish a prerelease

1. Make sure `package.json` contains the base version you want to prerelease from.
2. In GitHub Actions, run the `publish-js` workflow.
3. Choose:
   - `release_type=prerelease`
   - `branch=main`

Expected outcome:

- npm package `autoevals@<version>-rc.<run_number>` is published
- npm dist-tag `rc` is updated
- no git tag is created
- no GitHub Release is created

## Safeguards in the workflow

The workflow will fail early if:

- the stable tag `js-<version>` already exists on `origin`
- the npm version being published already exists

## Local validation

Useful commands before triggering a release:

```bash
pnpm install --frozen-lockfile
pnpm run build
npm publish --dry-run --access public
```

## Historical releases and source mapping

Older npm releases may not be traceable back to an exact git commit from npm alone because they were published before trusted publishing and provenance attestations were enabled. In particular:

- npm metadata for older releases may not include `gitHead`
- those releases do not have OIDC/provenance attestations tying the package to a workflow run and commit

For those historical versions, the best commit mapping may need to be inferred from repository history, publish timestamps, and version bumps. New releases published through `.github/workflows/publish-js.yaml` are expected to be easier to trace because they use trusted publishing with provenance.

## Future publishing work

Python publishing is not yet covered by this document. When a Python release workflow is added, keep Python tags and release process separate from the JavaScript `js-<version>` tag namespace.
