# Publishing

Publishes the npm and PyPI `autoevals` packages **together, at one shared version, behind one
approval gate** — via a single workflow, `.github/workflows/publish.yaml`, using the shared release
actions from [`braintrustdata/sdk-actions`](https://github.com/braintrustdata/sdk-actions) (pinned by SHA).

## Versioning

`package.json` and `py/autoevals/version.py` must hold the same version; `version-sync.yaml` (CI)
enforces it on PRs. The release publishes the **committed** version — bump both and merge first.

## Releasing

In GitHub Actions, run **`publish`** with:

- `release_type` — `stable` (default) or `prerelease`
- `sha` — full 40-char commit SHA to release (same commit for js + py)
- `prev_release` (optional) — release-notes anchor; empty → each lane's previous tag
- `dry_run` (default `false`) — build + pack + attest, no publish or release

One approval releases both packages; if either lane fails before the gate, neither ships.

- **stable** → npm `autoevals@<v>` (provenance) + PyPI `autoevals==<v>` (attestation); tags `js-<v>`
  and `py-<v>`; two GitHub Releases.
- **prerelease** → publishes the committed version under the npm `rc` dist-tag; no GitHub release.

### Steps

1. Bump `package.json` + `py/autoevals/version.py` together and merge to `main`.
2. Run `publish` with `release_type=stable`, `sha=<merged commit>`.
3. Approve the `publish` environment when the run parks at the gate.

## Requirements & notes

- **Environments** `publish` (required reviewers) and `publish-dry-run` must exist (repo Settings).
- **Trusted publishers** for npm `autoevals` and PyPI `autoevals` must be configured with **environment
  `publish`** (OIDC — no tokens). The environment claim is what confines publishing to the gated ship
  jobs; without it, publishing fails. A `dry_run` never publishes, so the first real publish is the
  first true test of this config.
- **pnpm**: install + build run on the repo's pinned `pnpm@10.33.0` (honoring the supply-chain
  config in `pnpm-workspace.yaml`). The packer automatically uses a pnpm ≥ 11.8 for `pnpm pack` and
  the SBOM — those two commands require it — so no release-specific pnpm setup is needed here.
- **Slack** notifications are optional: variable `SLACK_SDK_RELEASE_CHANNEL`, secret `SLACK_BOT_TOKEN`.

## Local checks

```bash
python3 .github/scripts/check_version_sync.py
pnpm install --frozen-lockfile && pnpm run build
uv build && uvx twine check dist/*
```
