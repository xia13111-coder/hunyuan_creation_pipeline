# Public Release Checklist

[English](./public-release-checklist.md) | [中文](./public-release-checklist.zh.md) | [Documentation index](../README.md)

Use this checklist for each source release. Automated checks complement, but do
not replace, a review of ownership, privacy, and license terms.

## Content and licenses

- Confirm that every first-party file may be released under Apache-2.0.
- Include `LICENSE`, `NOTICE`, `legal/THIRD_PARTY_NOTICES.md`, and the licenses for
  redistributed third-party source.
- Exclude model weights, NVIDIA assets, Isaac Sim files, Blender binaries, and
  generated results unless their licenses explicitly allow redistribution.
- Record the source, revision, hash, and license of checkpoints used for a
  published result. MVInverse-dependent use remains non-commercial unless
  separately licensed.

## Privacy and credentials

- Review photographs, CAD/USD/GLB files, masks, screenshots, logs, and JSON
  reports for confidential or personal information.
- Confirm that `.env`, credentials, private endpoints, usernames, host-specific
  paths, and private repository URLs are absent from both the tree and Git
  history.
- Revoke a credential if it has entered a commit, release directory, image
  build context, uploaded artifact, or shared log.
- Review `apps/material_audit_web/` separately if it will be published; the root
  source archive excludes this nested application and its local audit data.

## Reproducibility

- Test installation on a clean host or container.
- Keep example commands repository-relative.
- Supply models and NVIDIA materials through local configuration or read-only
  mounts; do not bundle them silently.
- Generate an SBOM and dependency-license inventory for redistributed
  containers or offline images.

## Verification

Run from the repository root:

```bash
python ./tools/release/check_public_tree.py
git diff --check
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

Record skipped integration tests and unavailable runtimes or models. Test a
binary or container release inside the exact artifact being distributed.

## Source archive

Build from a reviewed Git tag, not from the live working directory:

```bash
VERSION=0.1.0  # replace with the version being published
git rev-parse --verify "v${VERSION}^{commit}"
git archive --format=tar.gz \
  --prefix="hunyuan_creation_pipeline-${VERSION}/" \
  -o "hunyuan_creation_pipeline-${VERSION}.tar.gz" \
  "v${VERSION}"
sha256sum "hunyuan_creation_pipeline-${VERSION}.tar.gz"
```

Extract and inspect the archive before upload. Release notes should list
supported inputs and runtime versions, user-visible changes, migration steps,
known limitations, test coverage, the archive checksum, and third-party license
or model changes.
