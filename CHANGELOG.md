# Changelog

Notable user-visible changes are recorded here. The project follows semantic
versioning for published CLI behavior and schemas where practical.

## [Unreleased]

### Added

- A dedicated `part_id_evidence` stage now owns the production isolated-CAD,
  two-pass SAM3/EntitySeg, neighbour-guided, iterative mask-fusion route.
- `python -m asset_pipeline.manual_material_cli` is now equivalent to the
  installed `manual-material-pipeline` command.
- Apache-2.0 first-party source license and third-party license inventory.
- Public release, contribution, security, conduct, and citation documentation.
- Automated public-source audit for credentials, machine paths, model weights,
  nested repositories, build products, and generated outputs.
- Portable model/cache configuration through environment variables and the
  standard user cache directory.

### Changed

- Repository support files are now grouped by ownership: community policies in
  `.github/`, third-party notices in `legal/`, subsystem configuration in
  `configs/`, dependency overlays in `requirements/`, and detailed documents
  under `docs/{guides,development,release}/`.
- Part-ID request construction now lives with segmentation code; Qwen3.5 setup,
  result-viewer serving, and optional Blender/Isaac operators are grouped with
  their owning runtime instead of the generic scripts directory.
- Redundant source-checkout-only Python forwarders were removed from the
  repository root. Use the installed CLI commands, `python -m asset_pipeline.*`
  module entry points, `asset_pipeline.api:app`, and `asset_pipeline` imports.
- The dedicated STEP/STP material CLI fully validates human foreground
  annotation, image, decoded-pixel, mask, and document hashes before starting.
- The canonical new-workpiece CLI exposes `live` and exact `bundled` modes;
  ambiguous `auto` fallback remains only in the compatibility Python API.
- The default Part-ID material profile is the single production configuration;
  its duplicate alias, unused private helpers, and unreferenced development
  comparison reports were removed.
- Root English and Chinese READMEs now serve as concise public entry points;
  implementation details remain in module documentation.
- Python package metadata now exposes license, project URLs, audience, and
  platform classifiers.
- The material-package wheel excludes tests and stale package-build trees.

The first public release will include the multi-input asset pipeline and the
STEP/STP Part-ID material workflow. A numbered section and comparison links
will be added when that release is tagged.
