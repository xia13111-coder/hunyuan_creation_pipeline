# Changelog

Notable user-visible changes are recorded here. The project follows semantic
versioning for published CLI behavior and schemas where practical.

## [Unreleased]

### Added

- Apache-2.0 first-party source license and third-party license inventory.
- Public release, contribution, security, conduct, and citation documentation.
- Automated public-source audit for credentials, machine paths, model weights,
  nested repositories, build products, and generated outputs.
- Portable model/cache configuration through environment variables and the
  standard user cache directory.

### Changed

- Root English and Chinese READMEs now serve as concise public entry points;
  implementation details remain in module documentation.
- Python package metadata now exposes license, project URLs, audience, and
  platform classifiers.
- The material-package wheel excludes tests and stale package-build trees.

The first public release will include the multi-input asset pipeline and the
STEP/STP Part-ID material workflow. A numbered section and comparison links
will be added when that release is tagged.
