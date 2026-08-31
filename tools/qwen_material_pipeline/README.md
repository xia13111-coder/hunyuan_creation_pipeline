# Automatic Material Toolkit

[English](./README.md) | [中文](./README.zh.md)

This directory contains the STEP/STP automatic-material implementation:
segmentation, retrieval, material selection, USD binding, and validation.
Users should run `manual-material-pipeline` as described in
[Automatic materials](../../docs/guides/manual-part-id-materials.md); the
internal commands do not need to be called separately.

For development:

```bash
qwen-material --help
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

See [Architecture](../../docs/development/architecture.md) for the code layout. MVInverse is
non-commercial only; other licenses are listed in
[Third-party notices](../../legal/THIRD_PARTY_NOTICES.md).
