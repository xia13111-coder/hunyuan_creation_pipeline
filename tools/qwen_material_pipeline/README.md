# Automatic Material Toolkit

[English](./README.md) | [中文](./README.zh.md)

This directory contains the STEP/STP automatic-material implementation:
segmentation, retrieval, material selection, USD binding, and validation.
Users should run `manual-material-pipeline` as described in
[Automatic materials](../../docs/guides/manual-part-id-materials.md); the
internal commands do not need to be called separately.

## Relationship to the main workflow

```text
manual-material-pipeline
-> asset_pipeline.manual_material_cli
-> asset_pipeline.manual_cad
-> asset_pipeline.visual_materials
-> qwen_material_pipeline subprocess tools
```

`asset_pipeline` owns stage order, runtime boundaries, and final delivery. This
package owns model inference, segmentation, material candidates, image
comparison, and USD material operations.

| Directory | Contents |
| --- | --- |
| `segmentation/` | SAM3, EntitySeg, and fused segmentation. |
| `evidence/` | Part-ID, camera, and photograph evidence. |
| `mvinverse/`, `retrieval/`, `qwen/` | Appearance estimation, retrieval, and ranking. |
| `materials/` | Material plans, candidate comparison, and selection locks. |
| `usd/` | Part-ID rendering, material binding, and USD checks. |
| `workflows/` | Internal stages started by the owning workflow. |

Use `qwen-material --help` as the internal command reference. Artifacts are
written to the job's `visual_material/` directory, never into this source tree.

## Development checks

```bash
qwen-material --help
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

See [Architecture](../../docs/development/architecture.md) for the code layout. MVInverse is
non-commercial only; other licenses are listed in
[Third-party notices](../../legal/THIRD_PARTY_NOTICES.md).
