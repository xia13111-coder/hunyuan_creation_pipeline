# Repository Layout

[English](./repository-layout.md) | [中文](./repository-layout.zh.md) | [Documentation index](./README.md)

## Top-level directories

```text
asset_pipeline/                 workflows and orchestration
asset_pipeline/visual_materials/stages/
                                owned material-pipeline stages
asset_refiner/                  mesh-refinement package
tools/{blender,isaac,sam3d}/    workers for external runtimes
tools/qwen_material_pipeline/   material inference and USD tools
configs/                        versioned configuration
docs/                           user and developer documentation
tests/                          main-pipeline tests
apps/                           standalone web applications
examples/                       example instructions and public metadata
outputs/                        generated runs; ignored by Git
docker/                         container files and instructions
```

`asset_pipeline/project_layout.py` resolves repository paths. Other modules
should use it instead of rebuilding paths to `tools/`, `configs/`, or
`outputs/`.

## Main material-package directories

This is a functional overview, not an exhaustive directory listing.

```text
tools/qwen_material_pipeline/
├── workflows/       command workflows
├── evidence/        camera, Part-ID, color, PBR, and QA measurements
├── retrieval/       SigLIP2 and DINOv2 retrieval
├── materials/       MDL catalog, selection, and application rules
├── segmentation/    SAM3, EntitySeg, relation guidance, and hybrid masks
├── mvinverse/       MVInverse adapter and run records
├── qwen/            local and remote VLM adapters
├── usd/             part indexing, rendering, application, and validation
├── core/            shared data structures
├── configs/         package configuration
├── schemas/         JSON schemas
├── web/             annotation and result viewers
├── third_party/     vendored source with upstream licenses
├── models/          local weights; ignored by Git
├── var/             rebuildable indexes and caches; ignored by Git
└── results/         local results; ignored by Git
```

The material package has its own `pyproject.toml`, but it remains under
`tools/`; do not create a second copy or a compatibility symlink at the
repository root.

## Placement rules

- Store each run under `outputs/<run-id>/`.
- Do not place photographs, private CAD files, credentials, model weights, or
  generated results in the source release.
- Keep Blender, Isaac Sim, and SAM3D workers under their matching `tools/`
  directory; workflow orchestration belongs in `asset_pipeline/`.
- `apps/` may consume published results but must not be imported by
  `asset_pipeline`.
- Remove caches only when they are reproducible. Review user inputs, evidence,
  models, and USD files before deleting them.

Install both Python packages once in the active environment:

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

This installs `hunyuan-asset-pipeline`, `manual-material-pipeline`, and
`qwen-material`. The root `run_*.py` scripts remain compatibility entry points;
new material automation should use `manual-material-pipeline` or
`python -m asset_pipeline.manual_material_cli`.
