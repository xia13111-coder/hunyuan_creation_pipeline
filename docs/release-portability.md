# Release and Portability

[English](./release-portability.md) | [中文](./release-portability.zh.md) | [Documentation index](./README.md)

The project should run from any checkout directory after local runtimes,
models, credentials, and mounts are configured. Source files must not depend on
one developer's username or installation paths.

## Paths

- Use repository-relative paths for project files.
- Supply external programs, caches, and NVIDIA assets through root `.env`
  variables such as `BLENDER_BIN`, `ISAAC_PYTHON`, `MODEL_CACHE_ROOT`, and
  `VISUAL_MATERIAL_ROOT`.
- Keep local models in the same `.env`: Qwen uses `QWEN_MODEL_PATH` /
  `QWEN35_MODEL_PATH`; MVInverse uses `MVINVERSE_REPOSITORY` /
  `MVINVERSE_CHECKPOINT`; SAM3 uses `SAM3_REPOSITORY` / `SAM3_CHECKPOINT`;
  SAM3D uses `SAM3D_MOGE_CHECKPOINT`, `SAM3D_DINOV2_REPOSITORY`, and
  `SAM3D_DINOV2_CHECKPOINT`; material retrieval uses `SIGLIP2_MODEL_PATH` /
  `DINOV2_MODEL_PATH`.
- The runtime fixes `PIPELINE_LOCAL_MODELS_ONLY=1`; normal inference does not download
  weights and fails clearly for missing or incomplete paths. Hunyuan generation
  and ReduceFace are cloud APIs and are outside this local-weight policy.
- Do not commit absolute symlinks or copy run-specific absolute paths into
  versioned configuration and reports.
- `/workspace`, `/isaac-sim`, `/opt/blender`, `/opt/conda`, and
  `/home/pipeline` are container paths. When moving to another host, update the
  host side of volume mounts rather than these container destinations.

## What belongs in each artifact

| Category | Source archive | Delivery method |
| --- | --- | --- |
| First-party source, configuration, tests, and documentation | Include | Publish with source |
| Qwen, MVInverse, SAM3/SAM3D, MoGe, DINOv2, SigLIP2, and NVIDIA material assets | Exclude | Download or read-only mount under their licenses |
| Vendored third-party source | Review individually | Include only when redistribution is allowed |
| Credentials and local `.env` files | Exclude | Create from the provided templates |
| `downloads/`, `outputs/`, `results/`, `var/`, and `workspace/` | Exclude | Generate locally under ignored paths |
| Caches, logs, and build products | Exclude | Rebuild locally |
| Docker offline parts | Separate artifact | Publish all parts with their SHA-256 manifest |

`tools/qwen_material_pipeline/` is the only copy of the material package. It is
installed beside the root package with:

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

## Credentials

The repository contains templates only. Put real cloud credentials in ignored
local files or the process environment, never in Dockerfiles, image layers,
tests, logs, or result archives. Revoke credentials that may have been exposed.

## Checks

Run the portability checks from the project root:

```bash
python ./tools/release/check_public_tree.py
git diff --check
conda run -n hunyuan_sam3d \
  python -m pytest -q tests/test_pipeline_structure.py \
  -k publishable_tree
```

The automated scan cannot establish ownership or privacy. Build releases from
a reviewed Git tag and follow the
[public release checklist](./public-release-checklist.md).

Verify the Docker bundle separately:

```bash
cd docker/offline-images
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

See [Docker Operations](../docker/README.md) for loading and validation.
