# Contributing

Thank you for improving Hunyuan Asset Creation Pipeline. Contributions should
keep the repository portable, auditable, and safe to publish.

## Before opening a change

1. Discuss large workflow, schema, runtime, or dependency changes in an issue.
2. Do not attach proprietary CAD, private photographs, model weights, NVIDIA
   assets, credentials, or generated run directories.
3. Confirm that you have the right to submit every file in the change.
4. Read the [Code of Conduct](./CODE_OF_CONDUCT.md) and
   [security policy](./SECURITY.md).

## Development setup

```bash
conda env create -f environment.yml
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

External runtimes and weights are optional for unit tests. Integration tests
that require Blender, Isaac Sim, `pxr`, Gradio, CUDA, or local checkpoints may
skip when those dependencies are unavailable.

## Required checks

Run the checks relevant to the changed area. Before submitting a pull request,
run the complete source-only suite when possible:

```bash
python -m pytest -q -p no:cacheprovider tests
PYTHONPATH=./tools/qwen_material_pipeline/src python -m pytest -q -p no:cacheprovider \
  tools/qwen_material_pipeline/tests
python ./tools/release/check_public_tree.py
git diff --check
```

Do not loosen safety or quality checks merely to make a test pass. When a schema
or validation rule changes, update its implementation, tests, and documentation
together.

## Change guidelines

- Keep public entry points small and place implementation in the module
  responsible for that behavior.
- Use repository-relative paths in source, configuration, stable evidence, and
  documentation. Runtime paths belong in environment variables or local files.
- Preserve deterministic inputs, hashes, and provenance when changing resumed
  or cached stages.
- Add tests for bug fixes and public behavior.
- Update both English and Chinese user documentation for user-visible changes.
- Keep generated files under `outputs/`; never commit local results as fixtures
  unless they are minimal, synthetic, documented, and legally redistributable.

## Pull requests

A pull request should explain the problem, behavior change, verification,
compatibility impact, and any new third-party dependency or license. Keep
unrelated changes in separate pull requests.

Unless explicitly stated otherwise, a contribution intentionally submitted to
this project is provided under the Apache License 2.0, consistent with section
5 of [LICENSE](../LICENSE). Third-party code must retain its own license.
