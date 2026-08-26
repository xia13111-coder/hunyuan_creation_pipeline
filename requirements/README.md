# Requirement sets

The root Conda environment remains defined by [`environment.yml`](../environment.yml).
This directory contains smaller, purpose-specific dependency overlays:

- [`visual-materials.txt`](./visual-materials.txt) adds the optional local Qwen,
  MVInverse, retrieval, and image-processing dependencies used by the visual
  material Docker layers.

The independently installable Qwen material package keeps its environment-
specific requirement sets in `tools/qwen_material_pipeline/`. Keeping those
files at that package boundary allows it to be installed and tested on its own.
