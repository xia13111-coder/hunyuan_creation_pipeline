# Manual STEP/STP CAD Module

[English](./cad.md) | [中文](./cad.zh.md) | [Documentation index](../README.md)

This module converts STEP/STP directly to USD with the Isaac Sim/Omniverse CAD
Converter. It preserves assembly hierarchy and does not use Blender or GLB as
an intermediate format.

## Call flow

| Code | Responsibility |
| --- | --- |
| `asset_pipeline/manual_cad.py` | Coordinate CAD conversion, physics, optional automatic materials, and validation. |
| `asset_pipeline/jobs/cad.py` | Validate input and call CAD Converter. |
| `asset_pipeline/visual_materials/` | Select and bind visual materials from reference images. |
| `asset_pipeline/jobs/isaac.py` | Prepare geometry, collision, and USD dependencies. |
| `asset_pipeline/jobs/delivery.py` | Validate the final USD. |

```text
run_manual_cad_workflow
-> CAD to USD
-> geometry and physics
-> optional reference-image materials
-> collect dependencies and validate the final USD
```

## Usage

See the project [README](../../README.md) for basic CAD commands and
[Automatic materials](../guides/manual-part-id-materials.md) for the
reference-image workflow. `--manual-stp` accepts a file or directory;
reference-image assignment processes one CAD asset per run.

STEP/STP retains its engineering dimensions, so `--len-x/y/z` and
`--orientation` are not used.

## Geometry and collision

The physics stage converts units to meters, centers the visible asset, repairs
unambiguous reversed faces, and creates collision data. Open or nonmanifold
meshes are reported rather than guessed. See [Physics](./physics.md) for SDF
settings and repair guidance.
