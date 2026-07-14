# Blender Postprocess Module

[English](./blender.md) | [中文](./blender.zh.md) | [Documentation index](../README.md)

This module processes refined GLBs: bounding-box axis mapping, target sizing, geometry centering, and Z-up USD export. It does not own the Hunyuan API or PhysX properties.

## Code And Order

```text
asset_pipeline/workflows.run_postprocess_job
-> jobs.blender.blender_preflight
-> jobs.blender.run_align_job
   -> tools/blender/align_glb_axis_only.py
-> jobs.blender.run_resize_job
   -> tools/blender/resize_glb_xyz_and_center.py
-> jobs.blender.run_convert_job
   -> tools/blender/convert_glb_to_usd_zup.py
```

`BLENDER_BIN` selects Blender. The preflight checks the executable, version, and GLB count before any model is changed.

## Axis Mapping

`--orientation` uses `X=<rank>,Y=<rank>,Z=<rank>`:

| Symbol | Meaning |
| --- | --- |
| `L` | Longest current bounding-box dimension |
| `M` | Middle dimension |
| `S` | Shortest dimension |

Common mappings:

| Value | Result |
| --- | --- |
| `X=L,Y=M,Z=S` | Longest dimension along X, middle along Y, and shortest along Z when possible. |
| `X=M,Y=L,Z=S` | Swap the two horizontal dimensions. |
| `X=L,Y=S,Z=M` | Request the middle dimension as Z; this is exact only when the original vertical rank permits it. |

The current script rotates around world Z by `0`, `90`, `180`, or `270` degrees only. It does not tip the model around X/Y, so it never exchanges the original vertical axis with a horizontal axis merely to satisfy a rank request.

## Sizing And Centering

| Option | Meaning |
| --- | --- |
| `--len-x` | Target X size in meters. |
| `--len-y` | Target Y size in meters. |
| `--len-z` | Target Z size in meters. |

`resize_glb_xyz_and_center.py` applies nonuniform scaling to the three requested dimensions and moves the complete visible bounding-box center to the world origin. Axis mapping runs first, so sizes refer to aligned world X/Y/Z.

## USD Conversion

`convert_glb_to_usd_zup.py` uses Blender's USD exporter:

- The stage receives `upAxis = Z`.
- The pipeline exports `.usd` by default.
- A file input creates a same-name output directory; a directory input processes its GLBs.
- Blender exports visual materials and textures into USD and associated resources.

## Difference From CAD

STEP/STP inputs do not pass through this module. The CAD path must preserve assembly transforms and hierarchy, so CAD-to-USD, unit normalization, and origin cleanup happen directly in Isaac Sim.

## Handoff To Physics

`run_convert_job` returns `usd_input_path`. The workflow passes that exact USD file or directory to `jobs.isaac.run_add_physics_job`, avoiding unrelated USD files nearby.
