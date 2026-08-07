# Manual STEP/STP CAD Module

[English](./cad.md) | [中文](./cad.zh.md) | [Documentation index](../README.md)

The manual-CAD path accepts STEP/STP and converts it directly with the Isaac
Sim/Omniverse CAD Converter. It preserves assembly hierarchy and does not use a
Blender/GLB round trip.

## Code and flow

| Code | Responsibility |
| --- | --- |
| `asset_pipeline/manual_cad.py` | Coordinate conversion, physics, optional visual materials, collection, and validation. |
| `asset_pipeline/jobs/cad.py` | Validate STEP/STP input and run CAD Converter. |
| `asset_pipeline/visual_materials/` | Assign reference-image materials. |
| `asset_pipeline/jobs/isaac.py` | Prepare geometry, author PhysX data, and collect dependencies. |
| `asset_pipeline/jobs/delivery.py` | Validate the USD after material assignment and the collected USD. |

```text
run_manual_cad_workflow
-> run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> run_add_physics_job(center_origin=True)
   -> normalize units and local origins, repair winding, author collision
-> optional run_assign_visual_materials_job
   -> register cameras, gather Part-ID evidence, select NVIDIA Base MDLs
-> tools/isaac/collect_usd_flat.py
-> final structure, dependency, and render checks
```

Older names such as `run_stp_physics_job`, `run_manual_cad_job`, and
`asset_pipeline/jobs/material.py` remain compatibility aliases. New code should
use `run_manual_cad_workflow` and
`asset_pipeline.visual_materials.run_assign_visual_materials_job`.

## Command

Create the SAM3 foreground annotation first as shown in the
[Part-ID quick start](../manual-part-id-materials.md), then run:

```bash
hunyuan-asset-pipeline \
  --manual-stp ./input/manual_asset.stp \
  --cad-usd-output-dir ./outputs/manual/asset_run/cad_usd \
  --intermediate-output-dir ./outputs/manual/asset_run/intermediate \
  --final-output-dir ./outputs/manual/asset_run/final \
  --auto-visual-materials \
  --visual-reference front=./references/front.jpg \
  --visual-reference side=./references/side.jpg \
  --visual-reference top=./references/top.jpg \
  --visual-reference iso=./references/iso.jpg \
  --visual-foreground-annotations ./annotations/sam3_foreground_annotations.json \
  --visual-material-output-dir ./outputs/manual/asset_run/visual_material \
  --acknowledge-mvinverse-noncommercial \
  --allow-policy-material-fallback \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32 \
  --set-mass 30
```

`--manual-stp` may also be a directory. The converter finds `.stp` and `.step`
recursively and mirrors their relative paths. The Physics-only path can process
multiple files; reference-image assignment requires exactly one CAD asset.

## Main options

| Option | Meaning |
| --- | --- |
| `--manual-stp` | STEP/STP file or directory. |
| `--cad-usd-output-dir` | Raw CAD Converter output; default is `<input>_cad_usd`. |
| `--cad-converter-option KEY=VALUE` | Extra converter option; repeat as needed. |
| `--intermediate-output-dir` | USD output after physics preparation. |
| `--final-output-dir` | Collected final asset directory. |
| `--auto-visual-materials` | Assign materials after geometry preparation and before collection. |
| `--visual-reference [ID=]IMAGE` | Reference image; provide 2–4 views of the same asset. |
| `--visual-foreground-annotations` | SAM3 whole-workpiece annotation created from those images. |
| `--visual-material-output-dir` | Material-analysis files and the USD after material assignment. |
| `--acknowledge-mvinverse-noncommercial` | Confirm that the run complies with the MVInverse license. |
| `--allow-policy-material-fallback` | Give unobserved or unresolved parts the configured default material. |
| `--material` | Physics-material preset from `materials.json`. |
| `--approx` | Collision approximation; `sdf` is recommended for complex dynamic CAD. |
| `--manual-sdf-resolution` | Manual-CAD SDF resolution, default `32`. |
| `--set-mass` | Total mass in kilograms; omit to estimate it from volume and density. |

STEP/STP does not accept `--len-x/y/z` or `--orientation`. Arbitrary scaling or
bounding-box rotation would change engineering dimensions and assembly
semantics. Unit conversion and origin centering preserve both.

## Processing details

### Geometry before visual materials

Physics geometry preparation runs before visual-material inference. Procedural
MDLs may use object-space coordinates, so changing units or mesh-local origins
after material selection can move or rescale a visible pattern. All material
renders therefore use the same normalized geometry.

Reference-image assignment handles one STEP/STP asset per run and assigns every
Mesh. Visible parts use their own evidence; unresolved parts use the configured
default material. See [Reference-image materials](./visual-materials.md).

### Geometry and collision

CAD conversion preserves the assembly hierarchy and transforms. The physics
stage converts units to meters, centers the visible asset with compensating
transforms, corrects unambiguous reversed faces, and creates collision data.
Open or nonmanifold meshes are reported rather than guessed. SDF behavior,
resolution guidance, and repair steps are documented in
[Physics](./physics.md).
