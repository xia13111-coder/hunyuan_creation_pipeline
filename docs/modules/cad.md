# Manual STEP/STP CAD Module

[English](./cad.md) | [中文](./cad.zh.md) | [Documentation index](../README.md)

Hand-modeled assets use STEP/STP input only. This path generates USD directly through the Isaac Sim/Omniverse CAD Converter, preserves assembly hierarchy, and avoids a Blender GLB round trip.

## Code And Flow

```text
asset_pipeline.workflows.run_stp_physics_job
-> jobs.isaac.run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> for each CAD USD:
   -> tools/isaac/add_physics.py --center-origin
   -> tools/isaac/collect_usd_flat.py
```

## Command

```bash
python ./run_asset_pipeline.py \
  --manual-stp ./input/manual_asset.stp \
  --cad-usd-output-dir ./manual_cad_usd \
  --intermediate-output-dir ./manual_output_intermediate \
  --final-output-dir ./manual_output_final \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32 \
  --set-mass 30
```

`--manual-stp` also accepts a directory. The converter recursively discovers `.stp` and `.step` files and mirrors their relative layout in the output.

## Options

| Option | Meaning |
| --- | --- |
| `--manual-stp` | STEP/STP file or directory. |
| `--cad-usd-output-dir` | Raw CAD Converter USD output; defaults to `<input>_cad_usd`. |
| `--cad-converter-option KEY=VALUE` | Extra Omniverse CAD converter option; may be repeated. |
| `--intermediate-output-dir` | Physics-authored USD output. |
| `--final-output-dir` | Final collection directory. |
| `--material` | Explicit physics material from `materials.json`. |
| `--approx` | Collision approximation; `sdf` is recommended for complex dynamic CAD and automatically enables SDF remeshing on this path. |
| `--manual-sdf-resolution` | SDF resolution for manual CAD, default `32`. Higher values preserve more collision detail but increase cooking, memory, and collision cost. |
| `--set-mass` | Total asset mass in kilograms; volume and density are used when omitted. |

`--len-x/y/z` and `--orientation` are not used by the CAD path because arbitrary nonuniform scaling and bounding-box rotation would damage assembly-transform semantics.

Manual CAD no longer inherits the direct script's general SDF default of `256`; it uses `32` to prioritize real-time performance for complex assemblies. Raise it to `64`, `128`, or `256` only when thin walls, holes, or small collision features are lost. This option affects only the manual STEP/STP path and does not change the Hunyuan or SAM3D paths.

## USD Hierarchy

The Xform/Mesh hierarchy produced by the CAD Converter is preserved. Physics preparation does not join every mesh:

- The top asset root remains stable.
- Assembly child transforms are retained.
- Instances/prototypes are deinstanced when physics processing requires it.
- Physics materials live below the asset anchor instead of being scattered at stage root.

## Units

The final stage uses:

```text
upAxis = Z
metersPerUnit = 1.0
```

When the CAD Converter outputs millimeters, `normalize_stage_units_to_meters` scales mesh points, translations, pivots, and other length values into meters while preserving world appearance and assembly relationships.

## World And Local Origins

The physics stage uses `--center-origin`:

1. Compute the complete visible world bounding-box center.
2. Move visible descendants to the world origin while keeping the top root transform at zero.
3. Move each mesh's points close to its local bounding-box center.
4. Compensate with `xformOp:translate:meshLocalOrigin`, preserving visual placement.

The result has zero top-level XYZ transform, geometry centered near world origin, and mesh-local points without large offsets.

## Inverted Winding And PhysX

STEP/STP tessellation can produce reversed faces, open shells, nonmanifold edges, or mirrored assembly transforms. These defects may remain visually hidden by double-sided rendering but fail when PhysX computes collision volume and mass:

```text
PhysX error: attachShape ... negative mass
```

Before collision cooking, the pipeline now performs the following checks:

1. Normalize USD mesh orientation from `leftHanded` to `rightHanded`.
2. Count boundary, nonmanifold, inconsistent shared, and invalid edges/faces.
3. For a closed mesh, compute signed volume using its full world transform, including mirrored parents.
4. Reverse all face indices when the resulting world-space volume is negative.
5. When `--approx sdf` is requested, keep the authored approximation as `sdf` and enable `sdfEnableRemeshing` so PhysX repairs problematic tessellation before SDF cooking.

The correction changes topology orientation only; it does not flatten the assembly hierarchy or change visible placement. Open and nonmanifold visual geometry is not rewritten because its outside direction is ambiguous. Repairing the STEP source is still recommended when an exact SDF collision shape is required.

The requested approximation is never silently replaced. A topology warning identifies the affected mesh while SDF remeshing handles common CAD winding, self-intersection, and open-shell defects. If PhysX still reports a cooking error, the CAD source has geometry that remeshing cannot repair reliably and should be healed in the CAD authoring tool.
