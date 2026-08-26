# Isaac Sim Physics Module

[English](./physics.md) | [中文](./physics.zh.md) | [Documentation index](../README.md)

This module authors collision, rigid bodies, physics materials, and mass on USD, then collects the USD and its material/texture dependencies. Visual PBR and PhysX materials are separate data systems.

For manual STEP/STP assets, `add_physics.py` first normalizes units and local
origins, repairs winding, and authors collision, rigid-body, mass, and
Physics-purpose material data. Qwen/MVInverse then select appearance MDLs on
that prepared geometry. Visual `allPurpose` bindings and PhysX-purpose
materials remain separate.

## Code

```text
asset_pipeline/jobs/isaac.py
  run_add_physics_job
  -> tools/isaac/add_physics.py

  run_collect_job
  -> tools/isaac/collect_usd_flat.py
```

`ISAAC_PYTHON` must point to the installed Isaac Sim `python.sh`. `tools/isaac/isaac_sim_compat.py` handles the Isaac Sim 5/6 `SimulationApp` import locations.

## Physics Materials

`--material` explicitly selects a preset from
[`configs/physics/materials.json`](../../configs/physics/materials.json). The code does not infer a physics
material from prim names or visual appearance.

| Field | Unit/type | Purpose |
| --- | --- | --- |
| `density` | kg/m^3 | Estimate mass as volume times density when `--set-mass` is omitted. |
| `friction` | coefficient | Static friction. |
| `dyn_ratio` | ratio | Dynamic friction is `friction * dyn_ratio`; default is `0.9`. |
| `restitution` | coefficient | Bounciness. `0` is close to no rebound. |
| `combine` | string | Optional friction/restitution combine mode, such as `average`. |

The configuration contains general presets and more specific metals, polymers,
elastomers, wood, glass, ceramic, and concrete. Treat them as simulation
baselines; use measured values when the exact grade and contact pair are known.
Read the current values from `configs/physics/materials.json` rather than copying them into
another configuration.

## Collision Approximation

| Mode | Best use | Tradeoff |
| --- | --- | --- |
| `sdf` | Dynamic complex/concave meshes; current recommended default. | Good concave fidelity with higher cooking cost. Hand-authored CAD also enables SDF remeshing. |
| `convexHull` | Simple, mostly convex objects. | Fast and stable, but fills cavities. |
| `convexDecomposition` | Concave approximation without SDF. | Multiple hulls improve fidelity at higher compute/collision cost. |
| `boundingCube` | Coarse box proxy. | Fastest and least accurate. Aliases: `box`, `cube`. |
| `boundingSphere` | Round objects. | Fast and stable, inaccurate for elongated assets. Alias: `sphere`. |
| `sphereApproximation` | Assets that accept a PhysX sphere approximation. | Simple proxy. |
| `triangleMesh` | Static environments. | Automatically changed to `sdf` for dynamic rigid bodies. |
| `meshSimplification` | Static simplified meshes. | Automatically changed to `sdf` for dynamic rigid bodies. |

The main pipeline selects a mode with `--approx`. Direct `add_physics.py` usage also exposes SDF, VHACD, contact-offset, and solver-iteration options.

The direct script defaults `--sdf-res` to `256`. To reduce physics cost for complex CAD assemblies, the manual STEP/STP workflow explicitly passes `--manual-sdf-resolution`, which defaults to `32`.

Before creating an SDF collider, the script reports boundary edges,
nonmanifold geometry, inconsistent winding, invalid faces, and near-zero
volume. `--approx sdf` remains `sdf`; diagnostics do not silently replace it.
The manual STEP/STP path enables SDF remeshing for common CAD tessellation
problems. Its result may be less accurate than an SDF from a clean closed mesh.
The low-level `--force-sdf` option is only needed to override another explicit
approximation.

## Mass

- `--set-mass` is total asset mass in kilograms.
- With multiple rigid bodies, total mass is distributed by geometric volume weight.
- When omitted, the script computes world-space mesh volume and multiplies it by material density.
- Invalid or tiny volume uses a safe fallback instead of writing negative or zero mass.

## Geometry Preparation

Before physics APIs are authored, `prepare_geometry_for_physics` runs:

```text
normalize_stage_units_to_meters
-> deinstance_visible_subtree
-> center_stage_geometry_at_origin       # manual CAD path
-> center_mesh_local_origins
-> fix_inverted_mesh_winding
```

`center_mesh_local_origins` shifts mesh points and compensates visual position with `xformOp:translate:meshLocalOrigin`.

This preparation runs before visual-material inference. Procedural MDLs may use
object-space coordinates, so changing units or local origins after material
comparison can move or rescale their patterns. All comparisons and final
renders therefore use the same normalized geometry. Authored dimensions are
preserved and no target-size input is required.

`fix_inverted_mesh_winding` first normalizes `leftHanded` meshes to `rightHanded`, then computes signed volume after applying the complete local-to-world transform. This catches both directly inverted meshes and meshes inverted by a mirrored parent transform. A negative closed volume reverses every face index order and clears stale authored normals before PhysX collision cooking.

Winding repair preserves the USD hierarchy and placement. If PhysX still
reports a cooking error after an SDF topology warning, inspect or repair that
source mesh.

## Final Collection

`collect_usd_flat.py` creates an isolated directory per physics USD, copies the main USD, SubUSDs, materials, and textures, and rewrites required relative resource paths. The result can be used directly as a simulation asset directory.

With reference-image materials enabled, validation runs again after collection.
It reopens and renders both the USD after material assignment and the collected
USD, checks every Mesh and `GeomSubset` binding, preserves NVIDIA MDL defaults,
and verifies that all MDL dependencies resolve inside the final directory. Any
failure stops the pipeline.
