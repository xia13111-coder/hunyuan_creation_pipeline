# Isaac Sim Physics Module

[English](./physics.md) | [中文](./physics.zh.md) | [Documentation index](../README.md)

This module authors collision, rigid bodies, physics materials, and mass on USD, then collects the USD and its material/texture dependencies. Visual PBR and PhysX materials are separate data systems.

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

`--material` explicitly selects a preset from `materials.json`. The code no longer guesses a physics material from prim or visual-material names.

| Field | Unit/type | Purpose |
| --- | --- | --- |
| `density` | kg/m^3 | Estimate mass as volume times density when `--set-mass` is omitted. |
| `friction` | coefficient | Static friction. |
| `dyn_ratio` | ratio | Dynamic friction is `friction * dyn_ratio`; default is `0.9`. |
| `restitution` | coefficient | Bounciness. `0` is close to no rebound. |
| `combine` | string | Optional friction/restitution combine mode, such as `average`. |

Current presets:

| Material | Density | Static friction | Dynamic friction | Restitution |
| --- | ---: | ---: | ---: | ---: |
| `plastic` | 950 | 0.30 | 0.27 | 0.10 |
| `steel` | 7850 | 0.50 | 0.45 | 0.05 |
| `rubber` | 1100 | 1.00 | 0.70 | 0.60 |
| `wood` | 700 | 0.60 | 0.54 | 0.20 |
| `copper` | 8960 | 0.40 | 0.36 | 0.05 |

## Collision Approximation

| Mode | Best use | Tradeoff |
| --- | --- | --- |
| `sdf` | Dynamic complex/concave meshes; current recommended default. | Good concave fidelity with higher cooking cost. Manual CAD also enables SDF remeshing. |
| `convexHull` | Simple, mostly convex objects. | Fast and stable, but fills cavities. |
| `convexDecomposition` | Concave approximation without SDF. | Multiple hulls improve fidelity at higher compute/collision cost. |
| `boundingCube` | Coarse box proxy. | Fastest and least accurate. Aliases: `box`, `cube`. |
| `boundingSphere` | Round objects. | Fast and stable, inaccurate for elongated assets. Alias: `sphere`. |
| `sphereApproximation` | Assets that accept a PhysX sphere approximation. | Simple proxy. |
| `triangleMesh` | Static environments. | Automatically changed to `sdf` for dynamic rigid bodies. |
| `meshSimplification` | Static simplified meshes. | Automatically changed to `sdf` for dynamic rigid bodies. |

The main pipeline selects a mode with `--approx`. Direct `add_physics.py` usage also exposes SDF, VHACD, contact-offset, and solver-iteration options.

The direct script defaults `--sdf-res` to `256`. To reduce physics cost for complex CAD assemblies, the manual STEP/STP workflow explicitly passes `--manual-sdf-resolution`, which defaults to `32`.

Before authoring an SDF collider, the script checks mesh topology. Boundary edges, nonmanifold edges, inconsistent shared-edge winding, invalid faces, or near-zero closed volume produce a warning because PhysX cooking may fail or produce a poor collider.

`--approx` is strict: when `--approx sdf` is requested, the authored `physics:approximation` remains `sdf`; topology diagnostics never replace it with `convexDecomposition`. The direct-script option `--force-sdf` is only needed to override a different approximation supplied by rules or other command-line settings.

The manual STEP/STP workflow additionally passes `--sdf-remesh`. It authors `physxSDFMeshCollision:sdfEnableRemeshing = true`, allowing PhysX to rebuild problematic CAD tessellation before computing the SDF. This is intended for inconsistent winding, open shells, and self-intersections; the resulting collision surface can be slightly less accurate than an SDF cooked from a clean closed mesh.

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

`fix_inverted_mesh_winding` first normalizes `leftHanded` meshes to `rightHanded`, then computes signed volume after applying the complete local-to-world transform. This catches both directly inverted meshes and meshes inverted by a mirrored parent transform. A negative closed volume reverses every face index order and clears stale authored normals before PhysX collision cooking.

Typical diagnostic logs are:

```text
fixed inverted world-space mesh winding: 3 mesh(es)
[WARN] SDF topology warning on /Asset/Part/Mesh: boundary=12, nonmanifold=0, inconsistent=2, invalid_faces=0; keeping requested sdf
SDF remeshing enabled on /Asset/Part/Mesh
```

The first line means winding was repaired while retaining the USD hierarchy and placement. The second means the requested SDF was retained, but the mesh should be inspected if PhysX reports a cooking error.

## Final Collection

`collect_usd_flat.py` creates an isolated directory per physics USD, copies the main USD, SubUSDs, materials, and textures, and rewrites required relative resource paths. The result can be used directly as a simulation asset directory.
