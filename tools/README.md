# Tools

Standalone scripts called by `asset_pipeline.jobs` live here.

```text
tools/blender/  Scripts executed by Blender with `blender -b -P ...` or `--python ...`
tools/isaac/    Scripts executed by Isaac Sim Python with `python.sh ...`
tools/sam3d/    Headless SAM 3D Objects image segmentation and reconstruction wrapper
```

Main SAM3D tool:

```text
tools/sam3d/run_reconstruct.py  Prepared image(s) -> SAM3 masks -> single/multi-view GLB
```

`asset_pipeline.jobs.sam3d` prepares the input layout and launches this script with
the current `hunyuan_sam3d` Python. Single-view inference uses
`SAM3D_SINGLE_VIEW_ROOT`; multi-view inference delegates to
`SAM3D_MULTI_VIEW_ROOT`. Repository layout is documented in
[`tools/sam3d/README.md`](./sam3d/README.md); user-facing inputs and options are in
[`docs/modules/sam3d.md`](../docs/modules/sam3d.md) and
[`docs/modules/sam3d.zh.md`](../docs/modules/sam3d.zh.md).

Main Isaac tools:

```text
tools/isaac/convert_cad_to_usd.py  STEP/STP -> USD through the Omniverse CAD converter
tools/isaac/add_physics.py         Unit normalization, CAD geometry cleanup, collision, mass, materials
tools/isaac/collect_usd_flat.py    Final USD/material/texture collection
tools/isaac/isaac_sim_compat.py    Isaac Sim 5/6 SimulationApp import compatibility
```

The CAD path is `STEP/STP -> convert_cad_to_usd.py -> add_physics.py -> collect_usd_flat.py`.
`add_physics.py` keeps the asset root transform at zero for `--center-origin`, centers mesh-local origins with `xformOp:translate:meshLocalOrigin`, and fixes inverted local winding before PhysX collision cooking.

Keep package-internal workers, such as `asset_refiner/blender_worker.py`, inside their owning package.
