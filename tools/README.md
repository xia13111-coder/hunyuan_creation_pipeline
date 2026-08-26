# Tools

Scripts that must run in Blender, Isaac Sim, or a model-specific environment live
here. Application orchestration stays in `asset_pipeline`.

```text
tools/blender/  Scripts executed by Blender with `blender -b -P ...` or `--python ...`
tools/isaac/    Scripts executed by Isaac Sim Python with `python.sh ...`
tools/sam3d/    Headless SAM 3D Objects image segmentation and reconstruction wrapper
tools/qwen_material_pipeline/  Independently installable Qwen + MVInverse material package
```

Main SAM3D entry point:

```text
tools/sam3d/run_reconstruct.py  Prepared image(s) -> SAM3 masks -> single/multi-view GLB
```

`asset_pipeline.jobs.sam3d` prepares the inputs and runs this script with the
current `hunyuan_sam3d` Python. Single-view inference uses
`SAM3D_SINGLE_VIEW_ROOT`; multi-view inference delegates to
`SAM3D_MULTI_VIEW_ROOT`. Repository layout is documented in
[`tools/sam3d/README.md`](./sam3d/README.md); user-facing inputs and options are
in [`docs/modules/sam3d.md`](../docs/modules/sam3d.md), with a
[Chinese version](../docs/modules/sam3d.zh.md).

Main Isaac tools:

```text
tools/isaac/convert_cad_to_usd.py  STEP/STP -> USD through the Omniverse CAD converter
tools/isaac/add_physics.py         Unit normalization, CAD geometry cleanup, collision, mass, materials
tools/isaac/collect_usd_flat.py    Final USD/material/texture collection
tools/isaac/isaac_sim_compat.py    Isaac Sim 5/6 SimulationApp import compatibility
```

Optional one-off operators are grouped below the runtime they require:

```text
tools/blender/utilities/create_hunyuan_upload_proxy.py
                               geometry-only proxy creation
tools/blender/diagnostics/render_topology_views.py
                               inspection and diagnostic renders
tools/isaac/utilities/apply_uniform_mdl.py
                               standalone uniform MDL binding
tools/isaac/utilities/group_meshes_as_compound.py
                               standalone compound-body authoring
```

The CAD path is:

```text
STEP/STP -> CAD-to-USD -> Physics normalization -> optional visual materials -> collection
```

`add_physics.py` also handles origin normalization and inverted mesh winding
before PhysX collision cooking.

Keep package-internal workers, such as `asset_refiner/blender_worker.py`, inside
the package responsible for them.

`tools/qwen_material_pipeline` is the only source location for the automatic
material package. Run it through an editable install or with `PYTHONPATH=./tools`;
saved evidence uses relative, hash-checked references instead of host paths. See
the
[release and portability rules](../docs/release/release-portability.md).

For interactive use, install both first-party packages in editable mode once:

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

This exposes `qwen-material`. The pipeline supplies the package path to Isaac Sim
and model subprocesses automatically.
