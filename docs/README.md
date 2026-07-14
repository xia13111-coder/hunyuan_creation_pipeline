# Documentation Index

[English](./README.md) | [中文](./README.zh.md) | [Back to project README](../README.md)

Documentation follows code ownership. The root README contains entry points and quick commands; detailed behavior and options live with each module guide.

| Module | Code owner | Guide |
| --- | --- | --- |
| Generation method selection | Hunyuan and SAM3D input paths | [Hunyuan vs. SAM3D](./generation-guide.md) |
| Overall architecture | `asset_pipeline/` | [Architecture and call graph](./architecture.md) |
| Raw Hunyuan generation | `asset_pipeline/hunyuan_generation.py`, `jobs/hunyuan.py` | [Hunyuan](./modules/hunyuan.md) |
| SAM3D reconstruction | `jobs/sam3d.py`, `tools/sam3d/` | [SAM3D](./modules/sam3d.md) |
| Mesh refine | `jobs/refine.py`, `asset_refiner/` | [Refine](./modules/refine.md) |
| Blender postprocess | `jobs/blender.py`, `tools/blender/` | [Blender](./modules/blender.md) |
| Isaac Sim physics | `jobs/isaac.py`, `tools/isaac/` | [Physics](./modules/physics.md) |
| Manual CAD | `workflows.run_stp_physics_job`, `convert_cad_to_usd.py` | [CAD](./modules/cad.md) |
| HTTP API / Docker | `asset_pipeline/api.py`, `docker/` | [API](./modules/api.md) |

Additional references:

- [Refine configuration](../configs/README.md)
- [Standalone tools](../tools/README.md)
- [Chinese project README](../README.zh.md)
