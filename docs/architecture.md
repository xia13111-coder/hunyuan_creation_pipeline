# Architecture and Call Graph

[English](./architecture.md) | [中文](./architecture.zh.md) | [Documentation index](./README.md)

This page describes module ownership and the main execution paths. User
commands and tuning options are documented in the relevant workflow guides.

## Layers

```text
CLI / HTTP API
    -> workflow orchestration
        -> jobs and visual-material stages
            -> shared subprocess runner
                -> Blender / Isaac Sim / model runtimes / cloud APIs
```

Dependencies point downward: jobs do not call the CLI, and runtime workers do
not coordinate complete workflows.

## Main modules

| Module | Responsibility |
| --- | --- |
| `asset_pipeline/cli.py` | Parse command-line options and select a workflow. |
| `asset_pipeline/api.py` | Run the same workflows as FastAPI background jobs. |
| `asset_pipeline/workflows.py` | Compose jobs for image and GLB inputs. |
| `asset_pipeline/manual_cad.py` | Compose the STEP/STP workflow. |
| `asset_pipeline/jobs/` | Run one conversion, refinement, physics, or delivery task. |
| `asset_pipeline/visual_materials/` | Coordinate reference-driven material assignment. |
| `asset_pipeline/command.py` | Log and execute subprocesses. |
| `asset_pipeline/runtime.py` | Resolve external runtimes and configuration. |
| `asset_pipeline/project_layout.py` | Resolve repository paths. |
| `asset_refiner/` | Refine meshes and transfer textures. |
| `tools/blender/` | Blender-only workers. |
| `tools/isaac/` | Isaac Sim-only workers. |
| `tools/sam3d/` | SAM3D reconstruction workers. |
| `tools/qwen_material_pipeline/` | Segmentation, evidence, retrieval, material selection, and USD tools. |

`pipeline_runner.py`, root `run_*.py` scripts, and
`asset_pipeline/jobs/material.py` remain compatibility entry points. New code
should import the module responsible for that behavior directly.

## Main command dispatch

The canonical reference-driven STEP/STP material command has one owner chain:

```text
manual-material-pipeline
-> manual_material_cli.main
-> manual_cad.run_manual_cad_workflow
-> visual_materials.run_assign_visual_materials_job
-> qwen_material_pipeline / Isaac Sim workers
```

`python -m asset_pipeline.manual_material_cli` is the equivalent module entry.
The general multi-input command remains responsible for the other workflows:

```text
run_asset_pipeline.py
-> asset_pipeline.cli.main
   -> runtime.configure_runtime
   -> select input workflow
```

```text
--manual-stp
-> manual_cad.run_manual_cad_workflow

--sam3d-input
-> workflows.run_sam3d_image_and_process_model_job

--sam3d-glb / --existing-glb
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job

image directory / image URL
-> workflows.run_generate_and_process_model_job
```

## Image and GLB workflows

Hunyuan generation:

```text
workflows.run_generate_and_process_model_job
-> jobs.hunyuan.run_generate_model_job
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job
```

SAM3D reconstruction:

```text
workflows.run_sam3d_image_and_process_model_job
-> jobs.sam3d.run_sam3d_image_job
   -> tools/sam3d/run_reconstruct.py
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job
```

Common GLB post-processing:

```text
workflows.run_process_model_job
-> Blender preflight, axis alignment, optional resize, and GLB-to-USD
-> optional reference-driven material assignment
-> Isaac Sim physics
-> USD collection
-> delivery validation
```

For refinement internals, see [Refine](./modules/refine.md). For Blender and
physics, see [Blender](./modules/blender.md) and
[Physics](./modules/physics.md).

## STEP/STP workflow

The hand-authored CAD path preserves source dimensions and accepts one STEP/STP
assembly per reference-image run.

```text
manual_cad.run_manual_cad_workflow
-> jobs.cad.run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> jobs.isaac.run_add_physics_job(center_origin=True)
-> optional visual_materials.run_assign_visual_materials_job
-> jobs.isaac.run_collect_job
-> optional delivery and final visual validation
```

Physics geometry preparation runs before material selection. This is important
for procedural MDLs whose visible pattern can depend on object-space scale and
origin. Camera alignment is used only to compare renders with photographs; it
does not rotate or resize the delivered CAD geometry.

The workflow intentionally has no `len_x`, `len_y`, `len_z`, or `orientation`
option. See [CAD](./modules/cad.md) and the
[CAD material quick start](./manual-part-id-materials.md).

## Visual-material workflow

Public entry points are exported from `asset_pipeline.visual_materials`.
`orchestrator.py` runs the stages; it does not implement the model algorithms.

```text
visual_materials.run_assign_visual_materials_job
-> VisualMaterialPipelineContext.create
-> VisualMaterialWorkspace.create
-> stages.source_preparation.prepare_source_evidence
   -> USD part index and instance expansion
   -> canonical RGB and Part-ID renders
   -> continuous 3D camera registration
-> stages.material_inference.run_material_inference
   -> local NVIDIA Materials/Base catalog
   -> SAM3 foreground evidence
   -> Qwen + MVInverse analysis
   -> SigLIP2 retrieval + DINOv2 reranking
-> stages.part_id_evidence.run_part_id_evidence_stage
   -> isolated CAD model-image Part-ID templates
   -> first-pass SAM3 and EntitySeg candidates
   -> leave-target-out assembly-neighbour localization
   -> second-pass segmentation and iterative hybrid fusion
-> per-Part-ID candidate selection and complete assignment plan
-> colour-blind material identity lock
-> reviewed colour calibration for corresponding-material assignments
-> preview USD after material assignment and render comparison
-> bounded candidate refinement when evidence permits
-> final MDL selection and selection record
-> final USD material layer
```

After collection:

```text
visual_materials.run_final_visual_acceptance_job
-> render collected USD
-> compare it with registered reference views
-> validate materials and external MDL dependencies
```

### Responsibilities inside `asset_pipeline/visual_materials`

- `context.py` holds validated run inputs and configuration.
- `workspace.py` defines output paths.
- `commands.py` builds subprocess arguments.
- `stages/runner.py` handles progress, subprocess failures, and bounded retries.
- `stages/source_preparation.py` prepares registries, renders, and camera data.
- `stages/material_inference.py` starts the material-analysis subprocess.
- `stages/part_id_evidence.py` owns model-image Part-ID localization, both
  segmentation passes, relation guidance, hybrid fusion, and all-view evidence
  coverage checks.
- `policy_exact_cover.py` ensures every mesh receives an applicable result.
- `exact_mdl_cache.py` verifies cached candidate renders.
- `tournaments.py` compares shortlisted MDLs by rendered appearance.
- `quality_contracts/` contains metrics, diagnostics, limited repair, and final
  result checks.
- `stages/final_acceptance.py` checks the collected deliverable.

### Model roles

- SAM3 supplies the confirmed whole-workpiece foreground. It may replay the
  user's point-based annotation.
- Qwen describes visible parts and chooses among bounded candidates.
- MVInverse estimates PBR appearance evidence; it does not decide the final MDL
  by itself.
- SigLIP2 retrieves visually related materials from the complete local NVIDIA
  `Materials/Base` catalog.
- DINOv2 reranks candidates using local surface appearance.
- CAD Part-ID renders connect photo evidence to individual meshes. Parts not
  visible in any accepted view receive the configured default material so the output
  still covers every mesh.

The final choice is made from rendered MDL candidates. Once the selection
record is created, later stages verify and apply that choice without silently
changing it. Exact thresholds and repair rules are documented in
[Visual materials](./modules/visual-materials.md).

## Runtime boundaries and resume

The top-level command runs in `hunyuan_sam3d`. Blender and Isaac Sim use their
own Python runtimes. Qwen may use the configured Python 3.11 environment; SAM3,
MVInverse, SigLIP2, and DINOv2 use the runtime recorded in the material
configuration.

GPU-heavy stages run sequentially to reduce peak memory use. Each reusable
visual stage records its inputs, model revision, configuration, and output
hashes. `--resume` is for the same `live` request; a reusable visual stage is
accepted only when those values still match. A failed model stage receives at
most one clean-process retry when its saved evidence is safe to reuse. Invalid
schemas, mismatched hashes, or insufficient visual evidence stop the workflow
and leave a diagnostic report under the run directory.
