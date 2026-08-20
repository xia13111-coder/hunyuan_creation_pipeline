# Reference-image materials for STEP/STP

[English](./visual-materials.md) | [中文](./visual-materials.zh.md) | [Quick start](../manual-part-id-materials.md)

This workflow assigns NVIDIA Base MDLs to a hand-authored STEP/STP assembly
from 2–4 photographs. CAD continues to define geometry, hierarchy,
and dimensions; photographs provide appearance evidence for visible Part-IDs.

## Flow

```text
STEP/STP conversion
-> physics geometry preparation
-> part index and reference renders
-> camera registration
-> evidence for each visible Part-ID
-> NVIDIA Base retrieval, Qwen ranking, and MDL render comparison
-> apply one material to every Mesh
-> collect the final USD and validate it
```

The delivered CAD is never resized, rotated, or deformed to match a photograph.
Camera transforms used during analysis do not modify the USD.

### Camera search in the production material path

`calibrate-cameras` can accelerate hypothesis search by loading the complete
CAD once in Kaolin's CUDA rasterizer and generating Part-ID, silhouette, and
occlusion buffers. It never moves an individual Mesh or uses material and
lighting appearance. The highest-ranked candidates are still rendered at final
resolution by Isaac Sim.

`calibrate-cameras --fast-search` accepts `auto` (the default, with fallback
when the optional runtime is unavailable), `required` (fail if the fast backend
is unavailable), and `disabled` (the legacy all-Isaac candidate search). The
report binds asset hashes, triangle and candidate counts, timings, and every
fast candidate to its full-resolution Isaac verification under
`candidate_search`.

The production material profiles set `camera_fast_search: disabled` and use
the complete Isaac candidate search. Material identity depends on every
registered reference view surviving camera and spatial registration; losing a
view turns visible small parts into false hidden-part fallbacks. Fast search
therefore remains available for explicit experiments, while the default
material path prioritizes complete four-view evidence.

Global camera seeds and per-phase candidates use the same geometry-gate
ordering: candidates that require out-of-contract 2-D scale, rotation, or
translation are rejected before silhouette IoU, boundary residual, and
structure scores are compared. Camera calibration and spatial Part-ID
projection share the sealed whole-workpiece foreground from the reference
manifest, and cross-resolution comparisons remove raster-size scale and
translation before enforcing physical residual limits. These rules contain no
asset name, camera-angle, or Part-ID special cases.

Each camera result also seals two path-independent content fingerprints. The
input fingerprint covers CAD geometry, actual Part-ID pixels, reference
image/mask pixels, and search configuration; the solution fingerprint covers
the final projection model, intrinsics, and extrinsics. Exact metric ties first
prefer the camera requiring the least whole-asset 2-D residual correction, then
use canonical camera parameters, never candidate generation order or temporary
view IDs. High-resolution seeding, checkpoint reuse, and downstream material
evidence verify current file hashes, objective versions, and the two-pass seed
hash. Changed inputs trigger explicit recalibration instead of silently mixing
old cameras, masks, or Part-ID evidence into a new run.

`asset_pipeline/visual_materials/` coordinates the workflow.
`tools/qwen_material_pipeline/` provides segmentation, evidence extraction,
retrieval, model calls, and USD workers. Isaac Sim handles CAD/USD, MDL
rendering, physics, and final validation. See [Architecture](../architecture.md).

## How evidence is built

The SAM3 UI records the accepted whole-workpiece mask for each photograph. The
run verifies the image hashes, then estimates one analysis camera per view.

Each CAD Part-ID is projected separately. The global camera supplies its modal
visible region and initial box. With that same sealed camera, the workflow also
projects the target mesh in isolation to obtain its complete amodal shape.
The same CAD request guides both SAM3 and class-agnostic EntitySeg with the
location/visibility template and isolated-mesh shape template. Neither model
mask is copied directly into the final evidence. Safe SAM3/EntitySeg estimates
initialize an iterative current-view optimizer: the isolated mesh constrains
the complete shape, the assembled Part-ID projection owns visibility and
occlusion, and photograph edges refine the boundary inside an automatically
scaled narrow band. The safest iterate maximizes the unweighted geometric mean
of those three agreements. CAD Part-ID remains the identity authority. Every Part-ID in a view shares one
whole-workpiece camera residual; no individual mesh may translate, rotate, or
scale. If local refinement is unreliable, the workflow uses the
global projection or marks the part as unobserved. These adjustments affect
evidence extraction only, never CAD geometry.
Production SAM3 and EntitySeg entry points use one fixed inference seed and seal
that seed together with request and model hashes. A retry with identical inputs
therefore cannot silently change Part-ID evidence through random initialization.

The identity-first path requires every registered reference view to contribute
real visible and selected Part-ID observations. It checks both per-part rows and
summary counts. A view rejected by camera or spatial registration stops the
run instead of silently continuing with a two-view material decision.

| Component | Role |
| --- | --- |
| SAM3 | Whole-workpiece foreground and per-part initialization candidates. |
| EntitySeg / CropFormer | Class-agnostic initialization candidates accepted only inside the CAD safety contract. |
| Iterative boundary optimizer | Combines isolated shape, assembled visibility, prior masks, and current-image edges; it never moves an individual mesh. |
| MVInverse | Albedo, roughness, and metallic estimates inside accepted masks. |
| SigLIP2 | Retrieve visually related MDLs from NVIDIA `Materials/Base`. |
| DINOv2 | Compare local surface and texture appearance. |
| Qwen3.5 | Rank a bounded candidate list using the available evidence. |
| MDL render comparison | Compare the actual candidate MDLs in registered CAD views. |

Models only rank catalog entries; they do not directly edit USD bindings.

Material identity and colour are separate stages. A colour-confirmed exact
library preset is retained directly. Otherwise each Part ID receives an
independent, colour-blind corresponding-material decision whose shortlist must
contain a physically compatible MDL with a reviewed colour interface. A
component vote is authoritative only with a strict majority; ties and weak
consensus are resolved from the intersection of independently ranked member
candidates, and an empty intersection fails closed. After identity is fixed,
actual-CAD renders select reviewed colour parameters and enforce local quality
for both the component and every sufficiently visible member, so small defects
cannot disappear inside a whole-view average.

## Assignment and validation rules

- Every Mesh receives exactly one applicable visual-material assignment.
- A visible part uses only evidence accepted for that Part-ID.
- Hidden or unresolved parts use the configured default material; the
  workflow does not invent photo evidence.
- Duplicate, incomplete, or review-only assignments stop the run before the
  material is applied.
- Once the winning MDL is recorded, its identity cannot change. Only a
  corresponding-material assignment may author reviewed colour inputs; exact
  presets, roughness, metallic value, textures, and normals remain unchanged.
- The USD after material assignment is rendered against usable references.
  After dependency collection, the delivered USD is reopened and checked
  again. A failed check stops the pipeline and reports the corresponding stage.

The default configuration is
`tools/qwen_material_pipeline/configs/pipeline/manual_part_id_materials.json`.
It searches only NVIDIA `Materials/Base` and makes decisions independently per
CAD Part-ID.

## Runtime and modes

| Runtime | Work |
| --- | --- |
| `hunyuan_sam3d` | Orchestration, SAM3, MVInverse, SigLIP2, and DINOv2. |
| Qwen3.5 Python environment | Local vision-language inference. |
| Isaac Sim Python | CAD/USD, MDL rendering, physics, and validation. |
| Blender | Blender-owned mesh and texture operations. |

The root `.env` centralizes local models:

- Qwen / Qwen3.5: `QWEN_MODEL_PATH`, `QWEN35_MODEL_PATH`;
- MVInverse: `MVINVERSE_REPOSITORY`, `MVINVERSE_CHECKPOINT`;
- SAM3: `SAM3_REPOSITORY`, `SAM3_CHECKPOINT`;
- EntitySeg: `ENTITYSEG_PYTHON`, `ENTITYSEG_CROPFORMER_ROOT`, `ENTITYSEG_CONFIG`, `ENTITYSEG_CHECKPOINT`;
- SAM3D MoGe / DINOv2: `SAM3D_MOGE_CHECKPOINT`,
  `SAM3D_DINOV2_REPOSITORY`, `SAM3D_DINOV2_CHECKPOINT`;
- material retrieval: `SIGLIP2_MODEL_PATH`, `DINOV2_MODEL_PATH`.

The runtime fixes `PIPELINE_LOCAL_MODELS_ONLY=1`; normal inference never downloads weights;
a missing or incomplete local path fails clearly. `VISUAL_MATERIAL_ROOT`,
`NVIDIA_BASE_OBSERVATION_BANK`, and `ISAAC_PYTHON` are also read from `.env`.
Weights and NVIDIA assets are not included in the repository. Hunyuan cloud
APIs are the exception and still require network access when used.

The EntitySeg interpreter is deliberately isolated from user-site packages.
After installing the external CropFormer/Detectron2 runtime, install the
pipeline compatibility layer into the same interpreter:

```bash
"$ENTITYSEG_PYTHON" -m pip install \
  -r tools/qwen_material_pipeline/requirements-entityseg.txt
PYTHONNOUSERSITE=1 "$ENTITYSEG_PYTHON" -c \
  'import black, cloudpickle, mmcv, yapf'
```

The YAPF pin is required by MMCV 1.x. A successful import only from
`~/.local` is not sufficient because pipeline subprocesses set
`PYTHONNOUSERSITE=1` for reproducibility.

| Mode | Use |
| --- | --- |
| `live` | Run inference for a new workpiece. |
| `auto` | Reuse an exactly matching recorded result; otherwise run generic inference. |
| `bundled` | Require an exact recorded-result match and stop if none exists. |

`--resume` continues the same hash-verified run. Use a new output directory to
start from zero.

The annotation JSON always identifies the reference images. Its manually
accepted foreground masks are used only by `live`; `auto` and `bundled` use the
recorded project's own data when a match exists.

## Troubleshooting

Find the first failed `[PROGRESS]` stage, then inspect its output:

| Symptom | Inspect |
| --- | --- |
| Incomplete foreground | SAM3 annotations and masks. |
| CAD and photograph do not overlap | Camera-registration overlays. |
| A visible part uses the default material | `part_id_reference_evidence.json`. |
| Qwen output is rejected | `.raw`, `.parse.json`, and the schema error. |
| The expected MDL is not selected | Candidate retrieval and MDL render scores. |
| Final USD looks different | Recorded material selection and final render comparison. |

Outputs are under
`RUN_ROOT/visual_material/{renders,analysis,visual_quality,final_visual_acceptance}`.

Further reading:

- [Material package](../../tools/qwen_material_pipeline/README.md)
- [MVInverse (Chinese)](../../tools/qwen_material_pipeline/docs/mvinverse.zh.md)
- [Base observation bank (Chinese)](../../tools/qwen_material_pipeline/docs/base_material_observation_bank.zh.md)

MVInverse is non-commercial. NVIDIA assets and Isaac Sim components have their
own terms; see [Third-party notices](../../THIRD_PARTY_NOTICES.md).
