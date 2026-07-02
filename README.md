# Hunyuan Creation Pipeline

A production-oriented asset pipeline for Tencent Hunyuan 3D outputs and existing GLB models. It can generate GLB assets, refine Hunyuan meshes, convert GLB to USD, author simulation physics, and collect final USD assets for downstream use.

## Related Docs

- [`CALL_GRAPH.md`](./CALL_GRAPH.md): main function call graph.
- [`configs/README.md`](./configs/README.md): refine mesh config guide.
- [`README.docker.md`](./README.docker.md): Docker and HTTP API usage.

## Pipeline Diagram

```mermaid
flowchart TD
    subgraph INPUTS["Inputs"]
        IMG["Image folder\n--input-dir"]
        URL["Image URL\n--image-url"]
        PROMPT["Text prompt\n--prompt"]
        EXISTING["Existing Hunyuan GLB\n--existing-glb"]
        MANUAL["Manual or third-party GLB\n--manual-glb"]
    end

    subgraph HUNYUAN["Hunyuan generation"]
        GEN["hunyuan_to3d_batch.py\nGenerate raw GLB"]
        RAW["./downloads/\nRaw Hunyuan GLB"]
    end

    subgraph REFINE["Hunyuan refine mesh"]
        REFINE_STEP["asset_refiner\nHunyuan ReduceFace + local Blender postprocess\nConfig: ./configs/hunyuan_reduce_local_postprocess.yaml"]
        REFINED["./downloads_refined_mesh/\nRefined GLB"]
    end

    subgraph BLENDER["Blender post-processing"]
        ALIGN["align_glb_axis_only.py\nAxis mapping"]
        RESIZE["resize_glb_xyz_and_center.py\nResize and center"]
        CONVERT["convert_glb_to_usd_zup.py\nGLB to USD\nSet USD upAxis = Z"]
    end

    subgraph PHYSICS["Isaac Sim physics authoring"]
        MATERIAL["materials.json\nMaterial preset"]
        PHYSICS_STEP["add_physics.py\nCollision, rigid body, mass"]
        INTERMEDIATE["./output_intermediate/\nPhysics-authored USD"]
    end

    subgraph COLLECT["Final collection"]
        COLLECT_STEP["collect_usd_flat.py\nCollect USD, materials, textures"]
        FINAL["./output_final/\nFinal USD assets"]
        SUMMARY["./pipeline_result.json\nPipeline summary"]
    end

    IMG --> GEN
    URL --> GEN
    PROMPT --> GEN
    GEN --> RAW
    RAW --> REFINE_STEP
    EXISTING --> REFINE_STEP
    REFINE_STEP --> REFINED
    REFINED --> ALIGN
    ALIGN --> RESIZE
    RESIZE --> CONVERT

    MANUAL --> MANUAL_PREP["Manual preprocessing\nDefault: preserve geometry\nOptional: --manual-align / --manual-resize"]
    MANUAL_PREP --> CONVERT

    CONVERT --> PHYSICS_STEP
    MATERIAL --> PHYSICS_STEP
    PHYSICS_STEP --> INTERMEDIATE
    INTERMEDIATE --> COLLECT_STEP
    COLLECT_STEP --> FINAL
    COLLECT_STEP --> SUMMARY
```

## Features

- Generate GLB assets through Tencent Hunyuan 3D from images, image URLs, or text prompts.
- Refine Hunyuan-generated or Hunyuan-style GLB files through Tencent Hunyuan ReduceFace, local Blender UV unwrap, and nearest-surface texture migration.
- Convert GLB files to USD and write Z-up USD stage metadata.
- Author physics materials, collision, rigid bodies, and mass through Isaac Sim.
- Collect final USD assets, materials, and textures into one output directory.
- Run the back half of the pipeline for manually modeled or third-party GLB assets.

## Pipeline Modes

### Hunyuan Generation

```text
image / image URL / prompt
-> Hunyuan GLB generation
-> Hunyuan ReduceFace refine mesh
-> local Blender UV and texture migration
-> axis alignment and resize
-> GLB to USD
-> add physics
-> collect final USD
```

### Existing Hunyuan GLB

```text
existing GLB
-> Hunyuan ReduceFace refine mesh
-> local Blender UV and texture migration
-> axis alignment and resize
-> GLB to USD
-> add physics
-> collect final USD
```

### Manual GLB

```text
existing GLB
-> GLB to USD
-> add physics
-> collect final USD
```

Manual GLB mode skips Hunyuan generation and refine mesh. By default, it preserves the authored geometry orientation and only writes USD `upAxis = "Z"` metadata.

## Refine Mesh Path

The production refine path is intentionally narrow:

```text
source GLB
-> Tencent Hunyuan SubmitReduceFaceJob
-> download hunyuan_reduce_target.glb
-> asset_refiner/blender_worker.py
   -> import original source as high-reference surface
   -> import ReduceFace result as external retopology target
   -> align, clean, shrinkwrap project, and fix normals
   -> generate UVs
   -> migrate PBR textures by nearest-surface sampling
   -> export refined_asset.glb and qc_report.json
```

The default refine config is `./configs/hunyuan_reduce_local_postprocess.yaml`.
`pipeline_runner.py` passes `--hunyuan-local-postprocess` to `asset_refiner`, so local Blender postprocess is part of the normal pipeline.

Large local GLBs need a temporary public URL for the Hunyuan API to download. The pipeline uses `REFINE_MESH_TEMP_UPLOAD=uguu` by default. Override it with `--refine-temp-upload PROVIDER`, or pass `--refine-temp-upload none` to disable temporary upload when the input is already reachable by Hunyuan.

## Physics Authoring Path

After GLB files are converted to USD, `pipeline_runner.py` calls `add_physics.py` with Isaac Sim Python:

```text
USD input
-> add_physics.py
   -> open USD stage
   -> set stage upAxis = Z
   -> find mesh or Xform bodies
   -> add rigid body and collision APIs
   -> write physics material from materials.json
   -> assign or estimate mass
   -> export *_phys.usd
-> collect_usd_flat.py
   -> collect USD, materials, and textures into the final output directory
```

`add_physics.py` writes collision, rigid body, mass, friction, dynamic friction, and restitution data. If `--set-mass` is provided, it is treated as the total asset mass and is distributed across rigid bodies by volume weight. If `--set-mass` is omitted, mass is estimated from material density and mesh volume.

Common physics options:

| Option | Description |
| --- | --- |
| `--material` | Material preset from `materials.json`, such as `plastic`, `steel`, `rubber`, `wood`, or `copper`. |
| `--approx` | Collision approximation, such as `sdf`, `convexHull`, `convexDecomposition`, or `triangleMesh`. Dynamic rigid bodies using static-style mesh approximations are converted to `sdf`. |
| `--set-mass` | Total asset mass in kilograms. Omit it to estimate mass from density and volume. |

Standalone example:

```bash
/home/user/isaacsim500/python.sh ./add_physics.py \
  --folder ./downloads_refined_mesh/postprocess_glbs \
  --material-file ./materials.json \
  --out-dir ./output_intermediate \
  --headless \
  --material plastic \
  --approx sdf
```

## Requirements

- Linux environment with NVIDIA GPU support for Isaac Sim workflows.
- Conda environment created from `./environment.yml`.
- Blender.
- Isaac Sim.
- Tencent Cloud credentials for Hunyuan generation or refine mesh.

The runner auto-detects common Blender and Isaac Sim locations, including `~/isaacsim500/python.sh`, `/isaac-sim/python.sh`, and `/opt/isaac-sim/python.sh`. If auto-detection fails, set the paths explicitly:

```bash
export BLENDER_BIN="blender"
export ISAAC_PYTHON="/home/user/isaacsim500/python.sh"
```

## Setup

Create the conda environment:

```bash
conda env create -f ./environment.yml
conda activate hunyuan
```

Update an existing environment:

```bash
conda env update -f ./environment.yml --prune
```

Set Tencent Cloud credentials when using Hunyuan generation or refine mesh:

```bash
export TENCENTCLOUD_SECRET_ID="your-secret-id"
export TENCENTCLOUD_SECRET_KEY="your-secret-key"
```

Do not commit real credentials.

## Usage

### 1. Full Hunyuan Pipeline

Place input images in `./data/`, then run:

```bash
python ./run_asset_pipeline.py \
  --input-dir ./data \
  --output-dir ./downloads \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --result-json ./pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --face-count 150000 \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --orientation "X=L,Y=M,Z=S" \
  --material plastic \
  --approx sdf
```

### 2. Existing Hunyuan GLB

```bash
python ./run_asset_pipeline.py \
  --existing-glb ./downloads/example_asset \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --result-json ./pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --material plastic \
  --approx sdf
```

### 3. Manual GLB

```bash
python ./run_asset_pipeline.py \
  --manual-glb ./input/manual_asset.glb \
  --intermediate-output-dir ./manual_output_intermediate \
  --final-output-dir ./manual_output_final \
  --result-json ./manual_pipeline_result.json \
  --material steel \
  --approx sdf \
  --set-mass 30
```

If the manual model also needs the pipeline axis mapping or resize step, add:

```bash
  --manual-align \
  --manual-resize \
  --len-x 0.6 \
  --len-y 0.4 \
  --len-z 0.5
```

## Key Options

| Option | Description |
| --- | --- |
| `--input-dir` | Input image directory for Hunyuan generation. |
| `--prompt` | Text prompt for Hunyuan text-to-3D generation. |
| `--image-url` | Image URL for Hunyuan image-to-3D generation. |
| `--existing-glb` | Existing Hunyuan GLB path; still runs refine mesh. |
| `--manual-glb` | Manual or third-party GLB path; skips Hunyuan and refine mesh. |
| `--refine-config-path` | Refine mesh config path. Defaults to `./configs/hunyuan_reduce_local_postprocess.yaml`. |
| `--refine-temp-upload` | Temporary upload provider for local GLBs. Defaults to `REFINE_MESH_TEMP_UPLOAD` or `uguu`; use `none` to disable. |
| `--refine-output-dir` | Refine mesh working/output directory. |
| `--refine-fail-on-qc-error` | Return failure when refine QC status is `fail`. |
| `--len-x`, `--len-y`, `--len-z` | Target size in meters. |
| `--orientation` | Axis mapping consumed by `align_glb_axis_only.py`. |
| `--material` | Material name from `./materials.json`. |
| `--approx` | Collision approximation, such as `sdf`, `convexHull`, or `triangleMesh`. |
| `--set-mass` | Total asset mass in kilograms. |
| `--skip-refine` | Skip refine mesh for existing GLB workflows. |
| `--usd-format` | USD output format, such as `usd` or `usda`. |

Print the full CLI reference:

```bash
python ./run_asset_pipeline.py --help
```

## Coordinate And Mass Conventions

- Output USD stages are authored with `upAxis = "Z"`.
- Manual GLB mode preserves source geometry orientation by default.
- `--set-mass` means total asset mass, not the mass of one individual mesh.
- If an asset contains multiple rigid bodies, the total mass is distributed by volume weight.
- If `--set-mass` is omitted, mass is estimated from material density and mesh volume.

## Outputs

```text
./downloads/                  Raw Hunyuan generation results
./downloads_refined_mesh/      Refine reports, Hunyuan targets, textures, and refined GLBs
./output_intermediate/         USD files after physics authoring
./output_final/                Final collected USD assets
./pipeline_result.json         Machine-readable pipeline summary
```

Generated assets, caches, logs, and local environment files are ignored by git.

## Project Layout

```text
./run_asset_pipeline.py          Main runner
./pipeline_runner.py             Pipeline orchestration
./CALL_GRAPH.md                  Main function call graph
./hunyuan_to3d_batch.py          Hunyuan generation client
./asset_refiner/                 Refine mesh package
./asset_refiner/hunyuan_backend.py  Hunyuan ReduceFace backend
./asset_refiner/blender_worker.py   Local Blender retopology, UV, texture migration, and QC
./align_glb_axis_only.py         GLB axis mapping
./resize_glb_xyz_and_center.py   GLB resize and centering
./convert_glb_to_usd_zup.py      GLB to USD conversion
./add_physics.py                 Isaac Sim physics authoring
./collect_usd_flat.py            Final USD collection
./configs/                       Refine mesh configs
./configs/README.md              Refine mesh config guide
./materials.json                 Physics material presets
```

Docker and HTTP API usage are documented in [`README.docker.md`](./README.docker.md).
