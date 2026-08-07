# SAM3D Tools

`run_reconstruct.py` is the entry point used by `asset_pipeline.jobs.sam3d`.
The two upstream repositories remain side by side:

```text
tools/sam3d/
├── run_reconstruct.py
└── third_party/
    ├── sam-3d-objects/
    └── sam-3d-objects-multiview/
```

Both projects contain modules with the same names, and the multi-view project
expects the single-view checkout at `../sam-3d-objects`. Do not merge them.

Activate the supported environment before running the wrapper:

```bash
conda activate hunyuan_sam3d
python ./tools/sam3d/run_reconstruct.py --help
```

Local paths can be set in the project `.env`:

```dotenv
SAM3D_SINGLE_VIEW_ROOT=/absolute/path/to/sam-3d-objects
SAM3D_MULTI_VIEW_ROOT=/absolute/path/to/sam-3d-objects-multiview
SAM3D_PIPELINE_CONFIG=/absolute/path/to/pipeline.yaml
SAM3_REPOSITORY=/absolute/path/to/sam3
SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
SAM3D_MOGE_CHECKPOINT=/absolute/path/to/moge-vitl/model.pt
SAM3D_DINOV2_REPOSITORY=/absolute/path/to/facebookresearch_dinov2_main
SAM3D_DINOV2_CHECKPOINT=/absolute/path/to/dinov2_vitl14_reg4_pretrain.pth
```

Normal reconstruction is local-only. Missing or incomplete weights fail before
inference; no model hub download is attempted. The wrapper builds a temporary
absolute-path configuration overlay and leaves the upstream `pipeline.yaml`
unchanged. Hunyuan generation and ReduceFace remain cloud APIs and are not
covered by this local-weight guarantee.

`third_party/` is ignored by Git and the Docker build context. Docker runs must
mount the upstream source and weights. See the
[Docker operations guide](../../docker/README.md) and the
[SAM3D module guide](../../docs/modules/sam3d.md).
