# SAM3D Tools

`run_reconstruct.py` is the stable wrapper used by `asset_pipeline.jobs.sam3d`.
The two upstream repositories are kept as sibling directories:

```text
tools/sam3d/
├── run_reconstruct.py
└── third_party/
    ├── sam-3d-objects/
    └── sam-3d-objects-multiview/
```

They are not overlaid because both projects contain modules with the same names,
while the multi-view project also refers to the single-view checkout through
`../sam-3d-objects`. The sibling layout preserves that upstream contract.

Activate the only supported environment before running the wrapper:

```bash
conda activate hunyuan_sam3d
python ./tools/sam3d/run_reconstruct.py --help
```

The defaults point to the two directories above. `SAM3D_SINGLE_VIEW_ROOT`,
`SAM3D_MULTI_VIEW_ROOT`, and `SAM3_CHECKPOINT` remain available when an external
checkout or checkpoint is required. `third_party/` is intentionally ignored by the
main Git repository and Docker build context because it contains upstream source,
model weights, and generated native extensions. Docker runs must bind-mount the
project directory so these local dependencies are visible in the container.
