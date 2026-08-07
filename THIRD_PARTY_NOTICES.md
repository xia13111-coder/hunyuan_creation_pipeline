# Third-party notices

The repository's Apache License 2.0 covers first-party source and documentation
only. Each component below keeps its own license. Review the terms shipped with
the exact version you install.

## Included third-party source

| Component | Location | License boundary |
| --- | --- | --- |
| MVInverse / Pi3 | `tools/qwen_material_pipeline/third_party/mvinverse/` | The included `LICENSE` permits redistribution but limits use to non-commercial purposes. It is not covered by Apache-2.0. |
| DINOv2 source embedded by MVInverse | `tools/qwen_material_pipeline/third_party/mvinverse/mvinverse/models/dinov2/` | Meta copyright; Apache License 2.0. The source headers and `DINOV2_LICENSE` apply. |
| Naver position-embedding utility embedded by MVInverse | `tools/qwen_material_pipeline/third_party/mvinverse/mvinverse/models/layers/pos_embed.py` | Naver copyright; [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), non-commercial use only. |

The workflow requires an acknowledgement of this restriction. Commercial use
requires a replacement component or separate permission from the rights holder.

## Required or optional external components

These components are not distributed in the source archive:

| Component | Role | Upstream terms |
| --- | --- | --- |
| SAM 3 / SAM 3D Objects | segmentation and image-to-3D reconstruction | Meta SAM License and the license accompanying the exact SAM 3D release: <https://github.com/facebookresearch/sam3> and <https://github.com/facebookresearch/sam-3d-objects> |
| Qwen checkpoints | visual-language inference | License and model card shipped with the exact checkpoint: <https://huggingface.co/Qwen> |
| SigLIP2 checkpoints | image/text retrieval | License and model card shipped with the exact Google checkpoint: <https://huggingface.co/google> |
| DINOv2 | dense visual features | Apache License 2.0 for the official code and weights: <https://github.com/facebookresearch/dinov2> |
| NVIDIA Isaac Sim | USD/CAD/physics/render runtime | Isaac Sim source and additional software/assets have different terms: <https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses-isaac-sim.html> |
| NVIDIA MDL and Base materials | material candidates and rendering | NVIDIA additional software and materials terms; assets are mounted from a user's installation and are not redistributed here: <https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html> |
| Blender | mesh conversion and baking | GNU General Public License; Blender runs as an external executable: <https://www.blender.org/about/license/> |
| Tencent Hunyuan services and SDKs | generation and ReduceFace | Tencent Cloud service, API, SDK, and generated-content terms applicable to the user's account: <https://www.tencentcloud.com/document/product/301/17315> |

Model terms may vary by checkpoint, revision, and region. Record the exact
source, revision, hashes, and accepted license. An open-source client does not
grant rights to redistribute model weights or service output.

## Python and JavaScript dependencies

Packages installed through `environment.yml`, `requirements*.txt`,
`pyproject.toml`, or an app's lock file retain their own licenses. Before
redistributing a container or offline bundle, generate a dependency inventory
from the final environment and include all required license texts and notices.

Useful sources include `pip-licenses`, Conda package metadata, and the package
manager's production dependency report. Review CUDA, PyTorch, Kaolin,
Omniverse/Isaac Sim, and binary wheels separately.

## Generated assets

The license of a generated GLB, USD, texture, render, or evidence report depends
on its inputs, service terms, models, and material dependencies. Generated files
are not automatically licensed under Apache-2.0 merely because this pipeline
created them.
