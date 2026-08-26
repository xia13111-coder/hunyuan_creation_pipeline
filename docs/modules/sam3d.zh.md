# SAM 3D Objects 流程

[English](./sam3d.md) | [中文](./sam3d.zh.md) | [文档索引](../README.zh.md)

SAM3D 从一张或多张图片重建指定物体，再把原始 GLB 交给 Hunyuan 精修、Blender
后处理和 Isaac Sim 物理流程。选择生成方式前可先阅读
[Hunyuan 与 SAM3D 的区别](../guides/generation-guide.zh.md)。

## 流程

```text
输入图片
-> prepare_sam3d_input：筛选并转为 RGB PNG
-> run_reconstruct.py：SAM3 分割 + 单视角/多视角重建
-> 原始 GLB
-> Hunyuan ReduceFace 和本地 Blender 精修
-> Blender 对齐、缩放、居中和 USD 导出
-> Isaac Sim 物理和最终依赖收集
```

正常生产流程不要使用 `--skip-refine`。

## 本地权重与环境

```bash
conda activate hunyuan_sam3d
```

流程使用当前 Python 启动 SAM3D。项目会自动查找默认目录，也可在根目录
`.env` 中明确指定所有本地路径：

```dotenv
SAM3D_SINGLE_VIEW_ROOT=/absolute/path/to/sam-3d-objects
SAM3D_MULTI_VIEW_ROOT=/absolute/path/to/sam-3d-objects-multiview
SAM3D_PIPELINE_CONFIG=/absolute/path/to/checkpoints/pipeline.yaml
SAM3_REPOSITORY=/absolute/path/to/sam3
SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
SAM3D_MOGE_CHECKPOINT=/absolute/path/to/moge-vitl/model.pt
SAM3D_DINOV2_REPOSITORY=/absolute/path/to/facebookresearch_dinov2_main
SAM3D_DINOV2_CHECKPOINT=/absolute/path/to/dinov2_vitl14_reg4_pretrain.pth
```

正常 SAM3/SAM3D 推理会强制 Hugging Face 和 Transformers 离线。任一源码目录、
配置或权重缺失、为空或不完整时，任务立即失败，不会转为联网下载。

运行时会从上游 `pipeline.yaml` 生成一份临时本地配置覆盖层，写入 MoGe、
DINOv2 和 SAM3D checkpoint 的绝对路径。上游 YAML 不会被修改。

Hunyuan 生成和 ReduceFace 是腾讯云 API，不属于本地模型权重。因此 SAM3D
重建可完全离线，但后续 Hunyuan 精修仍需网络和云凭据。USD 与物理阶段还需
Isaac Sim Python。

## 输入规则

`--sam3d-input` 可以是一张图片，也可以是包含 `.png`、`.jpg`、`.jpeg`、`.webp` 或
`.bmp` 的目录。

- 目录中存在 `images/` 时，只读取该子目录；
- 不继续递归扫描更深目录；
- 文件按名称排序，忽略 `mask_*` 或 `*_mask`；
- 输入统一复制为 RGB PNG，原图的 alpha 通道不作为掩码；
- 所有视角使用同一个分割提示词，无需预先生成掩码。

多视角图片必须展示同一个物体。建议使用尺度接近、构图稳定、遮挡少且角度互补的图片。

## 命令

```bash
hunyuan-asset-pipeline \
  --sam3d-input ./data/sam3d_images \
  --sam3d-mode auto \
  --sam3d-prompt "metal shelves" \
  --sam3d-seed 42 \
  --sam3d-steps 50 \
  --output-dir ./outputs/sam3d_example/generation \
  --intermediate-output-dir ./outputs/sam3d_example/intermediate \
  --final-output-dir ./outputs/sam3d_example/final \
  --result-json ./outputs/sam3d_example/pipeline_result.json \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## SAM3D 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sam3d-input` | 无 | 图片或图片目录；不能与 `--sam3d-glb`、`--existing-glb`、`--manual-stp` 同时使用。 |
| `--sam3d-mode` | `auto` | `single`、`multi`，或按图片数量自动选择。 |
| `--sam3d-prompt` | 无 | 用于二维分割的物体名；使用 `--sam3d-input` 时必填。 |
| `--sam3d-confidence-threshold` | `0.5` | SAM3 最低定位分数；降低后误检会增加。 |
| `--sam3d-seed` | `42` | 重建随机种子。 |
| `--sam3d-steps` | `50` | 第一阶段采样步数；增加会变慢，也不代表面数。 |
| `--output-dir` | `./downloads` | 工作目录；已有同名任务时追加 `_2`、`_3` 等。 |
| `--sam3d-glb` | 无 | 从已有 SAM3D GLB 继续后续处理。 |

模式行为：

| 模式 | 行为 |
| --- | --- |
| `auto` | 一张有效图片使用 `single`，多张使用 `multi`。 |
| `single` | 只使用排序后的第一张图片。 |
| `multi` | 把全部有效图片交给多视角后端；至少应有两个视角。 |

`--sam3d-prompt` 应使用简短英文物体名，例如 `industrial control cabinet` 或
`red plastic crate`。它只用于选择图片中的目标，不描述三维结构、尺寸或物理属性。没有
有效掩码的视角会被跳过；所有视角都无有效掩码时任务失败。

比较提示词或图片组合时先把随机种子固定为 `42`。只有测试重建随机性时再更换。增加采样
步数不保证提高几何质量，也不控制 ReduceFace 或碰撞精度。

## 后续参数

| 参数 | 说明 |
| --- | --- |
| `--refine-config-path` | Hunyuan ReduceFace 和本地 Blender 精修配置。 |
| `--refine-temp-upload` | 让 Hunyuan 能访问本地 GLB 的临时文件服务。 |
| `--skip-refine` | 仅用于诊断，生产资产不建议使用。 |
| `--len-x/y/z` | 最终包围盒尺寸，单位 m。 |
| `--orientation` | 包围盒长、中、短边的坐标轴映射。 |
| `--approx` | Isaac Sim 碰撞近似。 |
| `--set-mass` | 最终总质量，单位 kg。 |

## 输出和继续运行

单视角目录包含 `image.png`，通常输出 `result_obj0.glb`；多视角目录包含 `images/`、
`masks/`，通常输出 `result.glb`。最终选中的 GLB 会记录在 `--result-json` 指定文件的
`generation.selected_glb` 字段中。

如果重建成功但后续阶段失败，可以从已有 GLB 继续：

```bash
hunyuan-asset-pipeline \
  --sam3d-glb ./outputs/sam3d_example/generation/sam3d/sam3d_images/result.glb \
  --intermediate-output-dir ./outputs/sam3d_resume/intermediate \
  --final-output-dir ./outputs/sam3d_resume/final \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## 常见问题

| 信息 | 处理方法 |
| --- | --- |
| `No module named 'sam3.model'` | 激活 `hunyuan_sam3d`，检查单视角项目中的 SAM3 子模块。 |
| `Local model path ... is missing` | 在 `.env` 中设置报错所指的本地路径；流程不会自动下载。 |
| `SAM3 segmentation produced no valid masks` | 使用直接物体名并检查目标可见性；确认低分掩码正确后再降低阈值。 |
| `Multi-view SAM3D script not found` | 检查默认目录，或设置 `SAM3D_MULTI_VIEW_ROOT`。 |
| `FailedOperation.RequestTimeout` | 等待 ReduceFace 自动重试，或用 `--sam3d-glb` 从已有结果继续。 |
| Pillow、spconv 或 AMP 警告 | 通常是兼容性警告；重建正常结束时不影响结果。 |
