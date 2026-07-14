# SAM 3D Objects Pipeline 使用说明

[English](./sam3d.md) | [中文](./sam3d.zh.md) | [文档索引](../README.zh.md)

与 Hunyuan 生成的输入和适用场景区别见 [Hunyuan 与 SAM3D 生成方式选择](../generation-guide.zh.md)。

本文说明如何把单张或多张图片交给 SAM 3D Objects，并让生成的 GLB 继续经过 Hunyuan refine、Blender 后处理和 Isaac Sim 物理挂载。

## 完整流程

```text
输入图片
-> asset_pipeline.jobs.sam3d.prepare_sam3d_input
   -> 筛选图片并转换为 RGB PNG
   -> single: image.png
   -> multi: images/00000.png, images/00001.png, ...
-> tools/sam3d/run_reconstruct.py
   -> SAM3 根据 --sam3d-prompt 为每张图片生成 mask
   -> single: SAM 3D Objects 单视角重建
   -> multi: SAM 3D Objects multi-view 重建
   -> 输出原始 GLB
-> asset_refiner
   -> Tencent Hunyuan ReduceFace
   -> Blender 低模投影和 UV 展开
-> Blender 坐标轴映射、尺寸缩放、居中和 GLB-to-USD
-> Isaac Sim collision、rigid body 和 mass
-> 收集最终 USD
```

正常生产路径不要使用 `--skip-refine`。Hunyuan refine 和本地 Blender 后处理负责准备低模和后续导出。

## 环境准备

Hunyuan、refine、API、CLI 和 SAM3D 统一使用同一个环境：

```bash
conda activate hunyuan_sam3d
```

pipeline 会直接使用当前进程的 `sys.executable` 启动 SAM3D，并在启动时验证环境名。
不再查找或维护第二套 Python 路径。

默认项目和权重位置：

```text
./tools/sam3d/third_party/sam-3d-objects/
./tools/sam3d/third_party/sam-3d-objects-multiview/
./tools/sam3d/third_party/sam-3d-objects/checkpoints/sam3.pt
```

放在其它位置时设置：

```bash
export SAM3D_SINGLE_VIEW_ROOT="/path/to/sam-3d-objects"
export SAM3D_MULTI_VIEW_ROOT="/path/to/sam-3d-objects-multiview"
export SAM3_CHECKPOINT="/path/to/sam3.pt"
```

Hunyuan refine 还需要 `TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY` 和可用的 Blender。后续 USD/物理处理需要 Isaac Sim Python。

## 输入图片

`--sam3d-input` 可以是一个图片文件，也可以是图片文件夹。支持扩展名：

```text
.png  .jpg  .jpeg  .webp  .bmp
```

单图可以直接传文件：

```text
data/shelf.jpg
```

多图可以直接放在一个目录中：

```text
data/sam3d_images/
├── front.jpg
├── left.jpg
├── right.jpg
└── back.jpg
```

也可以使用 `images/` 子目录：

```text
data/sam3d_images/
└── images/
    ├── 00000.jpg
    ├── 00001.jpg
    └── 00002.jpg
```

读取规则：

- 如果输入目录中存在 `images/`，只读取该子目录中的图片。
- 只读取当前目录，不递归扫描更深的子目录。
- 文件按名称排序后处理；multi 模式会重命名为 `00000.png`、`00001.png` 等。
- 名称以 `mask_` 开头或以 `_mask` 结尾的图片会被忽略。
- 所有输入都会复制成 RGB PNG；原图片的 alpha 通道不会直接作为 mask。
- 不需要提前提供 mask。wrapper 会使用同一个 `--sam3d-prompt` 为每个视角自动生成 mask。

多视角图片应展示同一个物体。建议主体完整、遮挡较少、各视角尺度接近，并覆盖正面、侧面和背面等互补角度。图片不需要预先提供相机标定，但清晰背景和稳定构图通常更容易得到一致的 mask。

## 完整命令

```bash
python ./run_asset_pipeline.py \
  --sam3d-input ./data/sam3d_images \
  --sam3d-mode auto \
  --sam3d-prompt "metal shelves" \
  --sam3d-seed 42 \
  --sam3d-steps 50 \
  --output-dir ./sam3d_downloads \
  --intermediate-output-dir ./sam3d_output_intermediate \
  --final-output-dir ./sam3d_output_final \
  --result-json ./sam3d_pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## SAM3D 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--sam3d-input` | 无 | 原始图片文件或图片目录。使用 SAM3D 图片路径时必填，并且不能同时传 `--sam3d-glb`、`--existing-glb` 或 `--manual-stp`。 |
| `--sam3d-mode` | `auto` | 输入模式，可选 `auto`、`single`、`multi`。它决定准备输入的目录结构以及调用单视角还是多视角重建。 |
| `--sam3d-prompt` | 无 | SAM3 文本分割提示词，使用 `--sam3d-input` 时必填。它用于在图片中找到目标物体，不是 text-to-3D 描述。 |
| `--sam3d-seed` | `42` | 重建阶段随机种子。相同软件、模型、输入和参数下有助于复现结果；它不负责选择文本分割目标。 |
| `--sam3d-steps` | `50` | 第一阶段几何采样步数，即传给 SAM3D 的 `stage1_inference_steps`/`stage1_steps`。值越大通常耗时越长，不等同于面数。第二阶段当前固定为 25 步。 |
| `--output-dir` | `./downloads` | SAM3D 工作目录根路径。实际运行目录是 `<output-dir>/sam3d/<输入名>`；已存在时自动追加 `_2`、`_3` 等，避免覆盖旧结果。 |
| `--sam3d-glb` | 无 | 已经生成好的 SAM3D GLB。跳过图片分割和 SAM3D 重建，但默认仍执行 Hunyuan refine 和后续 Blender/Isaac 处理。 |

### `--sam3d-mode`

| 模式 | 实际行为 | 建议用途 |
| --- | --- | --- |
| `auto` | 发现 1 张有效图片时解析为 `single`；超过 1 张时解析为 `multi`。 | 推荐默认值。 |
| `single` | 只使用排序后的第一张图片，其余图片会被忽略。 | 单张产品图，或明确只想使用一个视角。 |
| `multi` | 把全部有效图片送入 multi-view 项目；应至少准备两个互补视角。 | 同一物体的多角度照片。 |

最终采用的模式和有效图片数量会打印在日志中：

```text
SAM3D source images: ... | image_count=4 | mode=multi
```

### `--sam3d-prompt`

prompt 应描述要从图片中分割出来的物体，例如：

```text
metal shelves
industrial control cabinet
red plastic crate
```

尽量使用清晰的英文物体名和必要的外观限定词。不要写动作、物理属性、目标尺寸或大段生成描述，因为 prompt 只用于二维分割。同一 prompt 会应用到全部视角；如果某一视角找不到目标，该视角会被排除，所有视角都没有有效 mask 时流程失败。

### `--sam3d-seed` 和 `--sam3d-steps`

- 先固定 `--sam3d-seed 42` 比较输入图片和 prompt，便于判断变化来自哪里。
- 几何不稳定时可以换 seed 重新采样；不同 seed 可能产生不同细节。
- `--sam3d-steps 50` 是当前推荐基线。增加步数会增加第一阶段推理时间，但不保证一定提高质量。
- 该参数不控制 Hunyuan ReduceFace 面数或 Isaac Sim collision 精度。

## 与后续处理有关的参数

| 参数 | 说明 |
| --- | --- |
| `--refine-config-path` | Hunyuan ReduceFace 和本地 Blender 后处理配置。生产路径使用 `configs/hunyuan_reduce_local_postprocess.yaml`。 |
| `--refine-temp-upload` | 本地 GLB 的临时上传 provider。`uguu` 用于让 Hunyuan API 能下载本地文件。 |
| `--skip-refine` | 跳过 Hunyuan refine。只建议诊断时使用；正常 SAM3D 资产应保留 refine。 |
| `--len-x`, `--len-y`, `--len-z` | refine 后资产在 X/Y/Z 方向的目标尺寸，单位 m。 |
| `--orientation` | 按包围盒长、中、短边做坐标轴映射，例如 `X=L,Y=M,Z=S`。 |
| `--approx` | Isaac Sim collision approximation，例如 `sdf`。 |
| `--set-mass` | 最终资产总质量，单位 kg。 |

## 输出文件

单视角工作目录通常为：

```text
sam3d_downloads/sam3d/<输入名>/
├── image.png
├── 0.png
├── result_obj0.glb
├── result_obj0.obj
├── result_obj0.ply
└── result_obj0_refined_mesh/ 或 result_refined_mesh/
```

多视角工作目录通常为：

```text
sam3d_downloads/sam3d/<输入名>/
├── images/
│   ├── 00000.png
│   └── 00001.png
├── masks/
│   ├── 00000.png
│   └── 00001.png
├── result.glb
├── result.obj
├── result.ply
└── result_refined_mesh/
```

pipeline 选择 GLB 的优先顺序是 `scene_combined.glb`、`result.glb`、`result_obj0.glb`，最后才选择目录中的其它 GLB。完整路径会记录在 `sam3d_pipeline_result.json` 的 `generation.selected_glb` 中。

## 从已有 SAM3D GLB 继续

如果 SAM3D 已成功生成 GLB，但后续 Hunyuan refine 因网络或服务超时失败，不必再次运行图片重建。改用 `--sam3d-glb` 继续：

```bash
python ./run_asset_pipeline.py \
  --sam3d-glb ./sam3d_downloads/sam3d/sam3d_images/result.glb \
  --intermediate-output-dir ./sam3d_output_intermediate \
  --final-output-dir ./sam3d_output_final \
  --result-json ./sam3d_pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## 常见问题

| 日志或现象 | 原因与处理 |
| --- | --- |
| `No module named 'sam3.model'` | 当前环境不完整，或 SAM3 子模块缺失。激活 `hunyuan_sam3d`，并检查 `tools/sam3d/third_party/sam-3d-objects/submodules/sam3`；自定义目录时再检查 `SAM3D_SINGLE_VIEW_ROOT`。 |
| `SAM3 segmentation produced no valid masks` | prompt 没有稳定命中目标。检查拼写，改成更直接的物体名，并确认目标在所有图片中清晰可见。 |
| `Multi-view SAM3D script not found` | `sam-3d-objects-multiview` 不在默认位置。设置 `SAM3D_MULTI_VIEW_ROOT`。 |
| `FailedOperation.RequestTimeout` | Hunyuan ReduceFace 后端临时超时。当前配置会自动进入下一次提交重试；也可以从已有 `result.glb` 使用 `--sam3d-glb` 续跑。 |
| Pillow、spconv 或 AMP 的 `Warning` | 这些通常是兼容性/弃用警告；只要最后出现 `SAM3D reconstruction finished` 就不是本次失败原因。 |
