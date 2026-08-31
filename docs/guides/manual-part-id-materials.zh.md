# STEP/STP 自动赋材质

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

流程使用同一工件的 2–4 张照片，为 STEP/STP 中的每个 CAD Part-ID 选择并绑定 NVIDIA
Base MDL。照片只提供外观信息；CAD 始终决定几何、尺寸和装配关系。

## 使用前准备

- 一个 `.stp` 或 `.step` 文件；
- 同一工件的 2–4 张不同视角照片；
- Isaac Sim、NVIDIA Base Materials 和约 24 GB 显存；
- 已通过 `qwen-material setup-models` 准备本地模型。

安装、自动下载模型和 `.env` 配置见项目 [README](../../README.zh.md)。安装完成后确认以下
三个命令可用：

```bash
hunyuan-asset-pipeline --help
manual-material-pipeline --help
qwen-material --help
```

## 运行

### 1. 确认整机前景

在页面中点选并确认每张照片里的整个工件。视角名称可以自定义，但不能重复。

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

这里只标注整机，不需要逐个零件标注。标注文件会保存照片路径、视角名称和内容哈希；照片
发生变化后必须重新标注。

### 2. 运行自动赋材质

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

常用可选参数：

| 参数 | 说明 |
| --- | --- |
| `--resume` | 继续同一输入和输出目录，只复用通过哈希检查的阶段结果。 |
| `--material NAME` | 物理材质预设，默认 `plastic`。 |
| `--approx NAME` | 碰撞近似方式，默认 `sdf`。 |
| `--sdf-resolution N` | SDF 分辨率，默认 `32`。 |
| `--set-mass KG` | 设置工件总质量。 |
| `--cad-option KEY=VALUE` | 传递 CAD Converter 参数，可以重复使用。 |
| `--config FILE` | 使用另一份自动材质配置。 |

新任务应使用新的输出目录。只有输入、配置和模型均未改变时，才对原目录添加 `--resume`。
STEP/STP 保留工程尺寸，因此不使用目标长宽高或朝向参数。

## 核心流程

```text
STEP/STP
  -> CAD 转 USD，统一单位、原点和物理几何
  -> 渲染 CAD 整机图和每个 Part-ID 的模型图
  -> 将 CAD 整机相机与照片相机对齐
  -> 用模型图中的零件形状和装配位置定位照片中的对应零件
  -> SAM3 + EntitySeg 两遍分割并迭代融合
  -> MVInverse、SigLIP2、DINOv2 和 Qwen 生成并排序材质候选
  -> 先固定材质身份，再对允许调色的对应材质校准颜色
  -> 为所有 Mesh 绑定材质并渲染检查
  -> 收集依赖，验证最终 USD
```

几个关键规则：

- Part-ID 形状来自 CAD 模型图，不是在照片上直接平移 CAD 掩码；照片中的位置还会结合相机
  对齐、邻近零件关系和分割结果重新推断。
- 局部优化只调整照片中的二维候选区域，不会移动单个 Mesh。工件姿态变化只能作为整件刚体
  和相机参数处理。
- SAM3 提供候选区域，EntitySeg 补充物体边界，CAD 模型图约束形状；最终结果由三类信息
  共同判断，不是简单二选一。
- 先判断材质种类，再判断颜色。材质库中存在可信的精确预设时直接保留；只能找到对应材质
  时，才修改程序明确支持的颜色参数。
- 颜色候选必须用真实 CAD 重渲染比较。局部颜色未达到目标时保留实测最优候选并记录
  `REVIEW`；最终 USD、材质身份或数据完整性检查失败时仍会停止。
- 所有 Mesh 都必须有结果。照片中看不到或无法可靠判断的零件使用安全默认材质，不伪造
  照片判断信息。
- 主流程不包含工件名称、固定 Part-ID 列表、按视角手写提示词或人工材质映射。

## 输出

| 路径 | 内容 |
| --- | --- |
| `RUN_ROOT/cad_usd/` | CAD Converter 的原始 USD。 |
| `RUN_ROOT/intermediate/` | 完成单位、几何和物理准备的 USD。 |
| `RUN_ROOT/visual_material/renders/` | CAD 整机图和 Part-ID 渲染图。 |
| `RUN_ROOT/visual_material/camera_calibration/` | 相机搜索、对齐参数和检查结果。 |
| `RUN_ROOT/visual_material/analysis/` | 分割、检索、材质选择和校色依据。 |
| `RUN_ROOT/visual_material/visual_quality/` | 参考图与赋材质渲染的比较结果。 |
| `RUN_ROOT/final/` | 收集依赖并通过最终检查的 USD。 |
| `RUN_ROOT/pipeline_result.json` | 整次任务的输入、状态和主要结果路径。 |

排查单个零件时，优先查看以下文件：

```text
visual_material/analysis/part_id_cad_amodal_templates/manifest.json
visual_material/analysis/part_id_relation_guidance/request.json
visual_material/analysis/part_id_hybrid_masks/manifest.json
visual_material/analysis/part_id_reference_evidence.json
visual_material/analysis/part_id_material_plan.json
visual_material/analysis/material_selection_lock.json
```

## 中断与错误

先查看日志中第一条 `FAILED`，后面的异常通常只是上游失败的结果。

| 情况 | 处理 |
| --- | --- |
| `CUDA out of memory` | 停止其他 GPU 进程，再对同一目录使用 `--resume`。 |
| `No space left on device` | 清理磁盘或换到更大的输出盘，再继续任务。 |
| SAM3 标注或策略不匹配 | 照片变化后重新生成标注；旧任务不要与新标注混用。 |
| 相机或 Part-ID 信息不完整 | 查看 `camera_calibration/` 和 Part-ID 对比图，不要跳过失败视角。 |
| EntitySeg 的 deterministic 警告 | 这是可复现性警告；是否失败以之后的 traceback 和 `FAILED` 为准。 |

代码调用关系见[架构说明](../development/architecture.zh.md)。MVInverse 仅限非商业用途；
其他许可见[第三方声明](../../legal/THIRD_PARTY_NOTICES.md)。
