# Hunyuan 资产创建流水线

[English](./README.md) | [中文](./README.zh.md) | [文档索引](./docs/README.zh.md)

把图片、已有 GLB 或手工建模 STEP/STP 转换为 USD，并可选执行网格精修、Isaac Sim 物理
处理和参考图驱动的 NVIDIA MDL 材质赋值。

## 支持流程

| 输入 | 输出 |
| --- | --- |
| Hunyuan 图片 | 生成/精修 GLB 和 USD |
| SAM3D 图片 | 重建 GLB 和 USD |
| 已有 GLB | 精修并调整朝向的 USD |
| STEP/STP | 保持原始尺寸的 USD，可选物理和材质 |

STEP/STP 材质流程会把 CAD 渲染图与 2–4 张实物照片对齐，按 CAD Part-ID 比较
NVIDIA `Materials/Base` 候选并验证最终 USD。对齐或图像信息不足时，流程会停止并
报告未通过的检查。

## 环境要求

- Linux x86-64 和 NVIDIA CUDA GPU；
- Conda 与 Python 3.10；
- 单独安装 Blender 和 Isaac Sim；
- 本地 Qwen3.5、SAM3、MVInverse、SigLIP2、DINOv2 权重；
- 本机 NVIDIA MDL 材质库；
- 只有 Hunyuan 生成或精修阶段需要腾讯云凭据。

完整本地材质流程当前需要约 24 GB 显存。模型、NVIDIA MDL、外部运行时和密钥不随
源码发布；Base 材质观察库已包含在仓库中。不同流程按需加载依赖：不加视觉材质的
STEP/STP 流程不需要 Qwen、SAM3、
MVInverse、SigLIP2、DINOv2 或 NVIDIA MDL 材质库。

## 安装

```bash
git clone https://github.com/xia13111-coder/hunyuan_creation_pipeline.git
cd hunyuan_creation_pipeline

conda env create -f environment.yml
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
cp .env.example .env
```

## `.env` 配置

`.env` 只保存本机路径。模型路径由下面的命令自动填写，其他空项由程序自动发现。

先在 Hugging Face 接受 [SAM3](https://huggingface.co/facebook/sam3) 和
[EntitySeg](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg) 的访问条件，然后运行：

```bash
hf auth login
qwen-material setup-models --model-root /data/hunyuan-models
```

该命令下载 Qwen3.5、MVInverse、SAM3、EntitySeg、SigLIP2 和 DINOv2，并自动更新
`.env`。重复运行会继续未完成的下载；正常推理仍然只读取本地权重。

仍需手动填写：

| 变量 | 内容 |
| --- | --- |
| `ISAAC_PYTHON` | Isaac Sim 的 `python.sh` |
| `VISUAL_MATERIAL_ROOT` | 包含 `Base/` 的 NVIDIA Materials 目录 |

SAM3、CropFormer 源码、EntitySeg 环境和材质观察库也由该命令自动准备；实际需要手填的
只剩 Isaac Sim 和 NVIDIA Materials 路径。

完整变量见 [.env.example](./.env.example)，Docker 配置见
[docker/README.zh.md](./docker/README.zh.md)。

验证入口：

```bash
hunyuan-asset-pipeline --help
manual-material-pipeline --help
qwen-material --help
```

容器部署见 [docker/README.zh.md](./docker/README.zh.md)。

## 快速开始

以下命令都在项目根目录运行，并为每次任务使用独立输出目录。

### Hunyuan 图片生成

```bash
RUN=./outputs/hunyuan/basket
hunyuan-asset-pipeline \
  --input-dir ./input/basket_images \
  --output-dir "$RUN/generation" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.3 \
  --material plastic --approx sdf
```

Hunyuan 只接受本地图片目录或单张公网图片 URL，不支持纯文本生成。使用公网图片时，
把 `--input-dir` 换成 `--image-url URL`。

### SAM3D 图片重建

输入目录中的图片应展示同一个物体：

```bash
RUN=./outputs/sam3d/cabinet
hunyuan-asset-pipeline \
  --sam3d-input ./input/cabinet_views \
  --sam3d-mode auto \
  --sam3d-prompt "industrial cabinet" \
  --output-dir "$RUN/generation" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --material steel --approx sdf
```

### 已有 GLB

```bash
RUN=./outputs/glb/model
hunyuan-asset-pipeline \
  --existing-glb ./input/model.glb \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --material plastic --approx sdf
```

已经精修完成的 GLB 可以加 `--skip-refine`。

### STEP/STP：不加视觉材质

这条路径只做 CAD 转 USD、物理属性和依赖收集，不调用 Qwen、MVInverse 或参考图流程：

```bash
RUN=./outputs/manual/asset_physics
hunyuan-asset-pipeline \
  --manual-stp ./input/asset.stp \
  --cad-usd-output-dir "$RUN/cad_usd" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32
```

### STEP/STP：根据参考图自动赋材质

先用 `qwen-material sam3-foreground-ui` 对 2–4 张照片进行 SAM3 人工点选分割并确认
整机前景：

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

所有视角确认并保存后，运行自动赋材质流程：

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

STEP/STP 保留原始尺寸。详细标注、恢复和结果说明见
[自动赋材质快速开始](./docs/guides/manual-part-id-materials.zh.md)。

详细说明见 [Hunyuan](./docs/modules/hunyuan.zh.md)、[SAM3D](./docs/modules/sam3d.zh.md)、
[CAD](./docs/modules/cad.zh.md)、
[生成方式选择](./docs/guides/generation-guide.zh.md)。

## 输出

```text
RUN_ROOT/
├── generation/       # 仅 Hunyuan/SAM3D
├── cad_usd/           # 仅 STEP/STP
├── intermediate/
├── visual_material/   # 仅自动视觉材质
├── final/
└── pipeline_result.json
```

具体任务只创建自己需要的目录。

生成结果可能包含本机路径、照片或私有输入。Git 默认忽略这些文件；分享前请先检查。

## 仓库结构

```text
asset_pipeline/                 编排与工作流
asset_refiner/                  网格精修
tools/{blender,isaac,sam3d}/    Blender、Isaac Sim 和 SAM3D 执行脚本
tools/qwen_material_pipeline/   材质推理和 USD 工具
configs/                        版本化配置
requirements/                   按用途拆分的依赖增量
docs/{guides,development,modules}/
                                详细文档
legal/                          第三方版权与许可证清单
tests/                          测试
outputs/                        生成结果
```

主要文档：

- [文档索引](./docs/README.zh.md)
- [架构](./docs/development/architecture.zh.md)
- [视觉材质](./docs/modules/visual-materials.zh.md)

## 测试

```bash
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

## 许可证

自研代码和文档采用 [Apache License 2.0](./LICENSE)。MVInverse 仅限非商业用途；模型、
NVIDIA 软件、MDL 材质和生成资产使用各自条款。见
[法律文件索引](./legal/README.zh.md)、
[第三方声明](./legal/THIRD_PARTY_NOTICES.md)。
