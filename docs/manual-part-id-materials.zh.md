# 手工建模 STEP/STP（CAD）逐 Part-ID 自动赋材质

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

使用同一工件的 2–4 张照片，为一个 STEP/STP 装配体选择 NVIDIA Base MDL。CAD 始终
决定几何和真实尺寸。

## 运行

首次安装命令：

```bash
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

命令会自动读取项目根目录的 `.env`；非空的 shell 环境变量优先。

1. 在每个视角中确认整机前景：

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

2. 启动全新的自动任务：

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

标注文件保存图片哈希和视角 ID。修改图片后需要重新标注。要从零运行，请使用新的输出
目录；只有继续同一个已校验任务时才加 `--resume`。

可选物理参数包括 `--material`、`--approx`、`--sdf-resolution`、`--set-mass` 和
`--cad-option KEY=VALUE`。STEP/STP 不使用目标长宽高参数。

## 命令执行内容

```text
STEP/STP -> USD 和物理准备 -> 相机对齐
         -> 提取每个可见 Part-ID 的外观信息
         -> NVIDIA Base 检索和 MDL 渲染比较
         -> 为全部 Mesh 应用材质 -> 最终验证
```

SAM3 前景确认是唯一必需的人工步骤。局部图片贴合不会修改 CAD；隐藏或无法判断的零件
使用预设默认材质，选中的 MDL 参数不会再修改。对齐、赋值覆盖、视觉对比或最终交付检查
失败时，流程会停止并报告对应阶段。

## 输出

```text
RUN_ROOT/{cad_usd,intermediate,visual_material,final}/
RUN_ROOT/pipeline_result.json
```

详细说明见[参考图赋材质流程](./modules/visual-materials.zh.md)、
[环境变量模板](../.env.example)和[第三方声明](../THIRD_PARTY_NOTICES.md)。MVInverse 仅限
非商业用途。
