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
目录；只有继续同一份输入时才加 `--resume`。视觉材质阶段只会复用输入、输出哈希均通过
校验的检查点。

可选物理参数包括 `--material`、`--approx`、`--sdf-resolution`、`--set-mass` 和
`--cad-option KEY=VALUE`。STEP/STP 不使用目标长宽高参数。

## 命令执行内容

```text
STEP/STP -> USD 和物理准备 -> 相机对齐
         -> 为每个可见 Part-ID 生成模型图 isolated 模板
         -> SAM3 + EntitySeg 第一遍分割
         -> 邻件关系引导的第二遍分割与迭代融合
         -> 不看颜色，先确定材质身份
         -> 只对合格的对应材质校准审核过的颜色参数
         -> 为全部 Mesh 应用材质 -> 最终验证
```

SAM3 前景确认是唯一必需的人工步骤。当前主线是通用方法，不包含工件名称、Part-ID 名单、
按视角定制的提示词或人工材质映射；所有已注册视角都必须贡献真实证据。局部图片贴合只调整
二维候选，不修改 CAD、单个 Mesh 变换或最终交付相机。

隐藏或无法判断的零件使用预设默认材质。精确库预设保持不变；“对应材质”只有在材质身份
固定之后，才能调整经过审核的颜色接口。局部颜色质量未达标时保留实测最优结果并记录
`REVIEW`，不会让整条流程中断；哈希损坏、Part-ID 覆盖不完整、材质身份变化或交付数据无效
仍会严格失败。

## 输出

```text
RUN_ROOT/{cad_usd,intermediate,visual_material,final}/
RUN_ROOT/pipeline_result.json
```

证据和选择审计位于 `RUN_ROOT/visual_material/analysis/`：

```text
part_id_cad_amodal_templates/manifest.json
part_id_relation_guidance/request.json
part_id_hybrid_masks/manifest.json
part_id_reference_evidence.json
part_id_material_plan.json
```

详细说明见[参考图赋材质流程](./modules/visual-materials.zh.md)、
[环境变量模板](../.env.example)和[第三方声明](../THIRD_PARTY_NOTICES.md)。MVInverse 仅限
非商业用途。
