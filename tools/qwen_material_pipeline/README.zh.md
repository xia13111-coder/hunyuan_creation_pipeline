# 自动材质工具包

[English](./README.md) | [中文](./README.zh.md) | [项目 README](../../README.zh.md)

为手工建模 STEP/STP 资产提供参考图驱动的 NVIDIA MDL 检索和 USD 材质工具。普通用户
运行根命令 `manual-material-pipeline`；本子包提供其中的材质阶段。

## 适用范围

默认配置为 `configs/pipeline/manual_part_id_materials.json`。流程按 CAD Part-ID 从
NVIDIA `Materials/Base` 中选择视觉效果最接近的材质，选定后保持原始参数。

本包负责参考图分析、相机配准、Part-ID 映射、材质检索、USD 绑定和验证。CAD 转换及
总流程由 `asset_pipeline` 负责。

## 安装

```bash
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
qwen-material --help
```

本机模型和应用路径写在 `.env` 中，可从 `.env.example` 复制。不要提交填写后的 `.env`。

可选的 Qwen3.5/SigLIP2 校验安装：

```bash
bash tools/qwen_material_pipeline/scripts/setup_qwen35_runtime.sh
```

SAM3、MVInverse、DINOv2、NVIDIA 材质和 Base 材质观察库是独立本机依赖，必须通过
预检。

## 流程

```text
归一 CAD + 已确认照片
  -> 对齐 RGB 图并提取 Part-ID 外观信息
  -> SAM3/MVInverse/SigLIP2/DINOv2/Qwen3.5
  -> Base MDL 候选渲染
  -> 为每个 Part-ID 生成一条材质分配
  -> 记录最终 MDL、写入 USD 并做视觉验证
```

每个可见 Part-ID 独立判断；不可见零件使用预设默认材质。模型只负责候选排序，USD 绑定由
经过校验的程序完成。选中的 MDL 保持材质库默认参数。

## 命令

| 命令 | 作用 |
| --- | --- |
| `sam3-foreground-ui` | 确认整机前景 |
| `staged` | 运行材质推理阶段 |
| `catalog` | 构建或检查 NVIDIA MDL 目录 |
| `base-bank` | 构建或验证 Base 材质观察库 |
| `part-id-qwen` | 按 Part-ID 排序候选 |
| `exact-mdl-tournament` | 渲染比较 MDL 候选 |
| `compare` | 比较参考图和渲染图 |
| `final-visual-gate` | 验证收集后的 USD |
| `usd` | USD 零件索引、渲染、绑定与验证 |

完整参数以 `qwen-material --help` 为准。普通用户通常只需
`manual-material-pipeline` 和 `sam3-foreground-ui`。

## 恢复与输出

`--resume` 只复用输入、配置、数据格式、模型和哈希均一致的产物。

主要结果：

```text
visual_material/renders/
visual_material/analysis/{reference_manifest,qwen_inference_ledger,
  part_id_reference_evidence,part_id_qwen_choices,material_selection_lock}.json
visual_material/analysis/mvinverse/
visual_material/visual_quality/
visual_material/final_visual_acceptance/
```

本机结果、缓存、模型和工作目录不进入源码发布包。

## 文档与测试

- [用户命令](../../docs/manual-part-id-materials.zh.md)
- [行为与排错](../../docs/modules/visual-materials.zh.md)
- [架构](./docs/architecture.zh.md)
- [MVInverse](./docs/mvinverse.zh.md)

```bash
PYTHONPATH=./tools python -m pytest -q -p no:cacheprovider \
  tools/qwen_material_pipeline/tests
```

## 许可证

自研代码采用 [Apache License 2.0](./LICENSE)。MVInverse 仍仅限非商业用途；模型和 NVIDIA
资产使用独立条款，见[第三方声明](../../THIRD_PARTY_NOTICES.md)。
