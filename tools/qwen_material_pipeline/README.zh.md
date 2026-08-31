# 自动材质工具包

[English](./README.md) | [中文](./README.zh.md)

本目录包含 STEP/STP 自动赋材质的内部实现：分割、检索、材质选择、USD 绑定和验证。
普通用户只需按[自动赋材质](../../docs/guides/manual-part-id-materials.zh.md)运行
`manual-material-pipeline`，不需要单独调用这里的子命令。

## 与主流程的关系

```text
manual-material-pipeline
-> asset_pipeline.manual_material_cli
-> asset_pipeline.manual_cad
-> asset_pipeline.visual_materials
-> qwen_material_pipeline 子进程工具
```

`asset_pipeline` 负责阶段顺序、跨运行时调用和最终交付；本包负责模型推理、分割、材质候选、
图片比较和 USD 材质操作。

| 目录 | 内容 |
| --- | --- |
| `segmentation/` | SAM3、EntitySeg 和融合分割。 |
| `evidence/` | Part-ID、相机和照片判断信息。 |
| `mvinverse/`、`retrieval/`、`qwen/` | 外观估计、候选检索和排序。 |
| `materials/` | 材质计划、候选比较和选择锁。 |
| `usd/` | Part-ID 渲染、材质绑定和 USD 检查。 |
| `workflows/` | 可由主流程启动的内部阶段。 |

内部命令以 `qwen-material --help` 为准。产物写入任务的 `visual_material/`，不写回源码目录。

## 开发检查

```bash
qwen-material --help
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

代码结构见[架构说明](../../docs/development/architecture.zh.md)。MVInverse 仅限非商业用途；其他许可见
[第三方声明](../../legal/THIRD_PARTY_NOTICES.md)。
