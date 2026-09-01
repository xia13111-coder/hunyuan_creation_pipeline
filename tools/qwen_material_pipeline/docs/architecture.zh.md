# 自动材质包架构

`qwen_material_pipeline` 提供 STEP/STP 自动赋材质的算法和 USD 工具。顶层流程由
`asset_pipeline` 编排，生产入口是 `manual-material-pipeline`。

## 目录

```text
src/qwen_material_pipeline/
├── core/          公共数据和阶段版本
├── evidence/      相机、Part-ID 和外观信息
├── segmentation/  SAM3、EntitySeg 与关系引导融合
├── retrieval/     SigLIP2、DINOv2 检索
├── materials/     MDL 候选、选择和颜色参数
├── mvinverse/     MVInverse 适配
├── qwen/          视觉语言模型适配
├── usd/           USD 渲染、绑定和验证
├── workflows/     子包工作流
├── configs/       生产配置
├── schemas/       JSON 数据格式
└── web/           标注和结果查看器
```

`tests/`、`docs/`、`scripts/` 和 `requirements/` 分别保存测试、文档、维护脚本和依赖。
第三方源码放在 `third_party/`；本机模型和缓存放在被 Git 忽略的 `runtime/`。

## 数据流

```text
CAD USD + 参考图
-> 相机和 Part-ID 信息
-> CAD 引导的 SAM3/EntitySeg 分割
-> MVInverse/SigLIP2/DINOv2/Qwen 候选
-> MDL 真实渲染比较
-> 材质身份与必要的颜色校准
-> USD 绑定和最终验证
```

分析相机和照片掩码不能修改 CAD 几何或物理属性。`--resume` 只复用输入、配置、模型、
数据格式和哈希完全匹配的结果。

## 职责边界

- `asset_pipeline/visual_materials/`：阶段顺序、运行时和交付检查；
- `evidence/`、`segmentation/`：生成判断信息；
- `materials/`：生成完整材质计划；
- `usd/`：执行计划并验证 USD；
- `web/`：标注和展示，不参与生产决策。

运行产物统一写入仓库 `outputs/<run-id>/`，不放入源码目录。

## 开发检查

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

用户入口见[快速开始](../../../docs/guides/manual-part-id-materials.zh.md)。
