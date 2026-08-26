# 自动材质包架构

`qwen_material_pipeline` 是 STEP/STP 主流程的视觉材质子包。生产入口是仓库根目录安装出的
`manual-material-pipeline`；CAD 转换、物理处理、结果目录和最终交付由 `asset_pipeline`
负责。

## 目录边界

```text
tools/qwen_material_pipeline/
├── src/qwen_material_pipeline/
│   ├── core/          # 公共合同与阶段状态
│   ├── evidence/      # 相机、Part-ID、颜色与质量证据
│   ├── segmentation/  # SAM3、EntitySeg 与关系引导融合
│   ├── retrieval/     # 视觉检索
│   ├── materials/     # MDL 目录、候选、选择与颜色参数
│   ├── mvinverse/     # MVInverse 适配
│   ├── qwen/          # VLM 适配
│   ├── usd/           # USD 渲染、绑定与验证
│   ├── workflows/     # 子包工作流
│   ├── configs/       # 版本化生产配置
│   ├── schemas/       # 跨阶段数据格式
│   └── web/           # 标注和结果查看器
├── tests/             # 子包测试
├── docs/              # 子包文档
├── scripts/           # 安装和维护脚本
├── requirements/      # runtime/dev/entityseg/qwen35 依赖
├── third_party/       # 固定版本源码及上游许可证
└── runtime/           # 本机模型、缓存和私有封存项目；Git 忽略
```

每次运行的图片、日志、分析文件和 USD 统一写入仓库 `outputs/<run-id>/`，不放入包源码或
`runtime/`。路径解析由 `asset_pipeline/project_layout.py` 和包内路径模块统一负责，业务
模块不得自行推算仓库层级。

## 模块规则

- `evidence/` 和 `segmentation/` 只生成、校验证据，不写 USD。
- `qwen/`、`mvinverse/` 和 `retrieval/` 只生成观测或候选。
- `materials/` 生成完整、可追溯的 Part-ID 材质计划。
- `usd/` 只执行已确定的计划并验证结果。
- `web/` 只标注或展示数据，不成为生产决策 owner。
- 第三方代码只放 `third_party/`，不得混入 `src/`；本机权重和私有数据只放 `runtime/`。

主仓库的 `asset_pipeline/visual_materials/orchestrator.py` 负责编排；
`asset_pipeline/visual_materials/stages/part_id_evidence.py` 负责逐 Part-ID 分割和融合顺序。

## 生产数据流

```text
CAD USD + 参考图
  -> 相机配准与 RGB/Part-ID 渲染
  -> isolated 模型图模板
  -> SAM3/EntitySeg + 邻件关系引导 + 迭代融合
  -> MVInverse/SigLIP2/DINOv2/Qwen 候选
  -> 真实 CAD 候选渲染比较
  -> 材质身份锁定与对应材质自动校色
  -> USD 绑定、重渲染和最终验证
```

几何和材质严格分离：分析相机、照片 mask 和颜色参数不得改变 CAD 尺寸、姿态、拓扑、
碰撞、质量或关节。`--resume` 仅在输入、配置、模型、数据格式和哈希一致时复用产物。

## 开发与查看

从仓库根目录安装并测试：

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

结果查看器源码位于
`tools/qwen_material_pipeline/src/qwen_material_pipeline/web/result_viewer/`；运行结果仍从
`outputs/<run-id>/` 读取。清理与发布规则见[维护说明](./maintenance/cleanup.zh.md)。
