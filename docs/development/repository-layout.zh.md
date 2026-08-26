# 仓库目录

[English](./repository-layout.md) | [中文](./repository-layout.zh.md) | [文档索引](../README.zh.md)

## 顶层目录

```text
asset_pipeline/                 工作流与编排
asset_pipeline/visual_materials/stages/
                                具有单一 owner 的材质流水线阶段
asset_refiner/                  网格精修包
tools/{blender,isaac,sam3d}/    外部运行时使用的执行脚本
tools/qwen_material_pipeline/   材质推理与 USD 工具
configs/                        版本化配置
requirements/                   按用途拆分的依赖增量
docs/{guides,development,release,modules}/
                                用户、开发和发布文档
legal/                          第三方版权与许可证清单
.github/                        贡献、行为规范与安全策略
tests/                          主流水线测试
apps/                           独立网页应用
examples/                       示例说明和可公开元数据
outputs/                        运行结果；Git 默认忽略
docker/                         容器文件与说明
```

仓库路径统一由 `asset_pipeline/project_layout.py` 解析。其他模块不应自行拼接
`tools/`、`configs/` 或 `outputs/` 的位置。

根目录的 `LICENSE`、`NOTICE` 和 `CITATION.cff` 特意保留在打包与托管工具能够识别的
位置。仓库级第三方声明放入 `legal/`；独立构建包或第三方源码自己的许可证继续与代码
放在一起。

## 材质工具包主要目录

下表按功能列出主要目录，不是完整文件树。

```text
tools/qwen_material_pipeline/
├── workflows/       命令工作流
├── evidence/        相机、Part-ID、颜色、PBR 与质量测量
├── retrieval/       SigLIP2 和 DINOv2 检索
├── materials/       MDL 目录、选择与应用规则
├── segmentation/    SAM3、EntitySeg、邻件关系引导与 hybrid mask
├── mvinverse/       MVInverse 适配器和运行记录
├── qwen/            本地与远程 VLM 适配器
├── usd/             零件索引、渲染、应用与验证
├── core/            共用数据结构
├── configs/         工具包配置
├── schemas/         JSON Schema
├── scripts/qwen35/  隔离的 Qwen3.5 运行环境安装脚本
├── web/             标注、结果查看器及其服务脚本
├── third_party/     保留上游许可证的第三方源码
├── models/          本机权重；Git 默认忽略
├── var/             可重建的索引和缓存；Git 默认忽略
└── results/         本机结果；Git 默认忽略
```

材质工具包有独立的 `pyproject.toml`，但实体目录仍位于 `tools/`。不要在仓库根目录
复制一份，也不要创建兼容软链接。

## 文件放置规则

- 每次运行的文件统一放在 `outputs/<run-id>/`。
- 照片、私有 CAD、密钥、模型权重和运行结果不得进入源码发布包。
- Blender、Isaac Sim 和 SAM3D 执行脚本放在各自的 `tools/` 目录；完整工作流放在
  `asset_pipeline/`。
- 各外部运行时的生产 worker 保持在对应目录顶层；可选操作统一放入 `utilities/` 或
  `diagnostics/`。
- `apps/` 可以读取已发布结果，但不能被 `asset_pipeline` 导入。
- 只有可重建缓存可以直接删除；删除输入、分析记录、模型或 USD 前应先确认是否需要保留。

在当前环境中安装两个 Python 包：

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

安装后可使用 `hunyuan-asset-pipeline`、`manual-material-pipeline` 和
`qwen-material`。根目录的 `run_*.py` 脚本仅用于兼容旧命令；新的材质自动化应使用
`manual-material-pipeline` 或 `python -m asset_pipeline.manual_material_cli`。
