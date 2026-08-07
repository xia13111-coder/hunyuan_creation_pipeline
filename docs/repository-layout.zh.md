# 仓库目录

[English](./repository-layout.md) | [中文](./repository-layout.zh.md) | [文档索引](./README.zh.md)

## 顶层目录

```text
asset_pipeline/                 工作流与编排
asset_refiner/                  网格精修包
tools/{blender,isaac,sam3d}/    外部运行时使用的执行脚本
tools/qwen_material_pipeline/   材质推理与 USD 工具
configs/                        版本化配置
docs/                           用户与开发文档
tests/                          主流水线测试
apps/                           独立网页应用
examples/                       示例说明和可公开元数据
outputs/                        运行结果；Git 默认忽略
docker/                         容器文件与说明
```

仓库路径统一由 `asset_pipeline/project_layout.py` 解析。其他模块不应自行拼接
`tools/`、`configs/` 或 `outputs/` 的位置。

## 材质工具包主要目录

下表按功能列出主要目录，不是完整文件树。

```text
tools/qwen_material_pipeline/
├── workflows/       命令工作流
├── evidence/        相机、Part-ID、颜色、PBR 与质量测量
├── retrieval/       SigLIP2 和 DINOv2 检索
├── materials/       MDL 目录、选择与应用规则
├── segmentation/    SAM3 分割与标注重放
├── mvinverse/       MVInverse 适配器和运行记录
├── qwen/            本地与远程 VLM 适配器
├── usd/             零件索引、渲染、应用与验证
├── core/            共用数据结构
├── configs/         工具包配置
├── schemas/         JSON Schema
├── web/             标注和结果查看器
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
- `apps/` 可以读取已发布结果，但不能被 `asset_pipeline` 导入。
- 只有可重建缓存可以直接删除；删除输入、分析记录、模型或 USD 前应先确认是否需要保留。

在当前环境中安装两个 Python 包：

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

安装后可使用 `hunyuan-asset-pipeline`、`manual-material-pipeline` 和
`qwen-material`。根目录的 `run_*.py` 脚本仅用于兼容旧命令。
