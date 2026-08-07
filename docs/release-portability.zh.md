# 发布与可移植性

[English](./release-portability.md) | [中文](./release-portability.zh.md) | [文档索引](./README.zh.md)

配置好本机运行时、模型、密钥和挂载目录后，项目应能从任意检出目录运行。源码不能依赖
某位开发者的用户名或安装路径。

## 路径

- 项目文件使用仓库相对路径。
- 外部程序、缓存和 NVIDIA 资产通过 `.env` 中的 `BLENDER_BIN`、`ISAAC_PYTHON`、
  `MODEL_CACHE_ROOT`、`VISUAL_MATERIAL_ROOT` 等变量传入。
- 本地模型同样集中在 `.env`：Qwen 使用 `QWEN_MODEL_PATH` / `QWEN35_MODEL_PATH`，
  MVInverse 使用 `MVINVERSE_REPOSITORY` / `MVINVERSE_CHECKPOINT`，SAM3 使用
  `SAM3_REPOSITORY` / `SAM3_CHECKPOINT`，SAM3D 使用 `SAM3D_MOGE_CHECKPOINT` /
  `SAM3D_DINOV2_REPOSITORY` / `SAM3D_DINOV2_CHECKPOINT`，材质检索使用
  `SIGLIP2_MODEL_PATH` / `DINOV2_MODEL_PATH`。
- 运行时固定使用 `PIPELINE_LOCAL_MODELS_ONLY=1`；正常推理不下载权重，缺失或不完整的
  路径会明确失败。Hunyuan 生成和 ReduceFace 是云 API，不受该本地权重约束。
- 不提交绝对软链接，也不把某次运行的绝对路径写入版本化配置和报告。
- `/workspace`、`/isaac-sim`、`/opt/blender`、`/opt/conda` 和 `/home/pipeline`
  是容器内路径。迁移主机时修改卷挂载的宿主机一侧，不修改这些容器目标路径。

## 各类文件的发布方式

| 类别 | 源码包 | 提供方式 |
| --- | --- | --- |
| 自研源码、配置、测试和文档 | 包含 | 随源码发布 |
| Qwen、MVInverse、SAM3/SAM3D、MoGe、DINOv2、SigLIP2 和 NVIDIA 材质资产 | 不包含 | 按各自许可下载或只读挂载 |
| 随仓库提供的第三方源码 | 逐项审核 | 仅在允许重新分发时包含 |
| 密钥和本机 `.env` | 不包含 | 从模板创建本机文件 |
| `downloads/`、`outputs/`、`results/`、`var/`、`workspace/` | 不包含 | 在 Git 忽略路径下生成 |
| 缓存、日志和构建产物 | 不包含 | 在本机重建 |
| Docker 离线分卷 | 独立发布 | 连同 SHA-256 清单一起提供 |

`tools/qwen_material_pipeline/` 是材质工具包的唯一目录。与根包一起安装：

```bash
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

## 密钥

仓库只保留模板。真实云密钥应放在 Git 忽略的本机文件或当前进程环境中，不能写入
Dockerfile、镜像层、测试数据、日志或结果包。发现可能泄露的密钥后应立即吊销。

## 检查

在项目根目录执行：

```bash
python ./tools/release/check_public_tree.py
git diff --check
conda run -n hunyuan_sam3d \
  python -m pytest -q tests/test_pipeline_structure.py \
  -k publishable_tree
```

自动扫描不能判断所有权和隐私。请从审核完成的 Git tag 构建发布包，并执行
[公开发布检查表](./public-release-checklist.zh.md)。

Docker 离线分卷需单独验证：

```bash
cd docker/offline-images
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

加载与验收方法见 [Docker 操作手册](../docker/README.zh.md)。
