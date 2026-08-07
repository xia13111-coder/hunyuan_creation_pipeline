# 清理与交付规则

源码、运行记录、本机依赖和生成结果应分开管理。源码包不能包含密钥、宿主机绝对路径、
Python 缓存、模型软链接或旧运行目录。

本页说明自动材质子系统的细则；整个仓库的统一发布规范见
[发布与可移植性](../../../../docs/release-portability.zh.md)。

## 目录分类

| 目录 | 含义 | 交付策略 |
|---|---|---|
| `core/`、`evidence/`、`materials/`、`mvinverse/`、`qwen/`、`usd/`、`workflows/` | 项目源码 | 保留 |
| `configs/`、`schemas/` | 运行配置和数据格式 | 保留，代码与文档同步更新 |
| `input_views/README.md` | 参考图输入规范 | 保留；实际客户图片不进入源码包 |
| `third_party/` | 固定第三方源码和许可 | 按上游再分发许可决定是否交付 |
| `models/` | 本机模型权重 | 不进入源码包；在目标机器单独配置 |
| `var/`、`workspace/` | 可重建目录和临时运行数据 | 清空，只保留 `.gitkeep` |
| `results/` | 本机验证结果 | 单独归档，不与源码包混装 |
| `web/result_viewer/` | 静态页面源码 | 保留；结果链接在服务启动时临时创建 |

## 可直接清理

- `__pycache__/`、`*.pyc`、`.pytest_cache/`、`.ruff_cache/`；
- `.coverage`、`htmlcov/`、`build/`、`dist/`、`*.egg-info/`；
- `.env.runtime`、失败下载、临时模型链接和未发布的运行目录；
- 已经单独归档且不再被引用的旧结果、预览和日志。

仍需复验的工件不能删除输入清单、推理记录或 PBR 结果图。清理后应运行测试，并扫描绝对
路径和绝对软链接。

## 发布要求

1. 运行配置只从无密钥模板复制，真实凭据不进入仓库或镜像。
2. 宿主机资源通过环境变量或 Docker 卷挂载提供。
3. 结果报告和 USD 依赖使用相对路径；Docker 内 `/workspace` 等固定路径属于容器接口。
4. 模型源码与权重的许可独立审核；MVInverse 当前仅允许非商业用途（non-commercial purposes）。
5. 结果包必须通过外观、物理属性和收集后验证，再单独发布。
6. `tools/qwen_material_pipeline/` 是唯一实体位置，不创建仓库根同名副本或兼容链接。
