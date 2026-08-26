# 清理与交付规则

源码、第三方代码、本机依赖和运行结果必须分开管理。

| 位置 | 内容 | Git / 交付规则 |
|---|---|---|
| `src/qwen_material_pipeline/` | 生产代码、配置、Schema、网页源码 | 保留并测试 |
| `tests/`、`docs/`、`scripts/`、`requirements/` | 测试与维护文件 | 保留并同步更新 |
| `third_party/` | 固定版本第三方源码与许可证 | 按上游许可交付 |
| `runtime/` | 本机模型、缓存、私有封存项目 | Git 忽略，不进入源码包 |
| 仓库 `outputs/` | 每次运行的图片、日志、分析和 USD | Git 忽略，按需单独交付 |

可直接清理 `__pycache__/`、`*.pyc`、`.pytest_cache/`、`.ruff_cache/`、`build/`、
`dist/` 和无引用的可重建缓存。不要删除仍需复验的输入、清单、推理记录、模型或 USD。

发布前必须：

1. 确认密钥、客户图片、私有 CAD、模型权重、绝对路径和绝对软链接未进入源码包。
2. 保留版本化配置、Schema、第三方许可证和依赖锁定信息。
3. 运行根测试、材质包测试、`git diff --check` 和公开树检查。
4. 将结果从 `outputs/<run-id>/` 单独审核和交付，不复制回源码目录。

`tools/qwen_material_pipeline/` 是材质包唯一实体位置，不创建根目录副本或兼容软链接。
仓库统一发布规范见[发布与可移植性](../../../../docs/release/release-portability.zh.md)。
