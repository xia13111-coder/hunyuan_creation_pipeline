# 公开发布检查表

[English](./public-release-checklist.md) | [中文](./public-release-checklist.zh.md) | [文档索引](../README.zh.md)

每次发布源码都应执行本检查表。自动检查不能替代对文件所有权、隐私和许可证的人工审核。

## 内容与许可证

- 确认所有自研文件都可以采用 Apache-2.0 发布。
- 保留 `LICENSE`、`NOTICE`、`legal/THIRD_PARTY_NOTICES.md`，以及随包提供的第三方源码许可证。
- 不发布模型权重、NVIDIA 资产、Isaac Sim 文件、Blender 二进制和生成结果，除非对应
  许可证明确允许重新分发。
- 公开运行结果时，记录模型来源、版本、哈希和许可证。除非另有授权，使用 MVInverse
  的场景仍受非商业条款限制。

## 隐私与密钥

- 检查照片、CAD/USD/GLB、掩码、截图、日志和 JSON 报告，确认不含保密信息或个人信息。
- 检查当前文件和 Git 历史，确认没有 `.env`、密钥、私有服务地址、用户名、本机路径或
  私有仓库地址。
- 密钥一旦进入提交、发布目录、镜像构建上下文、上传文件或共享日志，应立即吊销。
- 如需发布 `apps/material_audit_web/`，应单独审核。根源码包默认排除该嵌套应用及其本机
  审核数据。

## 可复现性

- 在干净机器或容器中验证安装步骤。
- 示例命令使用仓库相对路径。
- 模型和 NVIDIA 材质通过本机配置或只读挂载提供，不会自动打入源码包。
- 发布容器或离线镜像时，生成 SBOM 和依赖许可证清单。

## 验证

在仓库根目录执行：

```bash
python ./tools/release/check_public_tree.py
git diff --check
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

记录跳过的集成测试及缺少的运行时或模型。发布二进制或容器时，还应在最终交付物中执行
验收测试。

## 构建源码包

从审核完成的 Git tag 构建，不要直接压缩当前工作目录：

```bash
VERSION=0.1.0  # 替换为本次发布版本
git rev-parse --verify "v${VERSION}^{commit}"
git archive --format=tar.gz \
  --prefix="hunyuan_creation_pipeline-${VERSION}/" \
  -o "hunyuan_creation_pipeline-${VERSION}.tar.gz" \
  "v${VERSION}"
sha256sum "hunyuan_creation_pipeline-${VERSION}.tar.gz"
```

上传前解压并检查。发布说明应包括支持的输入和运行时版本、用户可见变化、迁移步骤、
已知限制、测试覆盖、源码包哈希，以及第三方许可证或模型变化。
