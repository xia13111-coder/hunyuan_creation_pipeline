# 自动材质工具包

[English](./README.md) | [中文](./README.zh.md)

本目录包含 STEP/STP 自动赋材质的内部实现：分割、检索、材质选择、USD 绑定和验证。
普通用户只需按[自动赋材质](../../docs/guides/manual-part-id-materials.zh.md)运行
`manual-material-pipeline`，不需要单独调用这里的子命令。

开发调试：

```bash
qwen-material --help
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

代码结构见[架构说明](../../docs/development/architecture.zh.md)。MVInverse 仅限非商业用途；其他许可见
[第三方声明](../../legal/THIRD_PARTY_NOTICES.md)。
