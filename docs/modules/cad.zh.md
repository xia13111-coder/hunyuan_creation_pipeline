# 手工建模 STEP/STP（CAD）模块

[English](./cad.md) | [中文](./cad.zh.md) | [文档索引](../README.zh.md)

该模块通过 Isaac Sim/Omniverse CAD Converter 把 STEP/STP 直接转成 USD，保留装配层级，
不经过 Blender 或 GLB 中转。

## 调用关系

| 代码 | 职责 |
| --- | --- |
| `asset_pipeline/manual_cad.py` | 编排 CAD 转换、物理处理、可选自动赋材质和最终验证。 |
| `asset_pipeline/jobs/cad.py` | 校验输入并调用 CAD Converter。 |
| `asset_pipeline/visual_materials/` | 根据参考图选择并绑定视觉材质。 |
| `asset_pipeline/jobs/isaac.py` | 准备几何、碰撞和 USD 依赖。 |
| `asset_pipeline/jobs/delivery.py` | 检查最终 USD。 |

```text
run_manual_cad_workflow
-> CAD 转 USD
-> 几何与物理处理
-> 可选：参考图自动赋材质
-> 收集依赖并验证最终 USD
```

## 使用

基础 CAD 命令见项目 [README](../../README.zh.md)；参考图自动赋材质见
[自动赋材质](../guides/manual-part-id-materials.zh.md)。`--manual-stp` 可以是文件或目录；
启用参考图赋材质时，一次处理一个 CAD 资产。

STEP/STP 保留原始工程尺寸，因此不使用 `--len-x/y/z` 或 `--orientation`。

## 几何和碰撞

物理阶段把单位转为米、居中可见资产、修复方向明确的反向面并生成碰撞数据。开放或非流形
网格只报告问题，不猜测修改。SDF 设置和修复方法见[物理处理](./physics.zh.md)。
