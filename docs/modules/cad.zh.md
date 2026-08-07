# 手工建模 STEP/STP（CAD）模块

[English](./cad.md) | [中文](./cad.zh.md) | [文档索引](../README.zh.md)

手工建模 STEP/STP（CAD）流程接收 STEP/STP，并直接通过 Isaac Sim/Omniverse CAD Converter 转为 USD。
它保留装配体层级，不经过 Blender/GLB 中转。

## 代码和流程

| 代码 | 职责 |
| --- | --- |
| `asset_pipeline/manual_cad.py` | 编排转换、物理、可选视觉材质、依赖收集和验证。 |
| `asset_pipeline/jobs/cad.py` | 校验 STEP/STP 并运行 CAD Converter。 |
| `asset_pipeline/visual_materials/` | 根据参考图分配视觉材质。 |
| `asset_pipeline/jobs/isaac.py` | 准备几何、写入 PhysX 数据并收集依赖。 |
| `asset_pipeline/jobs/delivery.py` | 验证赋材质后及最终收集的 USD。 |

```text
run_manual_cad_workflow
-> run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> run_add_physics_job(center_origin=True)
   -> 归一单位和局部原点、修复面序、写入碰撞
-> 可选 run_assign_visual_materials_job
   -> 相机对齐、提取 Part-ID 外观信息、选择 NVIDIA Base MDL
-> tools/isaac/collect_usd_flat.py
-> 检查结构、依赖和最终渲染
```

`run_stp_physics_job`、`run_manual_cad_job` 和 `asset_pipeline/jobs/material.py` 只用于兼容
旧调用。新代码应使用 `run_manual_cad_workflow` 和
`asset_pipeline.visual_materials.run_assign_visual_materials_job`。

## 命令

先按 [Part-ID 快速开始](../manual-part-id-materials.zh.md) 生成 SAM3 整机前景标注，再运行：

```bash
hunyuan-asset-pipeline \
  --manual-stp ./input/manual_asset.stp \
  --cad-usd-output-dir ./outputs/manual/asset_run/cad_usd \
  --intermediate-output-dir ./outputs/manual/asset_run/intermediate \
  --final-output-dir ./outputs/manual/asset_run/final \
  --auto-visual-materials \
  --visual-reference front=./references/front.jpg \
  --visual-reference side=./references/side.jpg \
  --visual-reference top=./references/top.jpg \
  --visual-reference iso=./references/iso.jpg \
  --visual-foreground-annotations ./annotations/sam3_foreground_annotations.json \
  --visual-material-output-dir ./outputs/manual/asset_run/visual_material \
  --acknowledge-mvinverse-noncommercial \
  --allow-policy-material-fallback \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32 \
  --set-mass 30
```

`--manual-stp` 也可以是目录。转换器会递归查找 `.stp` 和 `.step`，并保留相对目录结构。
只添加物理时可以批处理多个文件；启用参考图赋材质时，一次只能处理一个 CAD 资产。

## 主要参数

| 参数 | 说明 |
| --- | --- |
| `--manual-stp` | STEP/STP 文件或目录。 |
| `--cad-usd-output-dir` | CAD Converter 原始输出；默认 `<input>_cad_usd`。 |
| `--cad-converter-option KEY=VALUE` | 额外转换选项，可重复。 |
| `--intermediate-output-dir` | 完成物理准备后的 USD 输出。 |
| `--final-output-dir` | 收集依赖后的最终资产目录。 |
| `--auto-visual-materials` | 在几何准备后、收集依赖前自动赋材质。 |
| `--visual-reference [ID=]IMAGE` | 同一资产的参考图，提供 2–4 个视角。 |
| `--visual-foreground-annotations` | 根据这些图片生成的 SAM3 整机前景标注。 |
| `--visual-material-output-dir` | 材质分析结果和赋材质后的 USD 输出。 |
| `--acknowledge-mvinverse-noncommercial` | 确认本次运行符合 MVInverse 许可。 |
| `--allow-policy-material-fallback` | 为未观测或无法判断的零件使用配置中的预设默认材质。 |
| `--material` | `materials.json` 中的物理材质。 |
| `--approx` | 碰撞近似；复杂动态 CAD 推荐 `sdf`。 |
| `--manual-sdf-resolution` | 手工建模 STEP/STP 的 SDF 分辨率，默认 `32`。 |
| `--set-mass` | 总质量，单位 kg；不传时按体积和密度估算。 |

STEP/STP 不使用 `--len-x/y/z` 或 `--orientation`。任意缩放和包围盒旋转会改变工程尺寸
与装配语义；单位转换和原点居中则会保留它们。

## 处理细节

### 先准备几何，再判断视觉材质

物理几何准备在视觉材质推理之前完成。程序化 MDL 可能使用物体坐标；如果选材后再修改
单位或 Mesh 局部原点，纹理可能移动或缩放。因此所有材质渲染都使用同一份归一后的几何。

参考图赋材质一次处理一个 STEP/STP，并为每个 Mesh 赋值。可见零件使用各自的照片信息，
无法判断的零件使用配置中的预设默认材质。详见[参考图自动赋材质](./visual-materials.zh.md)。

### 几何和碰撞

CAD 转换保留装配层级和变换。物理阶段把单位转为米，通过补偿变换把可见资产居中，修复
方向明确的反向面并生成碰撞数据；开放或非流形网格只报告问题，不猜测修改。SDF 行为、
分辨率建议和修复方法见[物理处理](./physics.zh.md)。
