# Blender 后处理模块

[English](./blender.md) | [中文](./blender.zh.md) | [文档索引](../README.zh.md)

该模块处理精修后的 GLB：按包围盒方向对齐、缩放到目标尺寸、移动几何中心并导出 Z-up USD。它不负责 Hunyuan API 或 PhysX 属性。

## 代码和顺序

```text
asset_pipeline/workflows.run_postprocess_job
-> jobs.blender.blender_preflight
-> jobs.blender.run_align_job
   -> tools/blender/align_glb_axis_only.py
-> jobs.blender.run_resize_job
   -> tools/blender/resize_glb_xyz_and_center.py
-> jobs.blender.run_convert_job
   -> tools/blender/convert_glb_to_usd_zup.py
```

Blender 由 `BLENDER_BIN` 指定。预检会检查可执行文件、版本以及输入目录中的 GLB 数量。

GLB 转换成功后，如果启用 `--auto-visual-materials`，转换后的 USD 会交给
[`asset_pipeline/visual_materials`](./visual-materials.zh.md) 赋材质，物理处理阶段使用通过验证的
结果。未启用时，转换结果直接进入物理处理阶段。

这段只描述 GLB 后处理分支。手工建模 STEP/STP 不经过 Blender，其顺序是 CAD USD →
几何准备 → Qwen/MVInverse 选材 → 收集依赖。

## 坐标轴映射

`--orientation` 使用 `X=<rank>,Y=<rank>,Z=<rank>`：

| 符号 | 含义 |
| --- | --- |
| `L` | 当前模型包围盒最长维度 |
| `M` | 中间维度 |
| `S` | 最短维度 |

常用映射：

| 值 | 结果 |
| --- | --- |
| `X=L,Y=M,Z=S` | 最长边尽量沿 X，中间边沿 Y，最短边沿 Z。 |
| `X=M,Y=L,Z=S` | 交换两个水平维度。 |
| `X=L,Y=S,Z=M` | 请求中间维度作为 Z。仅当原始竖直维度排名允许时才能完全满足。 |

当前脚本只绕世界 Z 旋转 `0`、`90`、`180`、`270` 度。它不会把模型绕 X/Y 翻倒，因此不会为了满足映射而交换原始竖直轴和水平轴。

## 尺寸和居中

| 参数 | 说明 |
| --- | --- |
| `--len-x` | 目标 X 尺寸，单位 m。 |
| `--len-y` | 目标 Y 尺寸，单位 m。 |
| `--len-z` | 目标 Z 尺寸，单位 m。 |

`resize_glb_xyz_and_center.py` 按三个目标尺寸做非均匀缩放，并把整个资产的可见几何包围盒中心移动到世界原点。对齐在缩放之前执行，因此尺寸对应的是对齐后的世界 X/Y/Z。

## USD 转换

`convert_glb_to_usd_zup.py` 使用 Blender USD 导出器：

- USD 场景写入 `upAxis = Z`。
- 流程默认输出 `.usd`。
- 输入为单文件时创建同名输出目录；输入目录时处理目录中的 GLB。
- 视觉材质和贴图由 Blender USD 导出器写到 USD 及其资源目录。

## 与 CAD 的区别

STEP/STP 不经过本模块。CAD 路径必须保留装配体变换和层级，因此直接在 Isaac Sim 中完成 CAD 转 USD、单位归一化和原点清理。

## 输出到物理处理阶段

`run_convert_job` 返回 `usd_input_path`。流程只把该 USD 文件或目录传给
`jobs.isaac.run_add_physics_job`，避免处理同目录中的无关 USD。
