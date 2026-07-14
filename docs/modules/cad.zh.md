# 手工 STEP/STP CAD 模块

[English](./cad.md) | [中文](./cad.zh.md) | [文档索引](../README.zh.md)

手工建模资产只走 STEP/STP 输入。该路径直接使用 Isaac Sim/Omniverse CAD Converter 生成 USD，保留装配体层级，不再经过 Blender GLB 往返转换。

## 代码和流程

```text
asset_pipeline.workflows.run_stp_physics_job
-> jobs.isaac.run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> 对每个 CAD USD:
   -> tools/isaac/add_physics.py --center-origin
   -> tools/isaac/collect_usd_flat.py
```

## 命令

```bash
python ./run_asset_pipeline.py \
  --manual-stp ./input/manual_asset.stp \
  --cad-usd-output-dir ./manual_cad_usd \
  --intermediate-output-dir ./manual_output_intermediate \
  --final-output-dir ./manual_output_final \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32 \
  --set-mass 30
```

`--manual-stp` 也可以传目录。脚本递归发现 `.stp` 和 `.step`，并在输出目录中镜像相对层级。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--manual-stp` | STEP/STP 文件或目录。 |
| `--cad-usd-output-dir` | CAD Converter 原始 USD 输出目录；不传时使用 `<input>_cad_usd`。 |
| `--cad-converter-option KEY=VALUE` | 传递给 Omniverse CAD converter 的额外 option，可重复。 |
| `--intermediate-output-dir` | 添加物理后的 USD 输出。 |
| `--final-output-dir` | 最终收集目录。 |
| `--material` | `materials.json` 中的显式物理材质。 |
| `--approx` | collision approximation；复杂动态 CAD 推荐 `sdf`，手工 CAD 路径会同时自动启用 SDF remeshing。 |
| `--manual-sdf-resolution` | 手工 CAD 的 SDF 分辨率，默认 `32`。值越高越能保留碰撞细节，但 cooking、内存和碰撞计算开销也越大。 |
| `--set-mass` | 整个资产总质量 kg；不传时按体积和密度估算。 |

`--len-x/y/z` 和 `--orientation` 不用于 CAD 路径，因为任意非均匀缩放和包围盒旋转会破坏装配体 transform 语义。

手工 CAD 不再沿用底层脚本的通用 SDF 默认值 `256`，而是使用 `32`，优先保证复杂装配体的实时性能。如果薄壁、孔洞或小结构的碰撞轮廓明显丢失，再依次提高到 `64`、`128` 或 `256`。该参数只影响手工 STEP/STP 路径，不改变混元和 SAM3D 路径。

## USD 层级

CAD Converter 生成的 Xform/Mesh 层级会保留。物理准备不会把所有 mesh join 成一个对象：

- 顶层资产 root 保持稳定。
- assembly 子节点 transform 保留。
- instance/prototype 在物理处理需要时做 deinstance。
- 物理材质放在资产 anchor 下，避免散落到 stage 根节点。

## 单位

最终 stage 使用：

```text
upAxis = Z
metersPerUnit = 1.0
```

如果 CAD Converter 输出以 mm 为单位，`normalize_stage_units_to_meters` 会把 mesh points、translate、pivot 等长度量按比例转换到 m，同时保持世界空间外观和装配关系。

## 世界原点和局部原点

物理阶段使用 `--center-origin`：

1. 计算整个可见资产的世界包围盒中心。
2. 在保持顶层 root transform 为零的前提下，把下层可见几何移动到世界原点。
3. 对每个 mesh，把 points 移到局部包围盒中心附近。
4. 写 `xformOp:translate:meshLocalOrigin` 补偿，因此模型视觉位置不变。

结果是顶层 XYZ transform 为 0，资产几何中心位于世界原点附近，每个 mesh 的局部 points 也不会带很大的偏置。

## 反向面和 PhysX

STEP/STP tessellation 可能产生反向面、开放壳体、非流形边，也可能在装配层级中包含镜像 transform。这些问题在双面渲染时不一定能看出来，但 PhysX 计算碰撞体积和质量时可能失败：

```text
PhysX error: attachShape ... negative mass
```

collision cooking 之前，pipeline 现在会执行以下检查：

1. 把 USD mesh orientation 从 `leftHanded` 统一为 `rightHanded`。
2. 统计边界边、非流形边、共享边面序不一致和无效面。
3. 对封闭 mesh 应用完整世界变换后计算 signed volume，父级镜像 transform 也会包含在内。
4. 世界空间体积为负时，反转该 mesh 所有 face index 顺序。
5. 请求 `--approx sdf` 时，最终保持 `sdf`，并启用 `sdfEnableRemeshing`，让 PhysX 在 SDF cooking 前修复有问题的 tessellation。

修复只调整拓扑方向，不会拍平装配层级，也不会改变模型的视觉位置。开放或非流形视觉几何不会被强制反面，因为这类几何没有可靠的内外方向。如果需要精确 SDF collision，仍建议从 STEP 源文件修复几何。

脚本不会静默替换用户请求的碰撞模式。拓扑警告会指出对应 mesh，SDF remeshing 会处理常见的 CAD 面序、自相交和开放壳体问题。如果 PhysX 仍然报告 cooking 错误，说明该 CAD 几何无法被重网格可靠修复，应回到建模软件中执行几何修复或实体缝合。
