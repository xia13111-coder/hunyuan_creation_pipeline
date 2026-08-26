# Isaac Sim 物理模块

[English](./physics.md) | [中文](./physics.zh.md) | [文档索引](../README.zh.md)

该模块为 USD 添加碰撞体、刚体、物理材质和质量，再把 USD 及其材质、贴图依赖收集到
最终目录。视觉 PBR 和 PhysX 材质是两套独立数据。

手工建模 STEP/STP 启用参考图赋材质时，`add_physics.py` 会先归一单位和局部原点、修复面序，
并写入碰撞体、刚体、质量和物理材质。Qwen/MVInverse 随后在这份几何上
选择外观 MDL。视觉 `allPurpose` 绑定和用于 PhysX 的材质仍是两套独立数据。

## 代码

```text
asset_pipeline/jobs/isaac.py
  run_add_physics_job
  -> tools/isaac/add_physics.py

  run_collect_job
  -> tools/isaac/collect_usd_flat.py
```

`ISAAC_PYTHON` 必须指向当前 Isaac Sim 安装的 `python.sh`。`tools/isaac/isaac_sim_compat.py` 兼容 Isaac Sim 5/6 的 `SimulationApp` import 路径。

## 物理材质

`--material` 必须显式选择
[`configs/physics/materials.json`](../../configs/physics/materials.json) 中的预设。代码不会根据
prim 名称或视觉外观猜测物理材质。

| 字段 | 单位/类型 | 作用 |
| --- | --- | --- |
| `density` | kg/m^3 | 未传 `--set-mass` 时，用体积乘密度估算质量。 |
| `friction` | 系数 | 静摩擦系数。 |
| `dyn_ratio` | 比例 | 动摩擦为 `friction * dyn_ratio`；缺省为 `0.9`。 |
| `restitution` | 系数 | 回弹系数。`0` 接近无回弹。 |
| `combine` | 字符串 | 摩擦系数和恢复系数的组合方式，例如 `average`。 |

配置包含通用材质，以及更具体的金属、聚合物、弹性体、木材、玻璃、陶瓷和混凝土。
这些值是仿真基线；已知具体牌号和接触面时，应使用实测数据。当前数值以
`configs/physics/materials.json` 为准，文档不再复制一份容易过期的表格。

## 碰撞近似

| 模式 | 适用场景 | 取舍 |
| --- | --- | --- |
| `sdf` | 动态、复杂或凹形网格，当前推荐默认值。 | 凹形精度好，但碰撞数据生成成本较高；手工建模 STEP/STP 还会启用 SDF 重网格。 |
| `convexHull` | 简单且接近凸形的物体。 | 快且稳定，凹槽会被填平。 |
| `convexDecomposition` | 需要凹形近似但不使用 SDF。 | 多个凸包更准确，计算和碰撞成本更高。 |
| `boundingCube` | 粗略盒状代理。 | 最快，精度低。别名 `box`、`cube`。 |
| `boundingSphere` | 球形物体。 | 快且稳定，不适合细长资产。别名 `sphere`。 |
| `sphereApproximation` | 接受 PhysX 球体近似的资产。 | 简单代理。 |
| `triangleMesh` | 静态环境。 | 动态刚体会自动改为 `sdf`。 |
| `meshSimplification` | 静态简化网格。 | 动态刚体会自动改为 `sdf`。 |

主流程通过 `--approx` 选择模式。直接调用 `add_physics.py` 时还可以设置 SDF、VHACD、
接触偏移和求解器迭代参数。

直接脚本的 `--sdf-res` 默认是 `256`。手工建模 STEP/STP 流程为降低复杂装配体的物理开销，
会传入默认值为 `32` 的 `--manual-sdf-resolution`。

创建 SDF 碰撞体之前，脚本会报告边界边、非流形几何、面序不一致、无效面和接近零的
体积。请求 `--approx sdf` 时仍会写入 `sdf`，诊断不会自动换成其他模式。手工建模 STEP/STP
路径会启用 SDF 重网格来处理常见 CAD 三角化问题，但结果可能不如干净闭合 Mesh 精确。
底层 `--force-sdf` 只用于覆盖另一个明确指定的碰撞模式。

## 质量

- `--set-mass` 表示整个资产总质量，单位 kg。
- 存在多个刚体时，总质量按各刚体的几何体积权重分配。
- 不传时，脚本计算网格的世界空间体积并乘以材质密度。
- 体积无效或极小时使用安全默认值，避免写入负质量或零质量。

## 几何准备

在挂载物理 API 之前，`prepare_geometry_for_physics` 依次处理：

```text
normalize_stage_units_to_meters
-> deinstance_visible_subtree
-> center_stage_geometry_at_origin       # 手工建模 STEP/STP 路径
-> center_mesh_local_origins
-> fix_inverted_mesh_winding
```

`center_mesh_local_origins` 移动网格顶点，并用 `xformOp:translate:meshLocalOrigin` 补偿视觉位置。

手工建模 STEP/STP 路径先做几何准备，再进行视觉材质推理。程序化 MDL 可能使用物体坐标；如果
材质比较后才修改单位或局部原点，纹理可能移动或缩放。因此材质比较和最终渲染都使用
同一份归一几何。它保留 CAD 原始尺寸，也不需要目标尺寸输入。

`fix_inverted_mesh_winding` 先把 `leftHanded` 统一为 `rightHanded`，再应用完整世界变换
计算有符号体积。它既能发现 Mesh 自身反向，也能发现父级镜像变换造成的反向。闭合 Mesh
体积为负时，脚本会反转面索引并清除失效法线，再进行 PhysX 碰撞计算。

面序修复会保留 USD 层级和模型位置。如果出现 SDF 拓扑警告后 PhysX 仍报告碰撞数据生成
错误，应检查或修复对应源 Mesh。

## 最终收集

`collect_usd_flat.py` 为每个物理 USD 创建独立目录，复制主 USD、子 USD、材质和贴图，并重写必要的相对资源路径。输出目录可以直接作为仿真资产目录使用。

启用参考图赋材质时，收集依赖后还会再次验证。流程会重新打开并渲染赋材质后的 USD 和
最终 USD，检查每个 Mesh、`GeomSubset` 绑定和 NVIDIA MDL 默认参数，并确认全部 MDL
依赖都能从最终目录解析。任一检查失败都会停止流程。
