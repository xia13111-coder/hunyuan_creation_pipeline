# Isaac Sim 物理模块

[English](./physics.md) | [中文](./physics.zh.md) | [文档索引](../README.zh.md)

该模块在 USD 上挂载 collision、rigid body、物理材质和质量，再把 USD 及其材质/贴图依赖收集到最终目录。视觉 PBR 和 PhysX 材质是两套独立数据。

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

`--material` 必须显式选择 `materials.json` 中的 preset。代码不再根据 prim 或视觉材质名称自动猜测材质。

| 字段 | 单位/类型 | 作用 |
| --- | --- | --- |
| `density` | kg/m^3 | 未传 `--set-mass` 时，用体积乘密度估算质量。 |
| `friction` | 系数 | 静摩擦系数。 |
| `dyn_ratio` | 比例 | 动摩擦为 `friction * dyn_ratio`；缺省为 `0.9`。 |
| `restitution` | 系数 | 回弹系数。`0` 接近无回弹。 |
| `combine` | 字符串 | 可选 friction/restitution combine mode，例如 `average`。 |

当前 preset：

| 材质 | 密度 | 静摩擦 | 动摩擦 | 回弹 |
| --- | ---: | ---: | ---: | ---: |
| `plastic` | 950 | 0.30 | 0.27 | 0.10 |
| `steel` | 7850 | 0.50 | 0.45 | 0.05 |
| `rubber` | 1100 | 1.00 | 0.70 | 0.60 |
| `wood` | 700 | 0.60 | 0.54 | 0.20 |
| `copper` | 8960 | 0.40 | 0.36 | 0.05 |

## Collision Approximation

| 模式 | 适用场景 | 取舍 |
| --- | --- | --- |
| `sdf` | 动态复杂/凹形 mesh，当前推荐默认值。 | 凹形精度好，但 cooking 成本较高；手工 CAD 还会启用 SDF remeshing。 |
| `convexHull` | 简单且接近凸形的物体。 | 快且稳定，凹槽会被填平。 |
| `convexDecomposition` | 需要凹形近似但不使用 SDF。 | 多个凸包更准确，计算和碰撞成本更高。 |
| `boundingCube` | 粗略盒状代理。 | 最快，精度低。别名 `box`、`cube`。 |
| `boundingSphere` | 球形物体。 | 快且稳定，不适合细长资产。别名 `sphere`。 |
| `sphereApproximation` | 接受 PhysX 球体近似的资产。 | 简单代理。 |
| `triangleMesh` | 静态环境。 | 动态刚体会自动改为 `sdf`。 |
| `meshSimplification` | 静态简化网格。 | 动态刚体会自动改为 `sdf`。 |

主 pipeline 通过 `--approx` 选择模式。直接调用 `add_physics.py` 时还可以设置 SDF、VHACD、contact offset 和 solver iteration 参数。

直接脚本的 `--sdf-res` 默认是 `256`。手工 STEP/STP workflow 为降低复杂 CAD 装配体的物理开销，会显式传入 `--manual-sdf-resolution`，默认值为 `32`。

挂载 SDF collider 之前，脚本会检查 mesh 拓扑。如果发现边界边、非流形边、共享边面序不一致、无效面或接近零的封闭体积，脚本会输出警告，因为 PhysX cooking 可能失败或生成质量较差的碰撞体。

`--approx` 采用严格语义：请求 `--approx sdf` 时，最终写入的 `physics:approximation` 必须保持为 `sdf`，拓扑诊断不会再把它改成 `convexDecomposition`。直接脚本中的 `--force-sdf` 只用于覆盖规则或其它命令行设置提供的不同碰撞模式。

手工 STEP/STP workflow 会额外传入 `--sdf-remesh`，写入 `physxSDFMeshCollision:sdfEnableRemeshing = true`。PhysX 会在计算 SDF 前重新构造有问题的 CAD tessellation，用于处理面序不一致、开放壳体和自相交。与干净闭合 mesh 直接生成的 SDF 相比，重网格后的碰撞表面可能有少量精度损失。

## 质量

- `--set-mass` 表示整个资产总质量，单位 kg。
- 多个 rigid body 时，总质量按各 body 的几何体积权重分配。
- 不传时，脚本计算 mesh 世界空间体积并乘以材质密度。
- 体积无效或极小时使用安全 fallback，避免写入负质量或零质量。

## 几何准备

在挂载物理 API 之前，`prepare_geometry_for_physics` 依次处理：

```text
normalize_stage_units_to_meters
-> deinstance_visible_subtree
-> center_stage_geometry_at_origin       # 手工 CAD 路径
-> center_mesh_local_origins
-> fix_inverted_mesh_winding
```

`center_mesh_local_origins` 移动 mesh points，并用 `xformOp:translate:meshLocalOrigin` 补偿视觉位置。

`fix_inverted_mesh_winding` 会先把 `leftHanded` 统一为 `rightHanded`，再应用完整的 local-to-world transform 计算 signed volume。因此它既能发现 mesh 自身整体反向，也能发现父级镜像 transform 导致的世界空间反向。封闭体积为负时，脚本会反转所有 face index 顺序，并清除已经失效的 authored normals，然后才进行 PhysX collision cooking。

常见诊断日志：

```text
fixed inverted world-space mesh winding: 3 mesh(es)
[WARN] SDF topology warning on /Asset/Part/Mesh: boundary=12, nonmanifold=0, inconsistent=2, invalid_faces=0; keeping requested sdf
SDF remeshing enabled on /Asset/Part/Mesh
```

第一行表示面序已经修复，USD 层级和模型位置保持不变。第二行表示仍然保留用户请求的 SDF；如果 PhysX 随后报告 cooking 错误，应继续检查该 mesh。

## 最终收集

`collect_usd_flat.py` 为每个 physics USD 创建独立目录，复制主 USD、SubUSD、材质和贴图，并重写必要的相对资源路径。输出目录可以直接作为仿真资产目录使用。
