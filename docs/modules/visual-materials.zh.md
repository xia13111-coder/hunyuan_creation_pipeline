# STEP/STP 参考图自动赋材质

[English](./visual-materials.md) | [中文](./visual-materials.zh.md) | [快速开始](../manual-part-id-materials.zh.md)

该流程根据同一工件的 2–4 张照片，为手工建模 STEP/STP 装配体选择 NVIDIA Base MDL。
CAD 始终决定几何、层级和尺寸，照片只为可见 Part-ID 提供外观信息。

## 流程

```text
STEP/STP 转换
-> 物理几何准备
-> 建立零件清单和参考渲染
-> 相机对齐
-> 提取每个可见 Part-ID 的外观信息
-> NVIDIA Base 检索、Qwen 排序和 MDL 渲染比较
-> 为每个 Mesh 应用一个材质
-> 收集最终 USD 并验证
```

最终 CAD 不会为了匹配照片而缩放、旋转或变形。分析阶段估计的相机和二维修正不会写入
USD。

### 相机搜索与正式主线

`calibrate-cameras` 可以使用 Kaolin CUDA 光栅器加速候选搜索；它一次加载完整 CAD，只生成
Part-ID、轮廓和遮挡关系，不移动任何单独 Mesh，也不使用材质或灯光信息。每个视角排名最高
的候选仍会交给 Isaac Sim 以最终分辨率重新渲染。

`calibrate-cameras --fast-search` 支持三种模式：`auto` 默认启用并在可选运行时不可用时回退，
`required` 要求快速后端可用，否则停止，`disabled` 完整使用旧 Isaac 候选搜索。报告中的
`candidate_search.fast_raster_audit` 记录资产哈希、三角形数量、候选数量和耗时；
`candidate_search.full_resolution_verification` 绑定快速候选与 Isaac 最终复核结果。

正式材质主线的生产配置固定为 `camera_fast_search: disabled`。原因是 Part-ID 材质身份依赖
四个参考视角都通过相机和空间配准；快速候选缺失一个视角时，后续小零件会被错误当作不可见
零件。生产主线因此优先保证证据完整性，快速模式只保留给独立实验和明确选择它的配置。

相机的全局种子和各阶段候选使用同一套几何门禁排序：先排除需要超出合同的二维尺度、旋转
或平移才能重合的候选，再比较轮廓 IoU、边界残差和结构分数。相机标定与空间 Part-ID 投影
共同使用参考清单中封存的整机前景掩码；跨分辨率比较会先去除栅格尺寸带来的尺度和平移量。
这些规则不包含工件名称、视角角度或 Part-ID 特例，适用于任意完整 CAD 装配体。

相机结果同时封存两个路径无关的内容指纹：输入指纹由 CAD 几何、实际 Part-ID 像素、参考图/
掩码像素和搜索配置组成；解指纹由最终投影模型、内参和外参组成。完全并列的候选先选择需要
最少整机二维残差修正的解，再按规范化相机参数排序，候选生成顺序和临时 view ID 不参与决策。
高分辨率阶段、断点复用和下游材质证据都必须验证当前文件哈希、目标函数版本以及两阶段种子
哈希。输入发生变化时会明确重新标定，不能静默把旧相机、旧掩码或旧 Part-ID 证据混进新运行。

`asset_pipeline/visual_materials/` 负责编排；`tools/qwen_material_pipeline/` 提供分割、
图像信息提取、检索、模型调用和 USD 工具；Isaac Sim 负责 CAD/USD、MDL 渲染、物理和最终
验证。详见[架构](../architecture.zh.md)。

## 如何提取图像信息

SAM3 页面保存每张照片中已确认的整机前景。正式运行先校验图片哈希，再为每个视角估计
一台分析相机。

每个 CAD Part-ID 会独立投影。全局相机先给出可见区域和粗略框；流程随后用同一台封存相机
把目标 mesh 单独投影成完整的 amodal 形状模板。同一份 CAD 请求共同引导 SAM3 和类无关
EntitySeg。最终结果不再从两者中二选一或直接复制任一模型掩码；通过安全门的 SAM3/EntitySeg
结果只作为迭代初值。isolated mesh 约束完整形状，装配状态的 Part-ID 投影决定当前视图的可见性
和遮挡归属，优化器在自动按零件尺度生成的窄带内反复贴合照片边缘，并以三种一致性的无权重
几何均值选择最优安全迭代。CAD Part-ID 始终决定零件身份。
每个视角的所有零件共享同一个整机相机残差，禁止单个 mesh 平移、旋转或缩放。局部贴合
不可靠时，流程会使用全局投影或把该零件标为未观测。这些修正只
用于提取照片中的外观信息，不改变 CAD 几何。

材质身份优先主线要求所有已注册参考视角都真实贡献可见且被选中的 Part-ID observation。
校验同时检查逐零件 observation 和汇总计数；任一视角被相机或空间配准淘汰时流程立即停止，
不会静默降级成两视角后继续选择材质。

| 组件 | 作用 |
| --- | --- |
| SAM3 | 提供整机前景和逐零件迭代初值。 |
| EntitySeg / CropFormer | 提供类无关迭代初值；必须先通过 CAD 安全门。 |
| 迭代边界优化器 | 联合完整 mesh、装配可见性、前序掩码和当前照片边缘；绝不单独移动 mesh。 |
| MVInverse | 估计区域内的基础色、粗糙度和金属度。 |
| SigLIP2 | 从 NVIDIA `Materials/Base` 检索外观相近的 MDL。 |
| DINOv2 | 比较局部表面和纹理外观。 |
| Qwen3.5 | 根据已有分析结果排序有限的候选列表。 |
| MDL 渲染比较 | 在已对齐的 CAD 视角中比较真实 MDL 效果。 |

模型只负责对材质库候选排序，不会直接修改 USD 绑定。

## 赋值和验证规则

- 每个 Mesh 必须得到一个可应用的视觉材质；
- 可见零件只使用该 Part-ID 自己通过检查的照片信息；
- 不可见或无法判断的零件使用预设默认材质，不把推测结果当作照片观测；
- 出现重复、不完整或仍需复核的赋值时，流程会在应用材质前停止；
- 记录胜出的 MDL 后，不再修改它的颜色、粗糙度、金属度、纹理或法线；
- 赋材质后的 USD 会与参考图比较；收集依赖后，最终 USD 还会重新打开并再次检查。任何
  一项失败都会停止流程，并指出失败阶段。

默认配置为
`tools/qwen_material_pipeline/configs/pipeline/manual_part_id_materials.json`，只搜索 NVIDIA
`Materials/Base`，并按 CAD Part-ID 独立决策。

## 运行环境和模式

| 运行环境 | 工作 |
| --- | --- |
| `hunyuan_sam3d` | 编排、SAM3、MVInverse、SigLIP2 和 DINOv2。 |
| Qwen3.5 Python 环境 | 本地视觉语言推理。 |
| Isaac Sim Python | CAD/USD、MDL 渲染、物理和验证。 |
| Blender | Blender 负责的网格和纹理操作。 |

根目录 `.env` 统一管理本地模型：

- Qwen / Qwen3.5：`QWEN_MODEL_PATH`、`QWEN35_MODEL_PATH`；
- MVInverse：`MVINVERSE_REPOSITORY`、`MVINVERSE_CHECKPOINT`；
- SAM3：`SAM3_REPOSITORY`、`SAM3_CHECKPOINT`；
- EntitySeg：`ENTITYSEG_PYTHON`、`ENTITYSEG_CROPFORMER_ROOT`、`ENTITYSEG_CONFIG`、`ENTITYSEG_CHECKPOINT`；
- SAM3D MoGe / DINOv2：`SAM3D_MOGE_CHECKPOINT`、`SAM3D_DINOV2_REPOSITORY`、`SAM3D_DINOV2_CHECKPOINT`；
- 材质检索：`SIGLIP2_MODEL_PATH`、`DINOV2_MODEL_PATH`。

运行时固定使用 `PIPELINE_LOCAL_MODELS_ONLY=1`，正常推理不下载权重；本地路径缺失或不完整就
明确失败。`VISUAL_MATERIAL_ROOT`、`NVIDIA_BASE_OBSERVATION_BANK` 和 `ISAAC_PYTHON`
也从 `.env` 读取。模型权重和 NVIDIA 资产不随源码发布。Hunyuan 云 API 是例外，
使用生成或 ReduceFace 时仍需网络。

EntitySeg 解释器会与用户级 Python 包隔离。安装外部 CropFormer/Detectron2
运行环境后，必须把主线维护的兼容依赖安装到同一个解释器中：

```bash
"$ENTITYSEG_PYTHON" -m pip install \
  -r tools/qwen_material_pipeline/requirements-entityseg.txt
PYTHONNOUSERSITE=1 "$ENTITYSEG_PYTHON" -c \
  'import black, cloudpickle, mmcv, yapf'
```

YAPF 的固定版本用于兼容 MMCV 1.x。仅能从 `~/.local` 导入不算有效安装，
因为主线子进程会设置 `PYTHONNOUSERSITE=1` 以保证运行可复现。

| 模式 | 用途 |
| --- | --- |
| `live` | 对新工件运行完整推理。 |
| `auto` | 有完全匹配的历史结果时复用，否则执行通用推理。 |
| `bundled` | 必须匹配已有历史结果，否则停止。 |

`--resume` 只继续同一个、通过哈希校验的任务。要从零运行，请使用新的输出目录。

标注 JSON 在所有模式下都用于确定参考图片。人工确认的前景掩码只用于 `live`；`auto` 或
`bundled` 匹配到历史结果时，使用该结果自带的分析数据。

## 排错

先找到第一个失败的 `[PROGRESS]` 阶段，再检查对应输出：

| 现象 | 检查内容 |
| --- | --- |
| 前景不完整 | SAM3 标注和掩码。 |
| CAD 与照片不重合 | 相机对齐叠加图。 |
| 可见零件使用了默认材质 | `part_id_reference_evidence.json`。 |
| Qwen 输出被拒绝 | `.raw`、`.parse.json` 和 schema 错误。 |
| 预期 MDL 没有胜出 | 候选检索结果和 MDL 渲染分数。 |
| 最终 USD 外观变化 | 已记录的材质选择和最终渲染对比。 |

主要输出位于
`RUN_ROOT/visual_material/{renders,analysis,visual_quality,final_visual_acceptance}`。

进一步说明：

- [材质子包](../../tools/qwen_material_pipeline/README.zh.md)
- [MVInverse](../../tools/qwen_material_pipeline/docs/mvinverse.zh.md)
- [Base observation bank](../../tools/qwen_material_pipeline/docs/base_material_observation_bank.zh.md)

MVInverse 仅限非商业用途；NVIDIA 资产和 Isaac Sim 采用各自许可，见
[第三方声明](../../THIRD_PARTY_NOTICES.md)。
