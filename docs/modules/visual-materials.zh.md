# STEP/STP 参考图自动赋材质

[English](./visual-materials.md) | [中文](./visual-materials.zh.md) | [快速开始](../guides/manual-part-id-materials.zh.md)

该流程根据同一工件的 2–4 张照片，为手工建模 STEP/STP 装配体选择 NVIDIA Base MDL。
CAD 始终决定几何、层级和尺寸，照片只为可见 Part-ID 提供外观信息。

本文中，Part-ID 是 CAD 中的零件编号，Mesh 是零件中的网格对象，MDL 是 NVIDIA 材质文件，
掩码是图片中目标占据的像素区域，IoU 是两个区域的重合程度（越大越好）。“材质身份”只表示
选中了哪一种 MDL，不包含颜色。哈希或 SHA256 是用于判断文件内容是否变化的摘要。命令名和
输出字段保留英文，便于与程序输出直接对应。

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

`calibrate-cameras --fast-search` 支持三种模式：`auto` 默认启用安全混合搜索，`required` 要求
快速后端以及快速结果的最终验证都通过，否则停止，`disabled` 完整使用旧 Isaac 候选搜索。
正式材质主线使用 `camera_fast_search: auto`。每个视角会保留 12 个快速候选，由 Isaac 以最终
分辨率统一复核；系统逐视角检查候选数量、来源、IoU、综合目标和边界残差的一致性。某个视角
缺失或两种渲染器不一致时，只对这个视角自动重跑完整 Isaac 搜索，其余安全视角继续使用快速
结果。一次运行可以注册 2–4 个视角，但提交的每个视角都是硬约束，不能静默丢失；同时不需要
人工选择，也不会因单个坏视角让整个搜索退回慢速。

低清搜索会为每个参考视角输出经过 Isaac 复核的前三个相机；高分精调分别从这三个起始候选继续
优化，最后只启动一次 Isaac，对合并后的候选统一重排。这样不会让高成本阶段被单个局部最优
锁死，整个选择过程仍只依赖通用几何指标且结果可复现。

报告中的 `candidate_search.fast_raster_audit`（快速搜索检查记录）保存资产哈希、三角形数量、
候选数量和耗时；`candidate_search.full_resolution_verification` 关联快速候选与 Isaac 最终
复核结果。
`candidate_search.per_view_selected_backend` 和 `per_view_fallback` 分别记录每个视角采用的
计算方式和备用方案。

相机的初始候选和各阶段候选使用同一套几何检查规则排序：先排除需要超出允许范围的二维
尺度、旋转或平移才能重合的候选，再比较轮廓 IoU、边界残差和结构分数。相机标定与空间
Part-ID 投影共同使用参考清单中已确认并记录的整机前景掩码；跨分辨率比较会先去除栅格
尺寸带来的尺度和平移量。
这些规则不包含工件名称、视角角度或 Part-ID 特例，适用于任意完整 CAD 装配体。

相机结果同时记录两个与路径无关的内容摘要：输入摘要由 CAD 几何、实际 Part-ID 像素、
参考图、掩码像素和搜索配置组成；结果摘要由最终投影模型、内参和外参组成。候选分数完全
相同时，先选择整机二维修正量最小的结果，再按规范化相机参数排序。候选生成顺序和临时
view ID 不参与决策。
高分辨率阶段、断点复用和下游材质判断都必须验证当前文件哈希、目标函数版本以及两个阶段
起始候选的哈希。输入发生变化时会明确重新标定，不能静默把旧相机、旧掩码或旧 Part-ID
结果混进新运行。

`asset_pipeline/visual_materials/` 负责编排；
`asset_pipeline/visual_materials/stages/part_id_evidence.py` 单独负责逐 Part-ID 的模型图模板、
两遍分割、邻件关系引导和融合顺序。该阶段从 `orchestrator.py` 抽离后由一个可单独测试的
模块负责，但磁盘产物路径和阶段名称不变。`tools/qwen_material_pipeline/` 提供分割、
图像信息提取、检索、模型调用和 USD 工具；Isaac Sim 负责 CAD/USD、MDL 渲染、物理和最终
验证。详见[架构](../development/architecture.zh.md)。

## 如何提取图像信息

SAM3 页面保存每张照片中已确认的整机前景。正式运行先校验图片哈希，再为每个视角估计
一台分析相机。

每个 CAD Part-ID 会独立投影。全局相机先给出可见区域和粗略框；流程随后用同一组已确认的
相机参数，把目标零件单独投影成完整形状模板，包括被其他零件遮挡的部分。同一份 CAD 请求
共同引导 SAM3 和不依赖零件类别的 EntitySeg。最终结果不再从两者中二选一或直接复制任一
模型掩码；通过范围检查的结果只作为迭代优化的起点。单独渲染的目标零件约束完整形状，
装配状态的 Part-ID 投影决定当前视图的可见部分和遮挡关系。优化器在零件边缘附近反复贴合
照片边界，并综合三项一致性分数选择最好的一次结果。CAD Part-ID 始终决定零件身份。
目标 Part-ID 还会保留在 CAD 整机渲染图中的坐标和邻件关系。第一遍 SAM3/EntitySeg 只提供
用于定位的其他零件；定位某个目标时严格排除该目标自己的第一遍掩码。流程在 CAD 模型图的
统一坐标中，用其他零件估计二维位置、旋转和尺度变化，再根据多个近邻零件的位置误差推断
目标在参考图中的位置。
目标位置确定后才用 CAD 模型图中的完整 Part-ID 形状生成新框，并第二次运行 SAM3 与 EntitySeg。
第二遍结果、第一遍融合基线、完整模型形状、邻件禁入区和照片边缘共同进入迭代优化；不是在两个
分割模型中二选一，也不会把旧目标映射当作位置真值。神经候选全部拒绝时，邻件定位的 CAD 形状
仍会继续做照片边缘优化并输出结果。此时 CAD 形状只作为大致位置参考，不会再用它与自身的重叠率
形成 1.0 的硬门槛；局部相似变换由照片边缘增益与邻件净空的联合分数自动选择。真正来自
SAM3、EntitySeg 或前序融合的图像候选仍要通过“不能比原结果更差”的检查。可用于定位的
其他零件不足时，保留已经检查过的第一遍结果，因此不会中断整批运行。
对管、线缆、软管、杆和细导轨这类细长件，普通边缘分数可能只贴住单侧边缘或邻近金属边。
流程会根据 CAD 掩码的紧致度、骨架长度和包围框填充率自动识别这类几何，不使用 Part-ID
名单或材质提示；随后联合多尺度中心线亮脊、低饱和亮度、双侧边界和邻件净空选择二维相似
变换。只有照片中的结构信息和装配位置检查分数都严格提高，才采用新位置，否则保留原结果。
每个视角的所有零件共享同一个整机相机残差，禁止单个 mesh 平移、旋转或缩放。局部贴合
不可靠时，流程会使用全局投影或把该零件标为未观测。这些修正只
用于提取照片中的外观信息，不改变 CAD 几何。
SAM3 与 EntitySeg 的正式入口使用同一个固定随机数起点，并把该数值、请求哈希、模型权重
哈希和确定性算法设置写入清单；同一输入的重试不会因为随机初始化改变 Part-ID 判断结果。

材质身份优先主线要求所有已注册参考视角都提供真实、可见且已选中的 Part-ID 观测记录。
校验同时检查逐零件记录和汇总计数；任一视角被相机或空间配准淘汰时流程立即停止，
不会静默降级成两视角后继续选择材质。

| 组件 | 作用 |
| --- | --- |
| SAM3 | 提供整机前景和逐零件迭代初值。 |
| EntitySeg / CropFormer | 提供不依赖零件类别的优化起点；必须先通过 CAD 范围检查。 |
| 邻件关系定位器 | 排除目标自己的旧掩码，用多个其他零件和 CAD 装配关系推断目标位置。 |
| 迭代边界优化器 | 联合完整模型图零件形状、邻件禁入区、两遍分割候选和前序掩码；普通件使用照片边缘，自动识别的细长件还使用中心线亮脊和双侧边界。只变换二维候选掩码，绝不单独移动 mesh。 |
| MVInverse | 估计区域内的基础色、粗糙度和金属度。 |
| SigLIP2 | 从 NVIDIA `Materials/Base` 检索外观相近的 MDL。 |
| DINOv2 | 比较局部表面和纹理外观。 |
| Qwen3.5 | 根据已有分析结果排序有限的候选列表。 |
| MDL 渲染比较 | 在已对齐的 CAD 视角中比较真实 MDL 效果。 |

模型只负责对材质库候选排序，不会直接修改 USD 绑定。

材质身份与颜色分成两步。确认与照片颜色一致的精确库预设会直接保留；否则，每个 Part-ID
单独执行一次不含颜色的对应材质判断，候选中必须预留同物理类别、且程序明确支持调色的
MDL。组件只有在成员投票达到严格多数时才按票数统一；平票或低共识必须从所有成员独立排名
的共同候选中重新决策，没有共同候选就停止。选定材质身份后，才允许修改程序已经检查过的
颜色参数，并用实际 CAD 重渲染逐组件、逐成员检查局部外观，避免小零件被整图平均分掩盖。如果全部
颜色候选仍低于局部质量底线，流程会应用实测最优候选，并把受影响的零件或组件记录为 `REVIEW`。
这是对每个 Part-ID 和组件一致生效的通用回退，不包含工件特例。
不同颜色的零件选到同一个基础材质时不会停止流程。只有各颜色组都具备可实际写入的多视角参数
证据，而且材质类别与表面处理互相兼容，才会保留共同材质并分别校色；否则改用安全结果，记录
`REVIEW` 后继续。金属、塑料、橡胶等类别冲突即使颜色相同也按同样规则处理。

## 赋值和验证规则

- 每个 Mesh 必须得到一个可应用的视觉材质；
- 可见零件只使用该 Part-ID 自己通过检查的照片信息；
- 不可见或无法判断的零件使用预设默认材质，不把推测结果当作照片观测；
- 出现重复、不完整或仍需复核的赋值时，流程会在应用材质前停止；
- 记录胜出的 MDL 后不再更换材质身份；只有标记为“对应材质”的赋值可修改程序明确支持的颜色
  参数，精确库预设、粗糙度、金属度、纹理和法线保持不变；
- 局部颜色检查未通过时，不丢弃整个结果或中断阶段；保留实际 CAD 渲染中表现最好的候选，
  并把对应零件或组件明确记录为 `REVIEW`。格式损坏、身份变化、哈希错误和覆盖不完整
  仍会严格失败；
- 赋材质后的 USD 会与参考图比较；收集依赖后，最终 USD 还会重新打开并再次检查。任何
  一项失败都会停止流程，并指出失败阶段。

默认配置为
`tools/qwen_material_pipeline/src/qwen_material_pipeline/configs/pipeline/manual_part_id_materials.json`，只搜索 NVIDIA
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
  -r tools/qwen_material_pipeline/requirements/entityseg.txt
PYTHONNOUSERSITE=1 "$ENTITYSEG_PYTHON" -c \
  'import black, cloudpickle, mmcv, yapf'
```

YAPF 的固定版本用于兼容 MMCV 1.x。仅能从 `~/.local` 导入不算有效安装，
因为主线子进程会设置 `PYTHONNOUSERSITE=1` 以保证运行可复现。

| 模式 | 用途 |
| --- | --- |
| `live`（实时推理） | 新工件的正式主线；必须提供通过完整校验的人工前景标注。 |
| `bundled`（固定复现） | 必须精确匹配已有的固定复现项目，否则停止。 |

专用命令 `manual-material-pipeline` 只开放这两种语义明确的模式。底层 Python API 为兼容
固定复现项目的自动发现仍保留 `auto`，但它不是新工件的正式生产命令。

`--resume` 只用于 `live` 的同一份输入；可复用的视觉阶段会校验当前输入、配置、模型版本和
输出哈希。要从零运行，请使用新的输出目录。

CAD 转换开始前，专用命令会校验标注文档哈希、每张图片的文件哈希和解码像素、掩码文件
哈希和解码二值像素、图片尺寸、点选事件顺序以及全视角确认状态。人工前景掩码用于 `live`；
`bundled` 精确匹配后使用该复现项目中已确认、后续不再修改的分析数据。

## 排错

先找到第一个失败的 `[PROGRESS]` 阶段，再检查对应输出：

| 现象 | 检查内容 |
| --- | --- |
| 前景不完整 | SAM3 标注和掩码。 |
| CAD 与照片不重合 | 相机对齐叠加图。 |
| 可见零件使用了默认材质 | `part_id_reference_evidence.json`。 |
| Qwen 输出被拒绝 | `.raw`、`.parse.json` 和数据格式校验错误。 |
| 预期 MDL 没有胜出 | 候选检索结果和 MDL 渲染分数。 |
| 最终 USD 外观变化 | 已记录的材质选择和最终渲染对比。 |
| `CUDA out of memory` | 停止其他 GPU 任务后继续同一任务；SAM3、MVInverse、Qwen 和 EntitySeg 已按独立顺序进程运行。 |
| `cumsum_cuda_kernel ... deterministic implementation` | 这是 EntitySeg 的 `warn_only` 可复现性警告，不代表推理失败；真正错误要看后续 `[PROGRESS]` 或 traceback。 |
| `No space left on device` | 清理空间或把运行目录放到更大的文件系统，再使用 `--resume`；渲染图、恢复归档和模型缓存会远大于最终 USD。 |

主要输出位于
`RUN_ROOT/visual_material/{renders,analysis,visual_quality,final_visual_acceptance}`。

进一步说明：

- [材质子包](../../tools/qwen_material_pipeline/README.zh.md)
- [MVInverse](../../tools/qwen_material_pipeline/docs/mvinverse.zh.md)
- [Base 材质观察库](../../tools/qwen_material_pipeline/docs/base_material_observation_bank.zh.md)

MVInverse 仅限非商业用途；NVIDIA 资产和 Isaac Sim 采用各自许可，见
[第三方声明](../../legal/THIRD_PARTY_NOTICES.md)。
