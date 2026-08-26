# 材质身份优先、真实 CAD 校色流程

本文档固定当前已经通过四视图验证的材质主线。它适用于任意具有稳定 Part-ID、参考照片和
NVIDIA Base MDL 目录的 CAD 资产，不包含针对某个资产或某个 Part-ID 的特殊规则。

本文中，Part-ID 是 CAD 中的零件编号，MDL 是 NVIDIA 材质文件。“材质身份”只表示选中了
哪一种 MDL，不包含颜色。掩码是图片中目标占据的像素区域，哈希或 SHA256 是用于判断文件
内容是否变化的摘要。命令名和输出文件名保留英文，便于与程序输出直接对应。

## 1. 决策顺序

```text
CAD Part-ID 与参考照片对应
  -> 检查四个视角 + 用同一相机生成目标零件完整形状（包括遮挡部分）
  -> 逐零件材质预测与目录检索
  -> 精确材质 / 对应材质分类
  -> 统一照片中判断为同材质的零件
  -> 固定所选 MDL
  -> 仅对“对应材质”生成初始颜色
  -> 真实 CAD 重渲染并测量每个零件或组件的变化
  -> 每个组件或独立零件自动连续优化颜色增益
  -> 从全部已测结果中为每个零件或组件选择最高分结果
  -> 最终四视图质量检查
```

材质身份和颜色是两个不同变量：

- 能确认目录中的精确材质时，保留该 MDL 的原生颜色预设；
- 只能确认材质类别时，先固定对应 MDL，再调整它允许的颜色参数；
- 颜色阶段不允许更换 MDL；
- 同一个照片材质组件中的全部 Part-ID 始终共享一个 MDL 和一套颜色参数。

组件内的“材质身份一致”和“原生颜色预设精确”是两项独立结论。少数组件成员确认某个精确
MDL 时，可以用它帮助同组零件确定材质类型，但不能把这些成员的原生颜色结论套用到整组。
只有全部组件成员都独立选择同一预设、通过颜色和可信度检查，整组才保留原生预设；否则整组
保持“对应材质”状态，并用全部已确认并记录的照片信息共享校色。这可以避免小零件或重复结构
使大型外壳错误采用默认颜色。

## 2. 逐零件判断依据

判断依据来自相机配准后的 CAD 投影、单独渲染的零件形状、SAM3/EntitySeg 精细掩码、MVInverse
物理外观、SigLIP2/DINOv2 检索和 Qwen 有限候选判断。正式配置关闭快速相机候选搜索，并要求
全部已注册参考视角都产生真实的已选 Part-ID 观测记录。单独渲染的零件模板使用相同的整件相机
和原始 CAD 变换，只补足被邻件遮挡的目标形状，不允许移动单个 Mesh。模型输出只提出候选，
不能直接写 USD。

逐零件边界同时保留目标在 CAD 整机图中相对其他零件的位置。第一遍分割找出用于定位的参照
零件；对每个
目标都排除它自己的旧掩码，再根据多个其他零件估计位置、旋转和尺度变化，并用最近邻件的
位置误差共同推断目标位置。随后用 CAD 模型图中的完整 Part-ID 形状生成新框，第二次运行
SAM3 与 EntitySeg。
两遍候选、上一版融合结果、照片边缘、完整模型形状和邻件禁入区共同评分。神经候选拒绝时仍以
邻件定位的 CAD 形状做边缘优化；该 CAD 起始形状只约束搜索范围，不再同时充当自身的 1.0 候选支持
下限。边缘增益与邻件净空联合决定是否接受局部平移、旋转和尺度修正；有真实图像候选时仍执行
严格的候选质量检查。定位依据不足时保留已经检查过的结果，不终止流程。整个过程只优化二维
零件掩码，不修改整件相机、CAD 变换或单个 Mesh 位姿。

`workflows/part_id_qwen.py` 对精确材质使用双重入口：

1. Qwen 给出高置信的物理材质身份；或
2. 目录预设本身物理兼容，并同时通过形状、实测颜色和 Qwen 置信度验证。

精确预设的实测颜色差使用 CIEDE2000 指标衡量，数值越小越接近；上限为 `15.0`。不满足时
降为“对应材质”，以免把颜色接近但类别错误的 MDL 当成精确匹配。

## 3. 同材质组件与单视图补全

常规组件必须由至少两个参考视图支持。只有在已有组件已经由多个视图确认后，单视图零件才可能
自动并入，且必须同时满足：

| 条件 | 当前下限/上限 |
| --- | --- |
| 可信照片像素 | `>= 4096` |
| 去除异常像素后的有效颜色比例 | `>= 0.90` |
| 与组件 CIEDE2000 | `<= 10.0` |
| 零件在空间上相邻的支持分 | `>= 0.25` |
| 同一装配分支内的包围框间隔 | `<= 1` |
| 前景覆盖 | `>= 0.80` |
| CAD 形状准确率和覆盖率的较小值 | `>= 0.80` |
| 误包含的其他 Part-ID 比例 | `<= 0.05` |

还要求候选具有已接受的形状引导分割、相同装配分支和相同表面类别。任何条件不满足都保持
独立 Part-ID，不进行组件传播。这条规则解决单视图大零件遗漏，同时阻止掩码泄漏把邻近零件
错误合并。

## 4. 真实 CAD 颜色选择

`materials/corresponding_material_color.py` 只为“对应材质”生成颜色参数；精确库材质保持原生
预设不动。对应材质还必须具有 `tuning.py` 中明确支持调节颜色的参数；没有这类参数的库材质仍
保留已选 MDL 及其原生预设，不写入未知参数，也不会阻断其他材质校色。入口筛选、计划生成
和最终复验使用同一项检查。默认控制器从 `1.0` 开始，不再使用人工设定的固定数值列表
`DEFAULT_GAINS`。每轮都必须经过：

1. 将完整 Part-ID 计划应用到真实 CAD USD；
2. 重新建立 Part-ID 零件索引；
3. 用最终确认、后续不再修改的相机参数和 `material-neutral` 灯光渲染全部参考视图；
4. 在每个独立零件/组件的可信照片像素上计算实际外观分数。

`materials/adaptive_corresponding_material_color.py` 从可信参考掩码中读取 Lab 色彩空间的亮度，
再用“参考亮度 / 实渲亮度”为每个零件或组件计算连续增益。第二轮起根据前几轮结果估计渲染
对参数变化的响应；可用信息不足时使用经过限制的亮度比例。单步最大变化为两倍，总范围为
`[0.1, 8.0]`。达到亮度误差、增益步长或外观提升的停止条件时自动结束。所有零件和组件在
同一完整 CAD 中共同重渲染，
不存在按 Part-ID 单独修图。

`materials/corresponding_material_color_selection.py` 随后为每个零件或组件，从全部已渲染
结果中选择外观分数最高的一项。这样即使某一轮自动提议变差，也不会覆盖此前更好的颜色。
选择时还会重新校验计划、每个零件或组件的增益、应用报告、USD 和渲染后的零件索引，并确认
其路径和哈希正确、所有 MDL 保持不变。

## 5. 主流程接入与独立复现入口

默认 `manual-material-pipeline` 配置已经直接执行本流程。主流程确认每个 Part-ID 的最终材质计划后，
调用真实 CAD 自适应校色，并把 `material_identity_color/workflow_manifest.json`、最终计划、
应用报告、渲染后的零件索引与四视图报告逐一按路径和 SHA256 复验；后续发布检查改用校色后的
同一 MDL 计划，最终记录同时固定所选 MDL 和已经检查过的颜色参数。旧的跨材质外观比较不会
在该分支运行。

以下独立复现命令把自动提议、真实 CAD 渲染、历史择优、最终重渲染和质量检查合并为一次
运行：

```bash
python -m qwen_material_pipeline \
  run-corresponding-material-color-workflow \
  --source-plan RUN/part_id_material_plan.identity.json \
  --qwen-choices RUN/part_id_qwen_choices.json \
  --part-id-evidence RUN/part_id_reference_evidence.json \
  --spatial-mapping-report RUN/spatial_mapping_report.json \
  --asset-usd RUN/source_asset.usda \
  --catalog RUN/nvidia_mdl_catalog.json \
  --registry RUN/part_registry.camera_calibrated.json \
  --material-root /path/to/NVIDIA/Materials/Base \
  --view-specs RUN/camera_view_specs.json \
  --reference-manifest RUN/reference_manifest.json \
  --isaac-python /path/to/isaac-sim/python.sh \
  --output-dir RUN/material_identity_color_final \
  --require-pass
```

默认最多运行 5 轮，可用 `--max-adaptive-iterations` 修改上限。只有为了旧结果复现时，才重复
传入至少两个 `--gain` 切换到固定网格兼容模式。输出目录必须不存在，避免旧候选或旧渲染被
误用。命令会生成：

```text
workflow_manifest.json
candidates/iteration_*/          # 自动模式
# 或 candidates/gain_*/         # 显式固定网格兼容模式
  part_id_material_plan.color.json
  corresponding_material_color_audit.json
  material_look.usda
  apply_report.json
  part_registry.json
  renders/part_registry.rendered.json
final_selected/
  part_id_material_plan.color.selected.json
  corresponding_material_color_selection_audit.json
  material_look.usda
  apply_report.json
  part_registry.json
  renders/part_registry.rendered.json
  reference_render_comparison.json
```

`workflow_manifest.json` 保存全部输入路径和哈希、候选、选择规则、最终计划、USD、渲染后的
零件索引和质量报告。若已记录哈希的参考图只发生同目录改名，流程只允许按原始 SHA256 找回
唯一文件，禁止按视角名猜测；实际使用的路径也会写回清单。失败时保留
`workflow_state=FAILED` 及日志；不能在同一目录自动续跑。

## 6. 验收和边界

- 参考图清单中的每个视图都必须与同名实际渲染一一对应；
- 使用 `--require-pass` 时，最终质量检查不是 `PASS` 就返回状态码 `3`；
- 调色不修改几何、拓扑、姿态、物理属性、Part-ID 或 MDL 身份；
- 客户图片、模型权重、绝对运行目录和最终大体积渲染不提交到 Git；
- 代码、配置、测试和复现规则进入 Git，具体运行由清单和内容哈希复现。
