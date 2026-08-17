# 材质身份优先、真实 CAD 校色流程

本文档固定当前已经通过四视图验证的材质主线。它适用于任意具有稳定 Part-ID、参考照片和
NVIDIA Base MDL 目录的 CAD 资产，不包含针对某个资产或某个 Part-ID 的特殊规则。

## 1. 决策顺序

```text
CAD Part-ID 与参考照片对应
  -> 逐零件材质预测与目录检索
  -> 精确材质 / 对应材质分类
  -> 同照片材质组件收敛
  -> 固定 MDL 身份
  -> 仅对“对应材质”生成初始颜色
  -> 真实 CAD 重渲染并测量每个作用域的响应
  -> 每个组件或独立零件自动连续优化颜色增益
  -> 从全部已测历史中逐作用域选择最高分结果
  -> 最终四视图绝对 QA
```

材质身份和颜色是两个不同变量：

- 能确认目录中的精确材质时，保留该 MDL 的原生颜色预设；
- 只能确认材质类别时，先固定对应 MDL，再调整它允许的颜色参数；
- 颜色阶段不允许更换 MDL；
- 同一个照片材质组件中的全部 Part-ID 始终共享一个 MDL 和一套颜色参数。

## 2. 逐零件证据

零件证据来自相机配准后的 CAD 投影、隔离 mesh 形状、SAM3/EntitySeg 精细掩码、MVInverse
物理外观、SigLIP2/DINOv2 检索和 Qwen 有限候选判断。模型输出只提出候选，不能直接写 USD。

`workflows/part_id_qwen.py` 对精确材质使用双重入口：

1. Qwen 给出高置信的物理材质身份；或
2. 目录预设本身物理兼容，并同时通过形状、实测颜色和 Qwen 置信度验证。

精确预设的实测颜色上限为 CIEDE2000 `15.0`。不满足时降为“对应材质”，以免把颜色接近但
类别错误的 MDL 当成精确匹配。

## 3. 同材质组件与单视图补全

常规组件必须由至少两个参考视图支持。只有在已有组件已经多视图封存后，单视图零件才可能
自动并入，且必须同时满足：

| 条件 | 当前下限/上限 |
| --- | --- |
| 可信照片像素 | `>= 4096` |
| 稳健颜色内点率 | `>= 0.90` |
| 与组件 CIEDE2000 | `<= 10.0` |
| 空间邻接支持 | `>= 0.25` |
| 同分支 bbox 间隔 | `<= 1` |
| 前景覆盖 | `>= 0.80` |
| CAD 形状 precision/recall 最小值 | `>= 0.80` |
| 其他 Part-ID 串漏 | `<= 0.05` |

还要求候选具有已接受的形状引导分割、相同装配分支和相同表面类别。任何条件不满足都保持
独立 Part-ID，不进行组件传播。这条规则解决单视图大零件遗漏，同时阻止掩码泄漏把邻近零件
错误合并。

## 4. 真实 CAD 颜色选择

`materials/corresponding_material_color.py` 只为“对应材质”生成颜色参数；精确库材质保持原生
预设不动。默认控制器从 `1.0` 开始，不再使用人工设定的 `DEFAULT_GAINS` 网格。每轮都必须经过：

1. 将完整 Part-ID 计划应用到真实 CAD USD；
2. 重新建立 Part-ID registry；
3. 用封存相机和 `material-neutral` 灯光渲染全部参考视图；
4. 在每个独立零件/组件的可信照片像素上计算实际外观分数。

`materials/adaptive_corresponding_material_color.py` 把可信参考掩码中的 Lab 亮度转换为相对亮度，
用“参考亮度 / 实渲亮度”产生每个作用域各自的连续增益。第二轮起优先使用有界 log-secant
响应估计；证据不足时使用阻尼亮度比。单步最大变化为两倍，总范围为 `[0.1, 8.0]`，达到亮度
误差、增益步长或外观提升收敛条件时自动停止。所有作用域在同一完整 CAD 中共同重渲染，
不存在按 Part-ID 单独修图。

`materials/corresponding_material_color_selection.py` 随后按作用域从全部已实渲历史中选择外观
分数最高的结果。这样即使某一轮自动提议变差，也不会覆盖此前更好的颜色。选择时还会重新
校验计划、每作用域增益、apply report、USD 和 rendered registry 的路径与哈希，并证明所有
MDL 身份保持不变。

## 5. 统一运行入口

以下命令把自动提议、真实 CAD 渲染、历史择优、最终重渲染和绝对 QA 固化为一次运行：

```bash
PYTHONPATH=tools python -m qwen_material_pipeline \
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

`workflow_manifest.json` 保存全部输入路径/哈希、候选、策略、最终计划、USD、渲染 registry 和
QA 报告。失败时保留 `workflow_state=FAILED` 及日志；不能在同一目录隐式续跑。

## 6. 验收和边界

- 参考 manifest 中的每个视图都必须与同名实际渲染一一映射；
- `--require-pass` 下，最终绝对质量不是 `PASS` 时返回状态码 `3`；
- 调色不修改几何、拓扑、姿态、物理属性、Part-ID 或 MDL 身份；
- 客户图片、模型权重、绝对运行目录和最终大体积渲染不提交到 Git；
- 代码、配置、测试和这份合同进入 Git，具体运行由清单和内容哈希复现。
