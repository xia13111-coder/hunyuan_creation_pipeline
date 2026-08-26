# 外观一致的零件组

`appearance-components` 位于相机配准之后、材质检索之前。它把照片中明显属于同一连续
表面的多个 CAD Part-ID 关联起来，避免同一外壳被选成不同颜色或不同材质。

## 输入与输出

输入必须来自同一次运行：

- 完成相机配准的 `part_registry.rendered.json`；
- 人工确认的 SAM3 前景标注；
- `camera_calibration_report.json`。

输出为 `analysis/appearance_components.json`。每个 Part-ID 只属于以下一种状态：

- `component_member`：有充分照片信息，可与其他零件共享一个 MDL；
- `observed_independent`：可见，但没有可靠的同外观关系；
- `unobserved_independent`：在参考图中不可可靠观察。

未观察到或边界不清的零件不会被强行分组。

## 分组规则

该阶段使用整机相机配准结果，不单独缩放、旋转或拉伸零件，也不会改变 CAD 的尺寸、
姿态、拓扑或 Part-ID。

两个零件只有在合格视图中同时可见、颜色接近、位置相邻且没有冲突视图时，才会进入
同一零件组。可信度较低的视图只作辅助参考。被拒绝的关系记录在
`rejected_conflicting_links` 中。

这个阶段不调用 Qwen、MVInverse、SigLIP 或 DINO，也不选择或写入 MDL。它只产生供后续
检索使用的成员约束。

## 后续选材

对每个零件组，流水线会：

1. 汇总成员的 SigLIP2、DINOv2 和 MVInverse 分析结果；
2. 用代表性 Part-ID 裁剪图与 Base MDL 渲染图生成候选；
3. 让 Qwen 在候选中选出一个 MDL；
4. 将同一个、未修改参数的 MDL 分别绑定到各成员。

独立零件仍按 Part-ID 单独选择。主要检查记录包括：

- `appearance_component_evidence.json`；
- `appearance_component_visual_retrieval/visual_retrieval.json`；
- `appearance_component_qwen_choices.json`；
- `appearance_component_mdl_selection.json`。

这些文件与当前零件索引和相机结果的哈希绑定，不能用于其他工件或运行结果。

## 独立运行

通常不需要单独运行。复核已有相机配准结果时可执行：

```bash
python -m qwen_material_pipeline appearance-components \
  --rendered-registry ./run/visual_material/highres_final/part_registry.rendered.json \
  --reference-manifest ./sam3_foreground_annotations.json \
  --camera-report ./run/visual_material/search_pass/camera_calibration_report.json \
  --output ./run/visual_material/analysis/appearance_components.json
```

任一输入缺失、哈希不匹配或相机结果不合格时，命令会停止并且不输出零件组结果。
