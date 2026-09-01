# 材质身份与真实 CAD 校色

该流程先确定每个 Part-ID 使用哪一个 MDL，再对允许调色的对应材质调整颜色。规则适用于
任意具有稳定 Part-ID 和参考照片的 CAD，不包含工件或零件特例。

## 决策顺序

```text
Part-ID 照片信息
-> 材质库检索和真实 MDL 比较
-> 固定材质身份
-> 仅为可调的对应材质生成颜色候选
-> 在完整 CAD 上重渲染
-> 选择实测最优颜色
-> 最终多视图验证
```

- 精确库预设保留原生参数；
- 对应材质只能修改程序明确支持的颜色接口；
- 校色不能更换 MDL；
- 照片中判断为同一材质的一组零件共享材质身份；
- 物理类别或表面处理冲突时不能仅因颜色接近而共用材质。

## 判断信息

材质判断综合 CAD Part-ID 形状和位置、SAM3/EntitySeg 融合区域、MVInverse PBR 信息、
SigLIP2/DINOv2 检索、Qwen 候选排序和真实 MDL 渲染。模型只产生候选，经过校验的代码才会
写入 USD。

同材质零件组必须有一致的多视图证据。证据不足或冲突时保持独立，不依赖 Part-ID 名单或
人工材质映射。

## 自动校色

校色从真实参考区域和 CAD 渲染中估计下一轮颜色参数，并限制单次变化和总范围。每轮都渲染
完整 CAD；选择器从所有已测候选中保留每个零件或组件的最高分结果。

没有经过审核的颜色接口时，材质保持原生预设。局部质量未达到目标时，流程仍保存实测最优
结果并标记 `REVIEW`；材质身份变化、哈希错误或无效 USD 仍会失败。

## 主流程与结果

`manual-material-pipeline` 已自动运行该流程。主要结果位于：

```text
visual_material/material_identity_color/workflow_manifest.json
visual_material/material_identity_color/final_selected/
visual_material/analysis/material_selection_lock.json
```

独立复现入口仅用于开发和检查：

```bash
qwen-material run-corresponding-material-color-workflow --help
```

调色不会修改几何、拓扑、姿态、物理属性或 Part-ID。
