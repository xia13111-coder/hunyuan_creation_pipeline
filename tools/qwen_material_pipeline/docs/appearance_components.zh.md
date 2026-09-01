# 外观一致的零件组

该阶段把照片中属于同一连续表面的多个 CAD Part-ID 组成一组，避免同一外壳被选成不同
材质或颜色。它位于相机配准之后、材质检索之前。

## 判断规则

只有在合格视图中同时可见、外观接近、空间相邻且没有冲突视图的零件才会合并。未观察到、
边界不清或证据冲突的零件保持独立。

分组使用统一的整机相机，不移动零件，也不修改 CAD。该阶段只生成关系约束，不调用材质
模型、不选择 MDL，也不写 USD。

## 结果

输出为 `analysis/appearance_components.json`。零件状态包括：

- `component_member`：属于可信零件组；
- `observed_independent`：可见但保持独立；
- `unobserved_independent`：无法可靠观察。

后续流程综合组内成员的信息选择一个材质身份；独立零件仍按 Part-ID 判断。所有结果与当前
零件索引和相机哈希绑定，不能复制到其他工件。

该阶段通常由主流程自动执行。独立命令和参数见：

```bash
qwen-material appearance-components --help
```
