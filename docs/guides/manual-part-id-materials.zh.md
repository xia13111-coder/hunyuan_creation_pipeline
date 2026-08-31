# STEP/STP 自动赋材质

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

流程使用同一工件的 2–4 张照片，为每个 CAD Part-ID 选择 NVIDIA Base MDL。照片只提供
外观信息，不修改 CAD 几何、尺寸或装配关系。

## 运行

先在页面中点选并确认每张照片里的整机前景：

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

保存后运行：

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

程序会自动完成相机对齐、零件分割、材质选择、必要的颜色调整、USD 绑定和最终验证。
人工只需确认整机前景，不需要逐个零件标注。

## 结果

```text
RUN_ROOT/pipeline_result.json
RUN_ROOT/visual_material/
RUN_ROOT/final/
```

中断后对同一输出目录添加 `--resume`。失败时先查看第一条 `FAILED`；显存或磁盘不足时，
释放资源后再继续。MVInverse 仅限非商业用途，其他许可见
[第三方声明](../../legal/THIRD_PARTY_NOTICES.md)。
