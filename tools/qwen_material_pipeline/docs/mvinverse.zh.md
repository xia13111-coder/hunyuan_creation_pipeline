# MVInverse 集成说明

MVInverse 从参考图预测 albedo、metallic、roughness、normal 和 shading。在本项目中，它只
提供材质检索依据：不识别 CAD Part-ID、不直接贴图，也不修改选中的 NVIDIA MDL。

## 1. 本地文件

```text
tools/qwen_material_pipeline/
├── src/qwen_material_pipeline/mvinverse/
│   ├── adapter.py       # 输入准备、离线调度、账本和输出校验
│   ├── runner.py        # 加载上游模型
│   ├── evidence.py      # 区域统计与多视图融合
│   └── autonomy.py      # 分析质量判断
├── third_party/mvinverse/
│   ├── REVISION
│   ├── LICENSE
│   └── mvinverse/models/mvinverse.py
└── runtime/models/mvinverse/model/
    ├── config.json
    └── model.safetensors
```

`third_party/mvinverse/REVISION` 固定上游源码版本。当前期望 revision 为：

```text
6172ff9a437444df028ed67523badfa523173f21
```

权重由 `.gitignore` 排除，不应进入源码或最终 USD 交付包。换机器时通过环境变量或只读
挂载提供模型，不要把开发机绝对路径写入版本化配置。

## 2. 许可

上游许可证仅允许 **non-commercial purposes**。因此：

- 运行时必须显式传入 `--acknowledge-mvinverse-noncommercial`；
- 商业使用、对外服务或再分发前必须另行取得授权；
- 上游源码、模型权重和模型卡可能有不同条款，应采用更严格者；
- `third_party/mvinverse/` 和 `runtime/models/mvinverse/` 不应直接打入客户交付包。

以上是工程使用提示，不构成法律意见。

## 3. 环境

MVInverse 与主流程共用 `hunyuan_sam3d` Conda 环境，但在独立子进程中加载 GPU 模型：

```bash
conda activate hunyuan_sam3d
```

依赖由根目录的 `requirements/visual-materials.txt` 管理。不要另装上游推荐的 CUDA 11.8
PyTorch，也不要用 Isaac Sim Python 运行 MVInverse。

## 4. 输入要求

适配器要求：

- 清单中包含有序、非空且 ID 唯一的 `source_views`；
- 所有 PNG/JPEG 都来自同一工件，并提供互补视角；
- `--repo` 指向本项目中的上游源码，并包含 LICENSE 和预期模块；
- `--checkpoint` 指向完整的本地模型目录或受支持的权重文件；
- 不使用 Hub ID，也不在自动任务中自动联网下载。

默认最大边长为 448。检测到 CUDA OOM 时可按配置降至 392 重试。所有视图保持宽高比并
使用一致的预处理尺寸。

## 5. 独立运行

从仓库根目录执行：

```bash
PIPELINE_ROOT="$PWD/tools/qwen_material_pipeline"

python -m qwen_material_pipeline.mvinverse.adapter \
  --reference-manifest ./inputs/reference_manifest.json \
  --repo "$PIPELINE_ROOT/third_party/mvinverse" \
  --python "$(command -v python)" \
  --checkpoint "$PIPELINE_ROOT/runtime/models/mvinverse/model" \
  --output-dir ./outputs/my_run/visual_material/analysis/mvinverse \
  --device cuda \
  --max-side 448 \
  --oom-retry-max-side 392 \
  --timeout-seconds 1800 \
  --acknowledge-mvinverse-noncommercial
```

`--dry-run` 只检查路径、许可、源码、权重和预处理。`--reuse-existing` 仅在图片、清单、
源码 revision、权重、runner、设备和尺寸全部一致时复用结果。

## 6. 输出与完整性

```text
mvinverse/
├── inputs_0448/                 # 预处理图
├── inputs_0392/                 # OOM 备用尺寸（如启用）
├── maps/
│   ├── 000_albedo.png
│   ├── 000_metallic.png
│   ├── 000_roughness.png
│   ├── 000_normal.png
│   └── 000_shading.png
├── attempt_01.stdout.log
├── attempt_01.stderr.log
└── mvinverse_inference_ledger.json
```

账本记录原图、预处理图、源码、许可证、权重、runner 和输出的哈希。只有状态为 `SUCCESS`
或通过完整校验的 `REUSED`，且五类 map 的数量、尺寸和哈希正确时，结果才进入下一阶段。

## 7. 在材质选择中的作用

MVInverse 的区域统计会与以下信息共同参与 Base MDL 候选排序：

- Qwen 对颜色、表面外观和可见物质的描述；
- SigLIP2 的整体视觉相似度；
- DINOv2 的局部纹理相似度；
- NVIDIA MDL 的默认颜色、metallic 和 roughness；
- 候选绑定到真实 CAD 零件后的重渲染结果。

使用时需要注意：

- albedo 不是可直接写入 USD 的纹理；
- 高光会影响 metallic，照明会影响颜色和 roughness；
- normal 没有相机、UV 和 texel 投影时不能直接作为模型 normal map；
- MVInverse 不知道像素属于哪个 Part-ID，零件对应关系来自相机配准和 Part-ID 渲染；
- 单视图或小区域估计只能降低权重使用，不能视为稳定的多视角结果。

最终候选必须来自 `NVIDIA/Materials/Base`，并经过真实 CAD 重渲染比较。当前生产配置在
选择完成后锁定 MDL；MVInverse 不能再改颜色、metallic、roughness、纹理或 face subset。

## 8. 缓存边界

每个工件生成独立的输入清单、推理记录、PBR 图和材质计划。只有输入和全部指纹一致时
才能恢复；不能把一个工件的 MVInverse 结果复制给另一个工件。需要长期保留的运行应将
这些分析结果与收集后的 USD 和最终验证报告一起归档。
