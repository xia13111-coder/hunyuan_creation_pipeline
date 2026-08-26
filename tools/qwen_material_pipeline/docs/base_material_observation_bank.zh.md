# NVIDIA Base 材质观察库

观察库为 `NVIDIA/Materials/Base` 中的材质预先生成统一渲染和视觉向量，供运行时离线
检索。它不读取 `vMaterials_2`，也不修改 MDL。

## 生成内容

每次构建会产生：

- `catalog.json`：从 Base 实时扫描的全部 MDL export；
- `allowlist.json`：与 catalog 完全一致的候选集合；
- `scope_report.json`：Base 根目录、MDL 源文件指纹和越界计数；
- `renders/`：每种材质在固定几何上的三张标准渲染图；
- `masks/`：对应的几何前景掩码；
- `render_manifest.json`：每张图的 SHA-256、渲染进度和完整覆盖状态；
- `appearance_profiles.json`：颜色、亮度、高光比例、纹理梯度及 MDL 原始 PBR 默认值；
- `visual_embeddings.npz`：SigLIP2 768 维和 DINOv2 1024 维材质向量；
- `index_manifest.json`：模型身份、聚合方式和索引文件哈希。

三个标准视图为 `neutral_iso`、`grazing_front` 和 `top_soft`，使用固定几何、相机和光照。

`Chain_Link_Fence` 在部分 Isaac Sim 版本的批量渲染场景中不稳定，因此使用同一 NVIDIA
MDL 附带的官方预览图。清单会将其明确标记为
`nvidia_official_preview_opacity_safe`；该图片不计作标准渲染，也不会改用不透明材质。

## 完整构建

在项目根目录、`hunyuan_sam3d` 环境中运行：

```bash
python -m qwen_material_pipeline base-bank build \
  --material-root "$NVIDIA_MDL_BASE_ROOT" \
  --output-dir "$NVIDIA_BASE_OBSERVATION_BANK" \
  --isaac-python "$ISAAC_PYTHON" \
  --siglip2-model "$SIGLIP2_MODEL_PATH" \
  --dinov2-model "$DINOV2_MODEL_PATH"
```

命令先用 `ISAAC_PYTHON` 分批渲染，再用当前 Python 生成 SigLIP2/DINOv2 索引。变量来自
本机 `.env` 或 shell。默认每 40 种材质重启 Isaac；中断后重复命令可校验已有结果并续跑。

## 独立校验

```bash
python -m qwen_material_pipeline base-bank verify \
  --material-root "$NVIDIA_MDL_BASE_ROOT" \
  --output-dir "$NVIDIA_BASE_OBSERVATION_BANK"
```

只有以下条件同时满足才会通过：

1. catalog 根目录名称为 `Base`；
2. catalog 与 allowlist 完全覆盖当前 Base exports；
3. 任何路径和 material ID 中都没有 `vMaterials_2`；
4. 160 种材质的三张观察图及掩码均通过 SHA-256；
5. SigLIP2、DINOv2、外观特征和索引清单的哈希一致。

## 与单个 STP 推理的关系

观察库是一次性离线资产，不包含具体工件的人工选择。生产配置通过
`retrieval.observation_bank` 使用它。SAM3、MVInverse、SigLIP2、DINOv2 和视觉模型共同
生成候选，再把排名靠前的候选绑定到真实 CAD 零件并重新渲染比较。最终材质只能来自
Base；如果配置了 `immutable_after_selection=true`，选定后会原样绑定。
