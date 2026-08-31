# NVIDIA Base 材质观察库

主线已在 `assets/nvidia_base_observation_bank_v1/` 自带完整观察库。它保存 NVIDIA
`Materials/Base` 的标准渲染和视觉向量，不包含具体工件信息，也不修改 MDL。

## 内容

每个版本包含：

- Base MDL 目录和允许列表；
- 标准材质渲染、前景掩码和外观特征；
- SigLIP2、DINOv2 视觉向量；
- 模型、文件和输出哈希。

## 重新构建

只有 NVIDIA Base 材质内容或检索模型发生变化时才需要重新构建：

```bash
qwen-material base-bank build \
  --material-root "$VISUAL_MATERIAL_ROOT/Base" \
  --output-dir "$NVIDIA_BASE_OBSERVATION_BANK" \
  --isaac-python "$ISAAC_PYTHON" \
  --siglip2-model "$SIGLIP2_MODEL_PATH" \
  --dinov2-model "$DINOV2_MODEL_PATH"

qwen-material base-bank verify \
  --material-root "$VISUAL_MATERIAL_ROOT/Base" \
  --output-dir "$NVIDIA_BASE_OBSERVATION_BANK"
```

构建中断后可重复执行同一命令；已有结果会先校验再复用。校验要求目录范围、材质覆盖、
标准渲染、视觉向量和所有哈希一致。

## 在主流程中的作用

生产配置通过 `retrieval.observation_bank` 读取观察库，检索得到的候选仍需绑定到真实 CAD
并重新渲染比较。最终材质只能来自当前允许的 NVIDIA Base 目录。
