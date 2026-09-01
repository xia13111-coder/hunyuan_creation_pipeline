# MVInverse 集成

MVInverse 从参考图估计基础色、金属程度、粗糙度、法线和光照信息。在本项目中它只提供
材质判断信息，不识别 Part-ID、不直接贴图、不选择 MDL，也不修改 USD。

## 许可

MVInverse 上游许可证仅允许非商业用途。运行时必须确认
`--acknowledge-mvinverse-noncommercial`。商业使用、服务或再分发需要另行取得授权。

上游源码和许可证位于 `third_party/mvinverse/`；模型权重位于被 Git 忽略的
`runtime/models/mvinverse/`，不进入源码或 USD 交付包。

## 运行要求

MVInverse 在独立 GPU 子进程中运行，使用主流程配置的仓库、权重、设备和图像尺寸。参考图
必须来自同一工件，视角 ID 唯一。本地路径不完整或源码版本不匹配时会停止，不会在线下载。

CUDA 显存不足时可按配置降低输入尺寸重试。不要使用 Isaac Sim Python 运行 MVInverse。

独立检查和运行参数见：

```bash
qwen-material mvinverse-run --help
qwen-material mvinverse-evidence --help
```

## 输出和恢复

主要输出包括预处理图、五类预测图、运行日志和
`mvinverse_inference_ledger.json`。账本记录图片、源码、权重、配置和输出哈希。

只有状态、数量、尺寸和哈希全部有效的结果才会进入下一阶段。`--resume` 也只能复用当前
工件中完全匹配的结果，不能把一个工件的 MVInverse 输出复制给另一个工件。

## 在材质选择中的作用

MVInverse 与 CAD Part-ID 区域、Qwen、SigLIP2、DINOv2 和 MDL 实际渲染共同参与候选排序。
预测的 albedo 不是可直接写入 USD 的纹理，normal 也不能直接作为模型法线贴图。最终材质
必须来自允许的 NVIDIA Base 目录，并由真实 CAD 渲染验证。
