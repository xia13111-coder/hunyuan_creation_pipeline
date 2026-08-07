# Hunyuan 原始生成模块

[English](./hunyuan.md) | [中文](./hunyuan.zh.md) | [文档索引](../README.zh.md)

与 SAM3D 的输入和适用场景区别见 [Hunyuan 与 SAM3D 生成方式选择](../generation-guide.zh.md)。

该模块把本地图片或图片 URL 提交给 Tencent Hunyuan 3D，下载原始 GLB，再交给
网格精修模块。它不负责 ReduceFace、UV、物理或 USD 转换。

## 代码

```text
asset_pipeline/jobs/hunyuan.py
  run_generate_model_job
  -> run_hunyuan_job
     -> python -m asset_pipeline.hunyuan_generation

asset_pipeline/hunyuan_generation.py
  Tencent SDK 请求、轮询和下载
```

## 输入方式

| CLI 参数 | 作用 |
| --- | --- |
| `--input-dir` | 扫描目录中的 JPG、JPEG、PNG、WEBP，逐张 image-to-3D。 |
| `--image-url` | 使用可公网访问的单张图片 URL。 |
| `--face-count` | Hunyuan Pro 请求面数，例如 `150000`。 |
| `--download-preview` | 下载 API 返回的 preview 图片。 |
| `--output-dir` | 原始生成文件目录，默认 `./downloads`。 |

`--input-dir` 和 `--image-url` 是 Hunyuan 的两个输入分支，不能同时使用。该项目不开放
Hunyuan 纯文本生成；图片目录不存在或没有有效图片时，CLI 会在提交前报错。

## 命令

```bash
hunyuan-asset-pipeline \
  --input-dir ./data \
  --output-dir ./outputs/hunyuan_example/generation \
  --face-count 150000 \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --intermediate-output-dir ./outputs/hunyuan_example/intermediate \
  --final-output-dir ./outputs/hunyuan_example/final \
  --len-x 0.4 --len-y 0.3 --len-z 0.3 \
  --material plastic \
  --approx sdf
```

只调试生成客户端：

```bash
python -m asset_pipeline.hunyuan_generation --help
```

## 凭据

```bash
export TENCENTCLOUD_SECRET_ID="your-secret-id"
export TENCENTCLOUD_SECRET_KEY="your-secret-key"
```

不要把真实凭据写入 YAML、README 或提交到 git。

## 输出到下一模块

`run_generate_model_job` 返回原始模型列表。完整流程默认把 `output_dir` 交给
`jobs.refine.run_refine_mesh_job`；只有显式使用 `--skip-refine` 才直接把原始 GLB 交给
Blender 后处理。

```text
Hunyuan 原始 GLB
-> 网格精修
-> Blender 后处理
-> Isaac Sim 物理处理
```

## 常见问题

| 问题 | 处理 |
| --- | --- |
| 缺少 SecretId/SecretKey | 设置两个腾讯云环境变量，并确认当前 shell 可以读取。 |
| API 提交失败 | 检查区域、服务地址、账号权限和服务配额。 |
| 下载结果为空 | 查看生成客户端日志中的 JobId、状态和 ResultFile3Ds。 |
| 想处理已有 GLB | 使用 `--existing-glb`，不要重新调用生成模块。 |
