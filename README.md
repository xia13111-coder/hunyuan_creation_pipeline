# Hunyuan Creation Pipeline

这是一个资产生成和后处理流水线，用于把 Hunyuan 生成的 GLB 或已有的 GLB 模型转换为带物理属性的 USD 资产。

## 功能

- 调用 Tencent Hunyuan 3D 生成 GLB。
- 对 Hunyuan 生成的 GLB 执行 refine mesh。
- 将 GLB 转为 Z-up USD。
- 使用 Isaac Sim 添加物理材质、碰撞体、刚体和质量。
- 收集最终 USD 资产到统一输出目录。
- 支持手工建模或第三方 GLB 直接走后半段流水线。

## 流水线模式

### Hunyuan 生成模式

```text
图片 / prompt
-> Hunyuan 生成 GLB
-> refine mesh
-> 轴向对齐和尺寸调整
-> GLB 转 USD
-> 添加物理属性
-> 收集最终 USD
```

### 已有 Hunyuan GLB

```text
已有 GLB
-> refine mesh
-> 轴向对齐和尺寸调整
-> GLB 转 USD
-> 添加物理属性
-> 收集最终 USD
```

### 手工建模 GLB

```text
已有 GLB
-> GLB 转 USD
-> 添加物理属性
-> 收集最终 USD
```

手工建模模式不会做 Hunyuan refine mesh，也不会旋转或烘焙模型方向；它只会写入 USD `upAxis = "Z"` 元数据。

## 环境准备

安装依赖：

```bash
conda env create -f ./environment.yml
conda activate hunyuan
```

如果环境已经存在：

```bash
conda env update -f ./environment.yml --prune
```

Hunyuan 生成或 refine mesh 需要设置 Tencent Cloud 凭证：

```bash
export TENCENTCLOUD_SECRET_ID="your-secret-id"
export TENCENTCLOUD_SECRET_KEY="your-secret-key"
```

Blender 和 Isaac Sim 会自动检测常见安装位置。自动检测不到时，可以按本机环境设置：

```bash
export BLENDER_BIN="blender"
export ISAAC_PYTHON="./isaac-sim/python.sh"
```

## 运行命令

### 1. Hunyuan 完整流水线

把输入图片放到 `./data/`，然后执行：

```bash
python ./run_asset_pipeline.py \
  --input-dir ./data \
  --output-dir ./downloads \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --result-json ./pipeline_result.json \
  --face-count 150000 \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --orientation "X=L,Y=M,Z=S" \
  --material plastic \
  --approx sdf
```

### 2. 已有 Hunyuan GLB

```bash
python ./run_asset_pipeline.py \
  --existing-glb ./downloads/example_asset \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --result-json ./pipeline_result.json \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --material plastic \
  --approx sdf
```

### 3. 手工建模 GLB

```bash
python ./run_asset_pipeline.py \
  --manual-glb ./input/manual_asset.glb \
  --intermediate-output-dir ./manual_output_intermediate \
  --final-output-dir ./manual_output_final \
  --result-json ./manual_pipeline_result.json \
  --material steel \
  --approx sdf \
  --set-mass 30
```

如果手工模型也需要使用流水线的轴向映射和尺寸调整，可以增加：

```bash
  --manual-align \
  --manual-resize \
  --len-x 0.6 \
  --len-y 0.4 \
  --len-z 0.5
```

## 关键参数

| 参数 | 说明 |
| --- | --- |
| `--input-dir` | Hunyuan 生成模式的输入图片目录。 |
| `--prompt` | Hunyuan 文生 3D 的文本提示词。 |
| `--image-url` | Hunyuan 图生 3D 的图片 URL。 |
| `--existing-glb` | 已有 Hunyuan GLB，仍会执行 refine mesh。 |
| `--manual-glb` | 手工建模或第三方 GLB，跳过 Hunyuan 和 refine mesh。 |
| `--len-x`, `--len-y`, `--len-z` | 目标尺寸，单位为米。 |
| `--orientation` | `align_glb_axis_only.py` 使用的轴向映射。 |
| `--material` | `materials.json` 中的材质名称。 |
| `--approx` | 碰撞近似方式，例如 `sdf`、`convexHull`、`triangleMesh`。 |
| `--set-mass` | 整个资产的总质量，单位为 kg。 |
| `--skip-refine` | 对已有 GLB 跳过 refine mesh。 |
| `--usd-format` | USD 输出格式，例如 `usd` 或 `usda`。 |

查看完整参数：

```bash
python ./run_asset_pipeline.py --help
```

## 质量和坐标约定

- 输出 USD 会设置为 `upAxis = "Z"`。
- 手工建模 GLB 默认保留原始几何方向。
- `--set-mass` 表示整个资产的总质量，不是某一个 mesh 的质量。
- 如果资产中有多个刚体，总质量会按体积分配。
- 如果没有设置 `--set-mass`，质量会根据材质密度和模型体积估算。

## 输出目录

```text
./downloads/                  Hunyuan 原始生成结果
./downloads_refined_mesh/      refine mesh 结果和中间 GLB
./output_intermediate/         添加物理属性后的 USD
./output_final/                最终收集后的 USD 资产
./pipeline_result.json         流水线结果摘要
```

生成结果、缓存、日志和本地环境文件都不会提交到 git。

## 项目结构

```text
./run_asset_pipeline.py          主入口
./pipeline_runner.py             流水线编排
./hunyuan_to3d_batch.py          Hunyuan 生成客户端
./asset_refiner/                 refine mesh 模块
./align_glb_axis_only.py         GLB 轴向映射
./resize_glb_xyz_and_center.py   GLB 尺寸调整和居中
./convert_glb_to_usd_zup.py      GLB 转 USD
./add_physics.py                 Isaac Sim 物理属性写入
./collect_usd_flat.py            最终 USD 收集
./configs/                       refine mesh 配置
./materials.json                 物理材质配置
```

Docker 和 HTTP API 的使用方式见 `./README.docker.md`。
