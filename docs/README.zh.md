# 文档索引

[English](./README.md) | [中文](./README.zh.md) | [返回项目 README](../README.zh.md)

文档按代码模块拆分。主 README 只保留入口和快速命令，参数和实现细节在对应模块中维护。

| 模块 | 代码所有者 | 文档 |
| --- | --- | --- |
| 生成方式选择 | Hunyuan 与 SAM3D 输入路径 | [Hunyuan 与 SAM3D 对比](./generation-guide.zh.md) |
| 总体架构 | `asset_pipeline/` | [架构和调用关系](./architecture.zh.md) |
| Hunyuan 原始生成 | `asset_pipeline/hunyuan_generation.py`、`jobs/hunyuan.py` | [Hunyuan](./modules/hunyuan.zh.md) |
| SAM3D 图片重建 | `jobs/sam3d.py`、`tools/sam3d/` | [SAM3D](./modules/sam3d.zh.md) |
| Mesh refine | `jobs/refine.py`、`asset_refiner/` | [Refine](./modules/refine.zh.md) |
| Blender 后处理 | `jobs/blender.py`、`tools/blender/` | [Blender](./modules/blender.zh.md) |
| Isaac Sim 物理 | `jobs/isaac.py`、`tools/isaac/` | [Physics](./modules/physics.zh.md) |
| 手工 CAD | `workflows.run_stp_physics_job`、`convert_cad_to_usd.py` | [CAD](./modules/cad.zh.md) |
| HTTP API / Docker | `asset_pipeline/api.py`、`docker/` | [API](./modules/api.zh.md) |

其他说明：

- [Refine 配置文件说明](../configs/README.md)
- [独立工具说明](../tools/README.md)
- [英文项目 README](../README.md)
