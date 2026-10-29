# Genie Sim 官方源码

- **类型：** repo
- **机构：** 智元机器人
- **链接：** https://github.com/AgibotTech/genie_sim
- **项目页：** https://agibot-world.com/genie-sim
- **论文：** https://arxiv.org/abs/2601.02078
- **核查日期：** 2026-10-28；源码快照 [`30d2bc1`](https://github.com/AgibotTech/genie_sim/tree/30d2bc123d4eda624770dc0daa528541257de580)
- **一句话说明：** 场景重建与生成、机器人仿真、数据采集和策略评测工具链。
- **沉淀到 wiki：** [Genie Sim 3.0](../../wiki/entities/genie-sim-3.md)

## 开放状态与边界

官方仓库公开代码，README 链接 [资产](https://huggingface.co/datasets/agibot-world/GenieSimAssets)与[数据集](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)。README 的 roadmap 仍列出向 Hugging Face 上传全部资产和数据集，因此不能据此断言所有资产均已上传。`source/geniesim_*` 和 `source/data_collection` 默认 MPL 2.0，特殊许可及 `scene_reconstruction` 的多种许可证须分别核对。项目页本次抓取未返回正文，开放状态依据官方仓库 README，未能独立验证项目页。

## 代码入口与运行路径

| 入口 | 职责 |
|------|------|
| [source/README.md](https://github.com/AgibotTech/genie_sim/blob/30d2bc123d4eda624770dc0daa528541257de580/source/README.md) | 模块地图与两条运行栈的边界 |
| `source/geniesim_cli` | `geniesim` CLI、容器与工作区构建 |
| `source/geniesim_benchmark` | 直接使用 Isaac Sim 的基准任务、推理与评分 |
| [source/geniesim_ros](https://github.com/AgibotTech/genie_sim/blob/30d2bc123d4eda624770dc0daa528541257de580/source/geniesim_ros/README.md) | ROS 2 实时物理、渲染、机器人、MoveIt 与遥操作 |
| `source/geniesim_generator` / `geniesim_world` | 场景生成 / 全景图生成三维世界 |
| `source/geniesim_teleop` / `data_collection` | VR 遥操作 / 自动数据采集 |
| `source/rlinf_geniesim` / `scene_reconstruction` | RLinf 训练 / 三维重建，独立维护 |

官方快速启动先在宿主机执行 `pip install -e source/geniesim_cli/`、`geniesim docker build` 与 `geniesim docker up`；实时引擎再在容器内执行 `geniesim ros build dev` 并按场景和 launcher 启动。资产需单独下载；遥操作、生成器和 world 为可选依赖。详见固定版本 [README](https://github.com/AgibotTech/genie_sim/blob/30d2bc123d4eda624770dc0daa528541257de580/README.md)。

## 对 wiki 的映射

- [Genie Sim 3.0](../../wiki/entities/genie-sim-3.md)
