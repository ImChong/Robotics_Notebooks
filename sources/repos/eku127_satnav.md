# Eku127/SatNav（代码仓库）

> 来源归档

- **名称：** SatNav
- **类型：** repo（仿真环境、数据工具、基线与在线评测）
- **组织 / 作者：** Eku127（项目作者 Jiajun Jiang）
- **官方仓库：** <https://github.com/Eku127/SatNav>
- **项目页：** <https://eku127.github.io/SatNav/>
- **论文：** <https://arxiv.org/abs/2609.31507>
- **数据：** [SatNav-Episodes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1)（公开 episode）；[SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1)（GeoTIFF 场景需申请）
- **模型：** <https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo>
- **许可证：** 代码 MIT；数据、地图和第三方组件适用条款见仓库 DATA_LICENSE.md 与 NOTICE
- **入库日期：** 2026-10-14
- **一句话说明：** 提供从 GeoTIFF 场景与 episode 加载、SatSim 地理空间环境，到导航策略训练、基线接入、在线评测和轨迹可视化的 UAV-VLN 工具链。

## 仓库结构与可运行入口

| 路径 | 用途 |
|------|------|
| `satnav/` | 核心环境、数据集、模拟器、任务与评测 API |
| `applications/` | 地图、episode、轨迹与 viewer 工具 |
| `baselines/` | 经典 VLN 与多种 LVLM 基线 |
| `configs/` | 公开任务和基线配置 |
| `examples/` | 合成场景下的可运行示例 |
| `scripts/` | 训练、评测、验证及工具入口 |
| `docs/` | 英文与简体中文文档 |

README 快速开始可创建 `environments/satnav/conda.yml` 环境、`pip install -e .` 安装，再运行 `python examples/reference_follower_example.py`；参考 follower 会执行记录路径并输出导航指标、RGB 观测及俯视轨迹图。完整策略训练和评测还需按各 LVLM 的独立依赖指南配置环境。

## 数据与复现边界

- Episode 数据含 instruction、场景、起始位姿、目标、waypoint 和参考路径；提供 train、val_seen、val_unseen split。
- README 说明可直接用仓库自带合成场景跑首个 episode，无需下载全量数据或 checkpoint。
- 高分辨率卫星场景通过 SatNav-Scenes 申请；或使用自己的 API 凭据生成。仅有公开 episode 并不意味着对应 GeoTIFF 可直接获取。
- SwiftVLN 在独立仓库：[Eku127/SwiftVLN](https://github.com/Eku127/SwiftVLN)；SatNav README 指向它作为 SatNav 集成入口。

## 对 wiki 的映射

- 主实体：[SatNav（论文实体）](../../wiki/entities/paper-satnav.md)
- 项目页：[satnav-eku127-github-io.md](../sites/satnav-eku127-github-io.md)
- 论文摘录：[satnav_arxiv_2609_31507.md](../papers/satnav_arxiv_2609_31507.md)
