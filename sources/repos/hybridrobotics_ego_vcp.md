# HybridRobotics/Ego-VCP

> 来源归档（repo）

- **标题：** Ego-VCP — Ego-Vision World Model for Humanoid Contact Planning
- **代码：** <https://github.com/HybridRobotics/Ego-VCP>
- **项目页：** <https://ego-vcp.github.io/>
- **论文：** <https://arxiv.org/abs/2510.11682>
- **类型：** research-code（Isaac Lab + rsl_rl 人形视觉世界模型与采样式 MPC）
- **License：** MIT
- **首次入库：** 2026-07-26

## 一句话摘要

官方实现：Isaac Lab 中采集 demonstration-free ego-depth + 本体数据、离线训练世界模型，并用 `play_wm.py` 做采样式接触规划；配套低层控制器与 HuggingFace 数据集。

## 关键复现入口（README）

| 入口 | 作用 |
|------|------|
| `ego_vcp/scripts/collect.py` | 多环境采集（`g1_wall` / `g1_ball` / `g1_tunnel`） |
| 离线世界模型训练脚本（仓库 docs） | 用采集数据训 latent WM |
| `ego_vcp/scripts/play_wm.py` | 加载 `wm_logs/.../world_model.pt` 做在线规划演示 |
| `logs/g1_flat/.../policy.pt` | 低层控制器 |
| HF `Hang917/EgoVCP_Dataset` | 公开训练数据 |

## 对 wiki 的映射

- [`wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md`](../../wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)
- [`sources/papers/ego_vcp_arxiv_2510_11682.md`](../papers/ego_vcp_arxiv_2510_11682.md)
- [`sources/sites/ego-vcp-github-io.md`](../sites/ego-vcp-github-io.md)
