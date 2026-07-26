# Ego-VCP / Ego-Vision World Model for Humanoid Contact Planning

> 来源归档（paper）

- **标题：** Ego-Vision World Model for Humanoid Contact Planning（项目简称 **Ego-VCP**）
- **类型：** paper
- **机构：** UC Berkeley；University of Michigan, Ann Arbor；Chinese University of Hong Kong
- **Venue：** ICRA 2026
- **arXiv：** <https://arxiv.org/abs/2510.11682>
- **项目页：** <https://ego-vcp.github.io/>
- **代码：** <https://github.com/HybridRobotics/Ego-VCP>（MIT）
- **数据集：** <https://huggingface.co/datasets/Hang917/EgoVCP_Dataset>
- **入库日期：** 2026-07-26（相对 42 篇栈策展摘录的加厚归档）
- **一句话说明：** 用离线 demonstration-free 数据学习压缩潜空间世界模型，再以采样式 MPC（CEM + 学习价值）做人形接触感知在线规划；真机以本体 + 第一视角深度驱动，低层 RL 控制器执行高层命令。

## 核心论文摘录（MVP）

### 1) 问题：接触规划既难优化又难 RL

- **链接：** <https://arxiv.org/abs/2510.11682>
- **摘录要点：** 人形要利用接触（扶墙、挡球、钻矮拱）而非只避碰；传统优化规划遇接触组合爆炸，在线 RL 样本效率低且多任务弱。作者用 **learned world model + sampling-based MPC** 在潜空间预测任务结果，并用 **surrogate value** 缓解稀疏接触奖励与传感噪声。
- **对 wiki 的映射：**
  - [Ego-VCP 实体页](../../wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)
  - [WAM×运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md) — ① 在线规划

### 2) 方法：潜空间展开 + CEM 筛选 + 只执行第一步

- **链接：** <https://ego-vcp.github.io/>
- **摘录要点：** 深度图 + 本体 → 紧凑内部状态；MPC 采样大量高层动作序列，世界模型前滚估计好坏与安全性（失败概率/价值），CEM 多轮筛选后 **只执行最优序列第一步** 并重规划。公开答复给出部署口径：笔记本 RTX 2060 上高层视觉 MPC **约 25 Hz**，世界模型推理占主要耗时；项目页/公众号策展亦给出每轮约 **1024** 候选、时域约 **4** 步的读法。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 3) 开源与复现入口

- **链接：** <https://github.com/HybridRobotics/Ego-VCP>
- **摘录要点：** Isaac Lab 工作流：`collect.py` 采 demonstration-free 数据 → 离线训世界模型 → `play_wm.py` 加载 `wm_logs/.../world_model.pt` 做规划；低层控制器 checkpoint 与 HuggingFace 数据集一并公开。
- **对 wiki 的映射：**
  - [HybridRobotics/Ego-VCP 仓库归档](../repos/hybridrobotics_ego_vcp.md)
  - [ego-vcp.github.io 站点归档](../sites/ego-vcp-github-io.md)

## 关键术语

- **Ego-VCP：** Ego-Vision Contact Planning / 项目简称；论文全称为 Ego-Vision World Model for Humanoid Contact Planning。
- **Surrogate value：** 学习得到的稠密价值代理，替代稀疏接触奖励用于 MPC 打分。

## 关联 Wiki 页面

- [paper-hrl-stack-33-ego_vision_world_model_for_humanoid](../../wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)
- [wam-motion-control-five-paths](../../wiki/overview/wam-motion-control-five-paths.md)

## 当前提炼状态

- [x] arXiv / 项目页 / 代码 / 数据集入口
- [x] 在线规划机制与部署口径
- [x] wiki 映射（复用既有实体页，避免重复节点）
