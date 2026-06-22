# 10年间，哪些仿真平台真正改变了机器人学习？

> 来源归档（blog / 微信公众号）

- **标题：** 10年间，哪些仿真平台真正改变了机器人学习？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/iaw_lWAR--AwppyMeIK4lw
- **发表日期：** 2026-06-22
- **入库日期：** 2026-06-22
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））；正文约 0.52 万字 / 15 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **姊妹篇：** [训练栈分层解读](wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)（`Z9pgVa48wQKLYVRD3psnhw`）；[VLN 10 篇盘点](wechat_shenlan_vln_10_papers_survey.md)（`2_dYaN6IeWn_vvS_jmGqRQ`）
- **一句话说明：** 按 2010 年后具身智能演进脉络盘点 **TOP 8 仿真平台**（MuJoCo → BEHAVIOR-1K），从设计初衷、核心贡献与社区推动作用梳理；核心判断：仿真平台轨迹反映研究重心从底层物理精度 → 视觉交互 → 渲染吞吐 → GPU 并行 → 泛化基准 → 人类需求对齐的迁移。

## 核心摘录（归纳，非全文）

### 主判断

- **效率指数级跃迁：** VLA/VLN 等任务推高对高质量大规模交互数据的需求；每次智能跃迁都踩在仿真平台基石上。
- **演进脉络：** MuJoCo（物理控制奠基）→ AI2-THOR（室内视觉交互）→ Matterport3D Simulator（VLN 真实感基准）→ Habitat（渲染速度）→ iGibson（真实感+物理交互）→ Isaac Gym（GPU 并行）→ ManiSkill2（操作泛化基准）→ BEHAVIOR-1K（人类需求对齐）。
- **文尾补充：** CARLA（自动驾驶）、PyBullet/Flex（软体形变）、RoboGen、Genesis 等在各方向仍有重要贡献。

### TOP 8 速查

| # | 平台 | 年份 | 被引量（文内） | 核心贡献 |
|---|------|------|----------------|----------|
| 01 | MuJoCo | 2012 | 9530 | 连续时间动力学、接触优化；OpenAI Gym 底层 |
| 02 | AI2-THOR | 2017 | 1510 | 室内 3D + 细粒度物体状态交互 |
| 03 | Matterport3D Simulator | 2018 | 2270 | 真实扫描全景 RGB-D + R2R 基准 |
| 04 | Habitat | 2019 | 2426 | 单 GPU 数千–上万 FPS 渲染；Habitat 2.0/3.0 扩展交互 |
| 05 | iGibson | 2020 | 268 | PyBullet + 大规模真实感场景 + 可交互日常物体 |
| 06 | Isaac Gym | 2021 | 1820 | 端到端 GPU 原生仿真管线 |
| 07 | ManiSkill2 | 2023 | 260 | 基于 SAPIEN 的跨类别操作泛化基准 |
| 08 | BEHAVIOR-1K | 2023 | 514 | 1000 种人类日常活动 + Omniverse 逼真仿真 |

## 对 wiki 的映射

- [sim-platforms-decade-technology-map](../../wiki/overview/sim-platforms-decade-technology-map.md)（本次升格主页面）
- 实体节点：[mujoco](../../wiki/entities/mujoco.md)、[ai2-thor](../../wiki/entities/ai2-thor.md)、[matterport3d-simulator](../../wiki/entities/matterport3d-simulator.md)、[habitat-sim](../../wiki/entities/habitat-sim.md)、[igibson](../../wiki/entities/igibson.md)、[isaac-gym](../../wiki/entities/isaac-gym.md)、[maniskill2](../../wiki/entities/maniskill2.md)、[behavior-1k](../../wiki/entities/behavior-1k.md)
- 文尾提及：[carla](../../wiki/entities/carla.md)、[pybullet](../../wiki/entities/pybullet.md)、[robogen](../../wiki/entities/robogen.md)、[genesis-sim](../../wiki/entities/genesis-sim.md)、[sapien](../../wiki/entities/sapien.md)
- 交叉：[simulator-selection-guide](../../wiki/queries/simulator-selection-guide.md)、[robot-training-stack-layers-technology-map](../../wiki/overview/robot-training-stack-layers-technology-map.md)、[vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)

## 可信度与使用边界

- 本文为 **微信公众号策展盘点**；被引量与年份以文内统计为准，会随时间变化。
- 各平台细节以官方论文 / 文档 / 仓库 README 为准。
- 原始抓取正文见 [wechat_sim_platforms_top8_2026-06-22.md](../raw/wechat_sim_platforms_top8_2026-06-22.md)。

## 当前提炼状态

- [x] Agent Reach v1.5.0 + wechat-article-for-ai 正文抓取与 TOP 8 归纳
- [x] 各平台实体页与主 overview 升格
- [x] 与既有训练栈分层 / VLN 地图交叉映射
