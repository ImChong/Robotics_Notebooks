# NVIDIA Physical AI Learning（官方学习门户）

> 来源归档

- **标题：** Physical AI Learning — Welcome
- **类型：** site（厂商官方自学课程门户）
- **来源：** NVIDIA
- **链接：** https://docs.nvidia.com/learning/physical-ai/
- **入库日期：** 2026-05-21
- **一句话说明：** NVIDIA 面向 Physical AI 的免费自学路径索引：Isaac Sim/Lab/ROS、OpenUSD 数字孪生、医疗机器人与 SO-101 操作臂 Sim2Real 等，并指向 Brev 云 GPU 环境与社区支持。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-physical-ai-learning.md`](../../wiki/entities/nvidia-physical-ai-learning.md)

---

## 官方要点摘录

### 定位

- **Physical AI**：能感知真实世界、推理物理与空间关系、通过执行器行动、并适应不可预测环境的 AI 系统（与纯生成式/对话式 AI 区分）。
- 门户提供 **多条自学路径（Learning Paths）**，均为免费自学；可通过 [NVIDIA Omniverse Discord](https://discord.gg/nvidiaomniverse) 获取社区支持。

### 无本地 GPU 时的入口

- **NVIDIA Brev**：预配置 Isaac Sim、Omniverse 与 Physical AI 工作流的云端开发环境。
- 示例 Launchable：Kit App Template、Isaac（Isaac Lab + Isaac Sim 云端运行）。

### 学习路径目录（门户首页，2026-05 抓取）

| 路径 | 级别 / 时长（官方标注） | 关键技能 |
|------|-------------------------|----------|
| [Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) | Intermediate · 6–10 h | Isaac Lab、GR00T、LeRobot、Cosmos、sim-to-real 原则 |
| Assembling Digital Twins With Omniverse and OpenUSD | Intermediate · 3–4 h | OpenUSD、场景组合、资产组织 |
| Getting Started With Isaac Sim | Beginner · 2–3 h | 物理仿真、传感器、环境搭建 |
| Getting Started With Isaac Lab | Intermediate · 3–4 h | RL、GPU 并行训练 |
| Getting Started With Isaac ROS | Intermediate · 2–3 h | ROS 2、NITROS、真机部署 |
| Going Further With Robotics | Advanced · 4–5 h | URDF→USD、资产优化、企业级数字孪生 |
| Getting Started With Isaac for Healthcare | Intermediate · 3–4 h | 医疗场景安全与传感器 |
| Learn OpenUSD | Beginner–Advanced · 自学开源课纲 | USD 基础、组合弧、管线开发 |

### 推荐延伸阅读（页内链接）

- [What is Physical AI?](https://www.nvidia.com/en-us/glossary/generative-physical-ai/) — 生成式 AI 与 Physical AI 对比
- [Learn OpenUSD](https://docs.nvidia.com/learn-openusd/latest/index.html) — USD 完整课纲
- [OpenUSD Development Professional Certification](https://www.nvidia.com/en-us/learn/certification/openusd-development-professional/)

---

## 对 wiki 的映射

- 门户总览与路径选型 → [`wiki/entities/nvidia-physical-ai-learning.md`](../../wiki/entities/nvidia-physical-ai-learning.md)
- SO-101 操作臂 Sim2Real 动手课（本门户 flagship 路径之一）→ [`sources/courses/nvidia_sim_to_real_so101_isaac.md`](../courses/nvidia_sim_to_real_so101_isaac.md) → [`wiki/entities/nvidia-so101-sim2real-lab-workflow.md`](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md)
- Isaac / Omniverse 底座 → [`wiki/entities/nvidia-omniverse.md`](../../wiki/entities/nvidia-omniverse.md)、[`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md)
- LeRobot 数据采集栈 → [`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md)
