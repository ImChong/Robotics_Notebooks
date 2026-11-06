---
type: entity
title: RoboTwin 2.0
tags:
- simulation
- data-generation
- dual-arm
- dataset
- hku
- shanghai-ai-lab
- sjtu
- paper
- awesome-world-action-models-rcl
- rcl-wam-catalog
summary: RoboTwin 2.0 是专为双臂机器人设计的自动数据生成与仿真平台，基于 SAPIEN 引擎，支持大规模高质量专家数据合成。
updated: 2026-10-06
project_id: robotwin
sources:
- ../../sources/papers/rcl_awesome_wam_2506_18088_robotwin-2-0-a-scalable-data-generator-a.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
original:
- ../../sources/papers/robotwin_2_0_reading_arxiv_2506_18088.md
related:
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/generative-world-models.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
arxiv: '2506.18088'
venue: ICML 2026
---

# RoboTwin 2.0

**RoboTwin 2.0** 是一个专为双臂机器人操作设计的**自动数据生成与仿真平台**。它建立在 [SAPIEN (仿真引擎)](./sapien.md) 仿真引擎之上，旨在解决具身智能（Embodied AI）中高质量专家数据获取昂贵且难以规模化的问题。

## 一句话定义

RoboTwin 2.0 = **MLLM 写代码 + 仿真闭环纠错的双臂专家数据生成器**，配 731 物体资产库、五轴域随机化与 50 任务基准（arXiv:2506.18088）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 人工智能 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |

| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要？

在具身智能的训练中，**数据规模化（Scaling）** 是核心瓶颈。RoboTwin 通过以下方式提供解决方案：
- **自动数据合成**：不再依赖人类遥操作，通过脚本或预定义策略生成海量专家轨迹。
- **双臂任务聚焦**：针对当前最热门的双臂操作（如 [ALOHA](./aloha.md) 任务）提供深度支持。
- **低门槛上手**：作为 [Lumina 具身智能社区](./lumina-embodied.md)《具身智能百科全书》推荐的实践平台，提供了从数据采集到模型训练的全链路工具。

## 核心特性

- **基于 SAPIEN 引擎**：利用其优秀的物理仿真能力和对 PartNet-Mobility 数据集的支持。
- **任务库**：内置了 50+ 个双臂自动化任务，覆盖了常见的家庭和工业操作场景。
- **真机对齐**：强调 Sim-to-Real 的一致性，生成的轨迹可以直接用于训练并在真机上验证。

## 与其他系统的关系

- **底层驱动**：依赖 [SAPIEN (仿真引擎)](./sapien.md) 进行物理模拟。
- **任务目标**：通常用于生成 [behavior-cloning](../methods/behavior-cloning.md) 或 [action-chunking](../methods/action-chunking.md) (ACT) 所需的训练数据。
- **硬件对应**：其仿真场景常模拟 [ALOHA](./aloha.md) 或类似的双臂遥操作设备。
- **资产生成研究**：近期工作如 [PhysForge（论文实体）](./paper-physforge-physics-grounded-3d-assets.md) 将 **物理接地、可关节 3D 资产** 作为具身与游戏管线的数据引擎，并在论文演示中引用 **RoboTwin** 作为操作仿真导入场景之一（细节以原论文为准）。
- **记忆诊断基准**：**RoboTwin-MeM**（[EventVLA](./paper-eventvla-visual-evidence-memory.md)，arXiv:2606.20092）在 RoboTwin 2.0 上构建 8 项 **非马尔可夫** 双臂任务，用参数 $n$（须动态保留的中间关键帧数，1–5）分层评测 VLA 的 **瞬态证据记忆** 能力，区别于 RMBench 等可被静态初始帧/短期历史「取巧」解决的套件。[Chronos](./paper-chronos.md)（arXiv:2606.30318）在 **RMBench** 报告平均 **73.6%**，并在 **RoboTwin 2.0 Easy** 一般操作协议上平均 **70.0%**（点云 + 全历史 SSM，非记忆专项主张）。
- **AgentLoop 长程编排：** [EmbodiedSkills](../entities/paper-embodiedskills.md)（arXiv:2609.01281）在 RoboTwin 2.0 上报告 **86.20%** 均值（任务适配 π₀.₅ 低层 + Qwen3-VL AgentLoop）；去 verification 降至 **48.2%**。
- **冻结 VLM 语义接口：** [RoboDawn](../entities/paper-robodawn.md)（arXiv:2609.22966）在 C2R **1-shot 73.6%**（GPT-6 Astra），**零样本 53.2%** 仍高于 π₀.₅ **46.0%**（全量后训练）；**500** 评测轨迹 + 演示可在 [robodawn.top/results](https://robodawn.top/results) 回放。
- **世界模型下游栈**：[Dexmal DW05](./dexmal-dw05.md) 以 **RobotWin-style JSONL** 为一等数据接口，发布 **DW05-Robotwin** SFT 权重（含 `norm_stats.json`）与动作条件 online demo；通用 **DW05-Base** 不含 RobotWin policy 归一化统计。
- **物理条件多样性：** [RoboTwin-Phys](./paper-robotwin-phys.md)（arXiv:2609.26292）在 50 任务上 episode 级连续采样 **13** 个物理属性 + **5000+** 带 GT 物理参数的 expert demo；暴露 WAM/VLA 在 **质量/摩擦/动力学** 变化下的 robustness gap（相对仅视觉/布局 DR）。

## 数据速查

- **许可证：** 开源（官方 GitHub 仓库发布）；具体协议以仓库 LICENSE 为准。
- **重定向就绪度：** 仿真自动合成双臂轨迹，绑定特定 embodiment；跨形态部署需重定向或按目标本体重采。

## 核心原理

（据原论文通读，见 [通读归档](../../sources/papers/robotwin_2_0_reading_arxiv_2506_18088.md)）

- **闭环专家代码生成：** 代码 agent（DeepSeek-V3）按任务描述、API 库与约束写 Python；每轮仿真跑 10 次并记录失败原因，VLM 观察 agent 逐帧定位失败步骤；最多连续 5 次修复。多模态反馈使平均代码成功率 62.1% → 71.3%（RoboTwin 1.0 为 47.4% → 63.9%）。
- **五轴域随机化：** 杂物、背景纹理（11000 张生成纹理）、光照、桌面高度、轨迹级语言指令。
- **RoboTwin-OD 资产：** 731 物体 / 147 类，标注抓取点、功能点、放置点、抓取轴及 15 条语言描述。
- **本体感知抓取适配：** 每物体多候选操作位姿 + 偏向可达性的扰动，覆盖 Franka、Piper、UR5、ARX-X5、Aloha-AgileX；平均自动采集成功率 52.2% → 60.5%，低自由度 Piper 提升最大（2.4% → 25.1%）。

## 工程实践

- **数据怎么用：** 官方预采 10 万+ 轨迹（50 任务）；基准协议为 Aloha-AgileX、每任务 50 条干净演示、100 次 rollout，分 Easy（干净）与 Hard（随机化）。
- **预训练建议：** 论文中用域随机化数据预训练（32 任务 × 300 条）使 RDT/Pi0 对未见任务相对提升约 31.9%/29.3%，而干净数据预训练几乎无增益。
- **Sim2Real 配方：** 10 条真机 + 1000 条随机化合成，平均成功率提升 24.4 个点；纯合成零样本在未见背景上也有约 +20 个点；仿真相机位姿加 ≤1 cm 扰动以抗标定误差。
- **自建任务：** 用生成管线写新任务代码，VLM 观察器可选（失败时触发，约 6.9k token/次，可关闭以省成本）。

## 局限

- 论文基准显示 Hard 条件下所有策略大幅退化（Pi0 46.4% → 16.3%，RDT 34.5% → 13.7%，ACT/DP 近乎归零），鲁棒性仍是核心难题。
- DP3 的优势部分来自仿真里的完美点云与干净背景分割，不能直接外推到真机。
- Sim-to-real 仅在 COBOT-Magic 一个平台、4 个任务、RDT 一个骨干上验证；论文自述未来工作是真机部署与多物体复杂任务。
- 代码生成评测仅在 10 个任务上进行，且成功率为仿真内自评。
- 高自由度臂（Franka、UR5）从抓取适配几乎无增益（推断：瓶颈不在抓取候选）。

## 关联页面

- [Lumina 具身智能社区](./lumina-embodied.md) — Guide 与 RoboTwin 2.0 Talk 的社区入口
- [SAPIEN](./sapien.md) / [ALOHA](./aloha.md)
- [Behavior Cloning](../methods/behavior-cloning.md) / [Action Chunking](../methods/action-chunking.md)
- [Motubrain](./paper-motubrain.md) — 官方报 RoboTwin 2.0 Clean/Random 95.8/96.1（仓占位）
- [G0.5](./paper-galaxea-g05.md) — 开源 AR VLA；RoboTwin 93.7/92.8（arXiv:2608.11739）
- [Rift](./paper-rift-wam.md) — 免 rollout WAM；RoboTwin 92.9/92.6（arXiv:2608.11521；未开源）
- [Temporal GRPO](./paper-temporal-grpo.md) — OpenVLA-OFT + 阶段 GRPO；宏平均 75.8%（arXiv:2608.13026；未开源）
- [Ego2Robot](./paper-ego2robot.md) — 扩展 RoboTwin 2.0 为视觉/布局/本体/语义四轴解耦评测（arXiv:2608.02580）
- [ME-Dex 1.0](./paper-me-dex-1-0.md) — 触觉增强 WAM；Clean→Random avg **78.9%**；推理 runtime 已开源（arXiv:2609.21449）
- [RoboTwin-Phys](./paper-robotwin-phys.md) — 物理参数多样性评测轴（arXiv:2609.26292）

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

- [awesome-world-action-models-rcl](paper-rcl-wam-robot-learning-control-survey.md)
- [vla](../methods/vla.md)
- [locomotion](../tasks/locomotion.md)
- [agibot-go-3-preview](agibot-go-3-preview.md) — AGIBOT GO-3 Preview（RoboTwin 2.0 Co-train 榜单核对）
- [paper-a3-vla-attention-alignment](paper-a3-vla-attention-alignment.md) — A³-VLA（RoboTwin 2.0 上的注意力对齐微调）

## 参考来源
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)
- [Lumina 官网归档](../../sources/sites/lumina-embodied-ai.md)
- [RoboTwin 官方仓库](https://github.com/msc-robotwin/robotwin)

- [`sources/papers/rcl_awesome_wam_2506_18088_robotwin-2-0-a-scalable-data-generator-a.md`](../../sources/papers/rcl_awesome_wam_2506_18088_robotwin-2-0-a-scalable-data-generator-a.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://icml.cc/virtual/2026/poster/62192>

- [原论文与官方资源](https://robotwin-platform.github.io)

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://icml.cc/virtual/2026/poster/62192)
