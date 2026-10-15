---
type: entity
tags: [paper, embodied-ai, navigation, vln, vlm, pixel-to-3d, memory, grpo, zhejiang-university]
status: complete
updated: 2026-10-15
arxiv: "2608.17512"
project_id: embodied-navigator-tamp-nav
project: https://zju-omniai.github.io/Embodied-Navigator/
code: https://github.com/ZJU-OmniAI/Embodied-Omni
related:
  - ../tasks/vision-language-navigation.md
  - ../methods/vla.md
  - ./paper-supernav-agentic-navigation.md
sources:
  - ../../sources/papers/embodied_navigator_arxiv_2608_17512.md
summary: "Embodied-Navigator（TAMP-Nav，arXiv:2608.17512）：VLM 选二维像素，经深度投影给 SLAM；选择性 CoT、锚点轨迹记忆与 STI 压缩长程上下文，双层 GRPO 优化；R2R-CE Val-Unseen SR 66.2%，用 90k 轨迹。"
---

# Embodied-Navigator：用像素目标、选择性推理与记忆做导航

**Embodied-Navigator**（方法名 **TAMP-Nav**；[arXiv:2608.17512](https://arxiv.org/abs/2608.17512)，[项目页](https://zju-omniai.github.io/Embodied-Navigator/)，[代码](https://github.com/ZJU-OmniAI/Embodied-Omni)）由浙江大学与浙江人形机器人创新中心团队提出。它把导航接口、推理调度、长程记忆和策略优化放进同一个 VLN 框架：VLM 在 RGB 图像上选像素目标，深度将其投影为三维 waypoint，低层 SLAM 控制器负责执行。

## 一句话定义

**让 VLM 在熟悉的二维图像空间“指路”，再由几何感知与 SLAM 执行器把像素目标落到三维环境；把思考与高保真记忆留给关键节点。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TAMP-Nav | TAMP-Nav | 论文提出的统一 VLN 框架 |
| CoT | Chain-of-Thought | 逐步推理；该方法选择性触发 |
| STI | Space-Time Indicator | 以轻量时空标记压缩冗余轨迹历史 |
| GRPO | Group Relative Policy Optimization | 论文双层策略优化所用算法 |
| VLN | Vision-and-Language Navigation | 视觉与语言条件下的具身导航 |
| SR / SPL | Success Rate / Success weighted by Path Length | 成功率 / 路径长度加权成功率 |

## 方法结构

### 1. Point：Pixel-to-3D Action Formulation

VLM 从观测图像选一个二维像素点。系统读取该点的深度并投影至三维空间，将几何目标交给固定的低层 SLAM 控制器。这样把高层模型的动作接口贴近其二维视觉预训练能力，同时保留可执行的三维导航目标。

### 2. Think：按需触发推理

模型学习在导航关键决策处生成 CoT，而不是每一步都展开推理。论文报告，R2R-CE 上自动触发 CoT 的比率为 **26.3%**，SR 为 **66.2%**；密集 CoT 的 SR 为 **66.8%**，但推理比率为 100%。这些数值来自同一论文评测设置。

### 3. Memorize：锚点轨迹记忆

Anchor-Trajectory Memory（AT-Mem）以关键节点作为长程记忆锚点，高保真保留重要轨迹；冗余部分压缩为 STI，帮助模型在长距离导航中维持时空信息。论文长程子集实验中，AT-Mem 变体 SR 为 **49.8%**；消融结果显示去掉 STI 会降低成功率，说明压缩表示是方法组成部分。

### 4. Align：双层 GRPO

训练把全局轨迹结果奖励与细粒度过程奖励叠加，令导航策略同时受到任务结果和局部决策反馈约束。论文在 R2R-CE Val-Unseen 报告双层 GRPO 相对仅 SFT 将 SR 从 **55.7% 提至 66.2%**；此比较对应论文控制实验，不能泛化为对所有 SFT 模型的保证。

## 论文报告结果

| 基准 / 设置 | 结果 |
|-------------|------|
| R2R-CE Val-Unseen | SR 66.2%，SPL 58.8% |
| RxR-CE Val-Unseen | SR 65.7%，SPL 56.9% |
| 训练数据 | 90k 条训练轨迹，约 700k 次交互 |
| 平均交互长度 | 约 9 次 / 轨迹；论文对比 DualVLN 与 StreamVLN 约 30 次 |
| 推理速度 | 单张 A800 平均 16.58 秒 / 任务 |
| 长程子集 | AT-Mem 变体 SR 49.8%（专家路径 >12.5 m 的子集） |

数值仅按论文对应的验证集、轨迹子集和硬件设置解读。跨基准的 SR、SPL 或推理耗时不宜直接横向比较。

## 与其他工作对比

- [SuperNav](./paper-supernav-agentic-navigation.md) 同样用视觉点目标连接高层模型与移动执行，但重点是预训练 MLLM 的 agent harness、工具与 Skills；TAMP-Nav 的重点是对导航策略进行训练，并结合选择性 CoT、锚点记忆和双层 GRPO。
- **DualVLN / StreamVLN：** 论文对比的平均交互长度约 30 次 / 轨迹，TAMP-Nav 约 9 次（仅为论文报告口径，未与相同硬件下的端到端耗时逐项对齐）。
- **密集 CoT 变体：** SR 66.8% 略高于选择性 CoT 的 66.2%，但推理比率为 100% 对 26.3%；TAMP-Nav 的取舍是以极小 SR 代价换推理开销。

## 开源与复现状态

论文 v2 于 **2026-08-27** 更新。项目主页说明完整系统还会使用深度完成像素到三维投影，并用里程计编码 STI。代码链接现指向作者迁移后的 [ZJU-OmniAI/Embodied-Omni](https://github.com/ZJU-OmniAI/Embodied-Omni)；复现时需核对该仓当前提供的训练、数据与评测组件，不能仅凭论文主页推断所有真机部署依赖均已开源。

## 结论

**TAMP-Nav 把导航 VLM 的动作接口、推理时机和轨迹记忆共同纳入优化。**它的可借鉴点是二维像素目标降低高层动作表示难度，关键节点推理和记忆锚点控制长程上下文，而双层 GRPO 对局部执行过程提供更密集反馈。

## 关联页面

- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md) — 任务定义、数据集与相关方法
- [VLA 方法页](../methods/vla.md) — 更一般的视觉–语言–动作模型
- [SuperNav](./paper-supernav-agentic-navigation.md) — 同用视觉点目标的 agentic 导航 harness

## 参考来源

- [论文摘录](../../sources/papers/embodied_navigator_arxiv_2608_17512.md)
- [项目主页](https://zju-omniai.github.io/Embodied-Navigator/)
- [作者代码仓库](https://github.com/ZJU-OmniAI/Embodied-Omni)
