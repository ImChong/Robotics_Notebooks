---
type: overview
tags: [overview, survey, vla, manipulation, humanoid, world-model, benchmark, technology-map]
status: complete
updated: 2026-11-11
related:
  - embodied-station-2026-11-10-22-papers-technology-map.md
  - embodied-station-2026-11-09-29-papers-technology-map.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/wechat_embodied_chunk_replanning_53_papers_exrl_2026-11-11.md
original:
  - ../../sources/blogs/wechat_embodied_chunk_replanning_53_papers_exrl_2026-11-11.md
summary: "具身智能小站 2026-11-11 的 53 篇论文速览的阅读坐标：长动作块重规划、触觉与评测、人形与导航、世界模型、安全与基准、工具链；每条指向唯一详情节点（新建 43 / 复用 10），逐篇通读新建节点并标出文章说法与论文的出入。"
---

# 长动作块重规划、触觉与评测等 53 篇论文阅读坐标

> **本页定位**：为[具身智能小站 · 53 篇](https://mp.weixin.qq.com/s/jhHMptmxYuIHfDUjTsmdtw)提供阅读坐标，并把 53 条论文逐条指向站内唯一详情节点。

## 一句话定义

具身智能小站 2026-11-11 的 53 篇论文速览（1 篇深读、2 篇跟进、50 篇扫读）的阅读坐标页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 世界动作模型 |
| RL | Reinforcement Learning | 强化学习 |
| HRI | Human-Robot Interaction | 人机交互 |
| SLAM | Simultaneous Localization and Mapping | 同时定位与建图 |

## 为什么重要

- 小站每条只有一句话判断；本页让每条落到可追溯的详情节点，节点里有逐篇读论文后的核对结论。
- 文章标题谈「动作块重规划」，但 53 篇里只有 ExRL 研究它；归类时不能只看栏目标题。

## 核心原理

按文章分栏与本库新建节点的读后观察（以下归纳是本库判断）：

- **动作块与执行频率：** ExRL（学习执行长度，冻结 π0.5）、异步 ACT（控制感知指令生成，OpenVLA-OFT 部分仅一个任务）、MoE-DAgger（选择性人工纠正）、Look Ahead（冻结 ACT 加仿真器前瞻搜索）、Learning When to Think（隐表示上的自适应计算深度）。
- **VLA 适配与蒸馏：** weightsmith（闭式权重编辑）、RAGu（检索示范引导）、VLA-MOPD（步均值蒸馏）、Thermo-VLA（热成像残差）、DexForesight、ForeAct（几何前瞻 / 辅助头）、Streaming Failure Forecasting（失败监测）、世界上下文学习（复用）。
- **世界模型：** ObjectWorld、Σ-LeWM、最近邻候选训练（Nearest-of-m）、SWAM（复用）、DreamVLA。
- **触觉与数据集：** PCTac、LingBot-Touch（复用）、RoboMoCap、SurgTAP-3D、Grasping the Future。
- **人形与导航：** Kimodo-Policy、LHNav、Compass-1、SAM-VTP、Maze Runner、FLAVOR、PLS-GS（均涉及，部分复用）。
- **评测与安全：** Embodied First Exam（复用）、HARP-Bench、随机抽样挑战（KALLAX）、CREG-Bench、ArmedAgent（编码智能体的物理风险）、类型化计划的正确性边界、倾覆预警。
- **工具链与仿真：** Kinora、ROS2MR、ExoStack（复用）、编队控制教程、股静脉压缩仿真、LawKAN、ASPIRE。
- **证据强度：** 新建节点里带真机数字的多为每任务 10–30 次；不少结果全在仿真或离线评测，且若干论文的代码 / 数据仍标「coming soon」。

## 工程实践

- 查证某篇论文的细节时，以详情节点和其引用的 arXiv 原文为准，不以文章一句话判断为准。
- 新增条目时先按 arXiv ID 去重（同一 ID 只允许一个 canonical 节点）。

## 局限

- 小站只给一句话判断；新建的 43 个节点逐篇读了论文，复用的 10 个节点沿用既有页面、**没有重新读论文**。
- 上面的归类是本库判断，只基于新建节点读到的内容。

## 论文索引

| # | 档 | 论文节点 | arXiv | 状态 |
|---|----|----------|-------|------|
| 01 | 深读 | [长动作块，执行多久再重规划](../entities/paper-exrl-execution-length-control.md) | [2611.13482](https://arxiv.org/abs/2611.13482) | 新建 |
| 02 | 跟进 | [换一种触觉传感器，表征能否复用](../entities/paper-lingbot-touch.md) | [2611.12806](https://arxiv.org/abs/2611.12806) | 复用 |
| 03 | 跟进 | [会写代码之后，还要通过机器人实操题](../entities/embodied-first-exam.md) | [2611.12177](https://arxiv.org/abs/2611.12177) | 复用 |
| 04 | 扫读 | [从动作生成先验，走向人形闭环操作](../entities/paper-kimodo-policy.md) | [2611.11554](https://arxiv.org/abs/2611.11554) | 复用 |
| 05 | 扫读 | [把试错经验存进记忆，而非模型权重](../entities/paper-world-in-context-learning-roboscript.md) | [2611.12885](https://arxiv.org/abs/2611.12885) | 复用 |
| 06 | 扫读 | [一句指令被误解，能否直接修正策略权重](../entities/paper-weightsmith-instruction-distillation.md) | [2611.11894](https://arxiv.org/abs/2611.11894) | 新建 |
| 07 | 扫读 | [人在旁边动起来，操作策略还可靠吗](../entities/paper-harp-bench-active-humans.md) | [2611.11123](https://arxiv.org/abs/2611.11123) | 新建 |
| 08 | 扫读 | [只用单目视频，导航如何记住走过的路](../entities/paper-compass-1-navigation.md) | [2611.12087](https://arxiv.org/abs/2611.12087) | 复用 |
| 09 | 扫读 | [多位任务专家，怎样合成一套 VLA](../entities/paper-vla-mopd.md) | [2611.12286](https://arxiv.org/abs/2611.12286) | 新建 |
| 10 | 扫读 | [世界动作模型扩量，先守住跨模态对齐](../entities/paper-swam-cross-modal-alignment.md) | [2611.12870](https://arxiv.org/abs/2611.12870) | 复用 |
| 11 | 扫读 | [把世界模型的空间理解用于动作学习](../entities/paper-foreact.md) | [2611.11500](https://arxiv.org/abs/2611.11500) | 新建 |
| 12 | 扫读 | [可通行区域分割，需要记忆也需要几何](../entities/paper-sam-vtp.md) | [2611.12099](https://arxiv.org/abs/2611.12099) | 新建 |
| 13 | 扫读 | [检索旧示范，在不改权重时引导新动作](../entities/paper-ragu-retrieval-augmented-guidance.md) | [2611.10894](https://arxiv.org/abs/2611.10894) | 新建 |
| 14 | 扫读 | [让系统辨识留下可检查的推理证据](../entities/paper-aspire.md) | [2611.13114](https://arxiv.org/abs/2611.13114) | 新建 |
| 15 | 扫读 | [让潜在世界模型保留与动作有关的信息](../entities/paper-sigma-lewm.md) | [2611.12732](https://arxiv.org/abs/2611.12732) | 新建 |
| 16 | 扫读 | [已知旋转组合规则，就把它写进模型](../entities/paper-lawkan.md) | [2611.12567](https://arxiv.org/abs/2611.12567) | 新建 |
| 17 | 扫读 | [视觉策略异步运行，怎样让动作接得更顺](../entities/paper-cac-async-act.md) | [2611.11089](https://arxiv.org/abs/2611.11089) | 新建 |
| 18 | 扫读 | [适配 VLA，验证损失未必选出最会动的模型](../entities/paper-vla-adaptation-chess.md) | [2611.12676](https://arxiv.org/abs/2611.12676) | 复用 |
| 19 | 扫读 | [把第一视角视频和高精度全身动作对齐](../entities/paper-robomocap.md) | [2611.10994](https://arxiv.org/abs/2611.10994) | 新建 |
| 20 | 扫读 | [世界模型候选更多，规划就一定更好吗](../entities/paper-nearest-of-m-world-models.md) | [2611.11887](https://arxiv.org/abs/2611.11887) | 新建 |
| 21 | 扫读 | [机器人从环境状态中提出下一项任务](../entities/paper-cafe-closed-loop-task-formulation.md) | [2611.12983](https://arxiv.org/abs/2611.12983) | 新建 |
| 22 | 扫读 | [自然语言车队调度，类型正确还不够](../entities/paper-typed-plans-fleet-control.md) | [2611.12654](https://arxiv.org/abs/2611.12654) | 新建 |
| 23 | 扫读 | [边听边跳，人形机器人提前接住节拍](../entities/paper-aheadance.md) | [2611.13370](https://arxiv.org/abs/2611.13370) | 新建 |
| 24 | 扫读 | [从网络视频联合预测手与物体的未来](../entities/paper-grasping-the-future.md) | [2611.12879](https://arxiv.org/abs/2611.12879) | 新建 |
| 25 | 扫读 | [跨实验室成功率，先统一任务怎么抽样](../entities/paper-stochastic-challenges-kallax.md) | [2611.12804](https://arxiv.org/abs/2611.12804) | 新建 |
| 26 | 扫读 | [双臂规划，把可执行条件带进每一步](../entities/paper-bridging-planning-execution-bimanual.md) | [2611.12409](https://arxiv.org/abs/2611.12409) | 新建 |
| 27 | 扫读 | [让压力直接变成颜色，供操作策略读取](../entities/paper-pctac.md) | [2611.12383](https://arxiv.org/abs/2611.12383) | 新建 |
| 28 | 扫读 | [长程人形导航，让规划与控制各按节奏运行](../entities/paper-lhnav.md) | [2611.12136](https://arxiv.org/abs/2611.12136) | 复用 |
| 29 | 扫读 | [仅看外观不够，让 VLA 感知物体冷热](../entities/paper-thermo-vla.md) | [2611.11981](https://arxiv.org/abs/2611.11981) | 新建 |
| 30 | 扫读 | [机器人翻物体，让代码智能体在试错中修策略](../entities/paper-flip-anything-agentic-self-learning.md) | [2611.11941](https://arxiv.org/abs/2611.11941) | 复用 |
| 31 | 扫读 | [手术视频三维追踪，先把接触关系保留下来](../entities/paper-surgtap-3d.md) | [2611.11697](https://arxiv.org/abs/2611.11697) | 新建 |
| 32 | 扫读 | [部署后的失败监测，持续更新最后一层](../entities/paper-streaming-failure-forecasting.md) | [2611.11688](https://arxiv.org/abs/2611.11688) | 新建 |
| 33 | 扫读 | [物体被遮挡以后，世界模型还要预测整体运动](../entities/paper-objectworld.md) | [2611.11085](https://arxiv.org/abs/2611.11085) | 新建 |
| 34 | 扫读 | [何时请人接管，要分清换阶段与真失败](../entities/paper-moe-dagger.md) | [2611.10825](https://arxiv.org/abs/2611.10825) | 新建 |
| 35 | 扫读 | [简单状态少算几步，复杂状态再多想一会](../entities/paper-aic-adaptive-internal-computation.md) | [2611.10822](https://arxiv.org/abs/2611.10822) | 新建 |
| 36 | 扫读 | [编程智能体接上机械臂，风险会怎样变化](../entities/paper-armedagent-coding-agent-embodied-safety.md) | [2611.11438](https://arxiv.org/abs/2611.11438) | 新建 |
| 37 | 扫读 | [多设备共享世界状态，证据链也要一起评测](../entities/paper-creg-evidence-grounded-shared-belief.md) | [2611.11842](https://arxiv.org/abs/2611.11842) | 新建 |
| 38 | 扫读 | [动作看起来要失败时，再触发仿真搜索](../entities/paper-look-ahead-search-pusht.md) | [2611.10769](https://arxiv.org/abs/2611.10769) | 新建 |
| 39 | 扫读 | [越野车倾覆预警，把不同风险信号分开学](../entities/paper-rover-tipover-multichannel.md) | [2611.12261](https://arxiv.org/abs/2611.12261) | 新建 |
| 40 | 扫读 | [大范围三维探索，把精细地图留在近处](../entities/paper-flavor-hierarchical-spatio-temporal-map-exploration.md) | [2611.11489](https://arxiv.org/abs/2611.11489) | 新建 |
| 41 | 扫读 | [灵巧操作，把未来空间变化蒸馏进策略](../entities/paper-dexforesight.md) | [2611.11166](https://arxiv.org/abs/2611.11166) | 新建 |
| 42 | 扫读 | [用激光几何约束，让高斯表示更贴近表面](../entities/paper-pls-gs.md) | [2611.10699](https://arxiv.org/abs/2611.10699) | 新建 |
| 43 | 扫读 | [窄走廊导航，把定位和避障接到一起](../entities/paper-maze-runner.md) | [2611.12949](https://arxiv.org/abs/2611.12949) | 新建 |
| 44 | 扫读 | [可穿戴机器人，把传感与助力模块解耦](../entities/paper-exostack.md) | [2611.12306](https://arxiv.org/abs/2611.12306) | 复用 |
| 45 | 扫读 | [生成动作反复修改，也需要版本和验收记录](../entities/paper-kinora.md) | [2611.11601](https://arxiv.org/abs/2611.11601) | 新建 |
| 46 | 扫读 | [多机器人搜救，把感知误差带进协同决策](../entities/paper-pa-sar.md) | [2611.11648](https://arxiv.org/abs/2611.11648) | 新建 |
| 47 | 扫读 | [只靠局部本体感知，估计软体整体运动](../entities/paper-deepsomo.md) | [2611.11242](https://arxiv.org/abs/2611.11242) | 新建 |
| 48 | 扫读 | [生成的人类动作，怎样变成机器人学习数据](../entities/paper-dreamvla.md) | [2611.10872](https://arxiv.org/abs/2611.10872) | 新建 |
| 49 | 扫读 | [静脉压缩仿真，速度之外还要看误差](../entities/paper-femoral-vein-compressibility-sim.md) | [2611.12818](https://arxiv.org/abs/2611.12818) | 新建 |
| 50 | 扫读 | [同一状态多采几种动作，保留示范多样性](../entities/paper-vine-sampling-diffusion-policy.md) | [2611.10949](https://arxiv.org/abs/2611.10949) | 新建 |
| 51 | 扫读 | [从顶视相机起步，搭建小车编队实验](../entities/paper-formation-control-tutorial.md) | [2611.11164](https://arxiv.org/abs/2611.11164) | 新建 |
| 52 | 扫读 | [把 ROS 2 轨迹搬进混合现实预览](../entities/paper-ros2mr.md) | [2611.11830](https://arxiv.org/abs/2611.11830) | 新建 |
| 53 | 扫读 | [量子策略到机器人动作，中间链路如何追溯](../entities/paper-qubit-to-motion.md) | [2611.11580](https://arxiv.org/abs/2611.11580) | 新建 |

## 关联页面

- [上一期（11-10）22 篇阅读坐标](./embodied-station-2026-11-10-22-papers-technology-map.md)
- [前期（11-09）29 篇阅读坐标](./embodied-station-2026-11-09-29-papers-technology-map.md)

## 参考来源

- [11-11 速览来源归档](../../sources/blogs/wechat_embodied_chunk_replanning_53_papers_exrl_2026-11-11.md)

## 继续阅读

- 各详情节点引用的 arXiv 原文；复用节点需要时再逐篇重读。
