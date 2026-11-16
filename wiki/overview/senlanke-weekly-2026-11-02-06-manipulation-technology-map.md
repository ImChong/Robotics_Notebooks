---
type: overview
tags: [survey, manipulation, vla, world-action-model, reinforcement-learning, senlanke]
status: complete
updated: 2026-11-09
related:
  - senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - senlanke-weekly-2026-11-02-06-humanoid-quadruped-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-11-02_06.md
original:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-11-02_06.md
summary: "senlanke 具身运控lab 11-02–11-06 Manipulation周报：23 条论文条目，每条对应唯一 arXiv 详情节点（新建 11 / 复用 12），本库逐篇通读新建节点并核对周报说法。"
---

# senlanke 周报索引：11-02–11-06 Manipulation（23 篇）

> **本页定位**：为 [【11.02-11.06前沿论文动态】Manipulation 23篇](https://mp.weixin.qq.com/s/Y21u5W8p8EYTAk91F5VZTQ) 提供 23 个条目到唯一 arXiv 详情节点的索引。

## 一句话定义

senlanke 具身运控lab 11-02–11-06 一周 Manipulation论文周报（23 篇）的分栏索引页，把每条映射到唯一 arXiv 详情节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 世界动作模型 |
| RL | Reinforcement Learning | 强化学习 |
| MPC | Model Predictive Control | 模型预测控制 |
| 3DGS | 3D Gaussian Splatting | 三维高斯溅射 |

## 为什么重要

- 周报只给摘要级描述；本页让每条都落到可追溯的详情节点，节点里有逐篇通读后的核对结论。
- 周报条目里不少是模型驱动或规划类工作，并不都是强化学习；索引页把这类差别写出来，避免只看 ★ 标注就归类。

## 核心原理

索引遵循「周报分栏 → 唯一 arXiv 节点」的映射：同一 arXiv 论文跨「人形 / 四足」与「Manipulation」两篇周报时复用同一节点（HITCH、CRISP 即如此）；新建节点已逐篇通读论文。

周报把 23 篇操作论文分成 10 个栏目，其中 3 篇「人形上半身操作」（HITCH、CRISP、Pixel-UMI）与人形 / 四足周报重叠（HITCH、CRISP 在两份周报里都出现，只建一个节点）。本库逐篇通读新建节点后的观察（本库判断）：

- **世界模型 / WAM 栏：** La-JEPA（冻结 V-JEPA 2.1 特征，沿用 Fast-WAM 的训练推理分离；LIBERO-Plus 官方指令下并非最优）、RoboImagine（对世界模型做奖励式后训练，只验证离线视频预测，训练开销约 27 倍）、WISE（免训练的推理时候选选择，相对最强基线只多 4.0 / 0.9 个百分点）。
- **效率类：** T3R 免训练剪枝视觉 token，实测 VLA 调用时延只降 14%，加速倍数不能按 LLM 前向的数字引用。
- **强化学习栏的新建节点：** DLO 关键点 RL 实证研究（三套 PPO 栈对比，瓶颈在物理仿真）、TactileLab（模拟触觉光流，真机优势小于标准差）。
- **数据与演示：** MoMaSplat 用 3DGS 合成新底盘位姿示范（原位姿下各方法相当，偏移后才拉开差距）；VREgo 用消费级 VR 头显采集（全部仿真评测）。
- **非学习类 / 规划类：** 双臂桌面重排用联合 MCTS，没有训练。
- **证据强度：** 本周新建的操作类节点里没有一篇有大规模真机验证；多数只有仿真或每任务 10–20 次真机试验。

## 工程实践

- 查证某篇论文的技术细节时，以详情节点和其引用的 arXiv 原文为准，不以周报摘要为准。
- 新增周报时按分栏补表，并先按 arXiv ID 去重（同一 ID 只允许一个 canonical 节点）。

## 局限

- 周报只给摘要级描述；新建节点逐篇通读了论文，复用节点沿用既有页面、**没有重新读论文**。
- 周报机构大多写「待核实」，新建节点已按论文首页补全；VREgo 论文首页没有机构，页面如实写明。
- 周报对若干论文的说法与论文不符，已在各详情节点核对表里列出（例如 WISE 只有三个打分分量、MoMaSplat 的物体运动是重放而非物理模拟、VREgo 全部为仿真评测）。
- 上面的归纳是本库判断，只基于新建节点读到的内容。

## 论文索引

### VLA / 通用操作策略（4）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [SLA：通过持续预训练扩展空间语言动作模型](../entities/paper-sla-spatial-language-action.md) | [2611.08373](https://arxiv.org/abs/2611.08373) | 复用 |
| 02 | [OopsieRecover：从语义干预中学习可迁移的 VLA 纠正策略](../entities/paper-oopsierecover.md) | [2611.08462](https://arxiv.org/abs/2611.08462) | 复用 |
| 03 | [通过校准潜在特征探针预测与分类 VLA 执行失败](../entities/paper-vla-tug.md) | [2611.08425](https://arxiv.org/abs/2611.08425) | 复用 |
| 04 | [T3R：面向高效鲁棒 VLA 的免训练两阶段 Token 优化](../entities/paper-t3r-training-free-two-stage-token-refinement.md) | [2611.08413](https://arxiv.org/abs/2611.08413) | 新建 |

### WAM / World Model 与操作预测（5）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [GPWM：通过三维粒子运动建模物理世界](../entities/paper-gpwm.md) | [2611.08338](https://arxiv.org/abs/2611.08338) | 复用 |
| 02 | [Spatial-UMI-Dream：利用度量几何扩展 UMI 操作世界动作模型](../entities/paper-spatial-umi-dream.md) | [2611.07229](https://arxiv.org/abs/2611.07229) | 复用 |
| 03 | [La-JEPA：基于潜在世界建模的紧凑机器人策略](../entities/paper-la-jepa.md) | [2611.08156](https://arxiv.org/abs/2611.08156) | 新建 |
| 04 | [通过语义与视觉反馈学习机器人世界模型](../entities/paper-roboimagine.md) | [2611.07850](https://arxiv.org/abs/2611.07850) | 新建 |
| 05 | [看穿预测幻象：世界动作模型的测试时计算扩展](../entities/paper-wise-wam-test-time-scaling.md) | [2611.03585](https://arxiv.org/abs/2611.03585) | 新建 |

### 在线/离线强化学习与后训练（4）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [GSBR：面向在线强化学习的几何支持集行为正则化](../entities/paper-gsbr.md) | [2611.05295](https://arxiv.org/abs/2611.05295) | 复用 |
| 02 | [通过潜在噪声表示桥接实现高效扩散策略微调](../entities/paper-latent-noise-representation-bridging.md) | [2611.07275](https://arxiv.org/abs/2611.07275) | 复用 |
| 03 | [面向数据高效灵巧操作的潜在强化学习先验](../entities/paper-latent-rl-priors.md) | [2611.08458](https://arxiv.org/abs/2611.08458) | 复用 |
| 04 | [D&A：通过描述与评估解耦实现零样本机器人奖励模型](../entities/paper-dna.md) | [2611.03622](https://arxiv.org/abs/2611.03622) | 复用 |

### 模仿学习、视觉运动策略与生成式动作策略（2）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [GUILD：面向模仿学习的引导式迭代数据采集](../entities/paper-guild.md) | [2611.05368](https://arxiv.org/abs/2611.05368) | 复用 |
| 02 | [MOMASPLAT：面向移动操作的动态新视角示范增强](../entities/paper-momasplat.md) | [2611.08372](https://arxiv.org/abs/2611.08372) | 新建 |

### 双臂操作（1）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [结合可移动堆叠与间隙感知推动转移的双臂桌面物体重排](../entities/paper-dual-arm-movable-stacking-gap-aware-push.md) | [2611.07726](https://arxiv.org/abs/2611.07726) | 新建 |

### 灵巧操作（1）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [TactileLab：利用高效剪切触觉仿真实现灵巧操作仿真到真实迁移](../entities/paper-tactilelab.md) | [2611.07881](https://arxiv.org/abs/2611.07881) | 新建 |

### 柔性/可变形物体操作（1）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [可变形线状物体关键点定位的强化学习实证研究](../entities/paper-dlo-keypoint-rl-empirical-study.md) | [2611.07261](https://arxiv.org/abs/2611.07261) | 新建 |

### 人形机器人上半身操作（3）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [HITCH：通过全身执行接口教会视觉语言模型控制人形机器人](../entities/paper-hitch.md) | [2611.08039](https://arxiv.org/abs/2611.08039) | 新建 |
| 02 | [CRISP：基于子系统交互约束求解的预测式全身协同控制](../entities/paper-crisp-whole-body-nmpc.md) | [2611.07268](https://arxiv.org/abs/2611.07268) | 新建 |
| 03 | [Pixel-UMI：从传感化人类示范学习环境感知全身操作](../entities/paper-pixel-umi.md) | [2611.03647](https://arxiv.org/abs/2611.03647) | 复用 |

### 长程任务、推理与规划（1）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [Pixel：面向通用移动操作学习的软硬件协同设计机器人系统](../entities/paper-pixel.md) | [2611.03648](https://arxiv.org/abs/2611.03648) | 复用 |

### 数据集、预训练、跨本体与硬件平台（1）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [VREgo：使用消费级 VR 头显采集第一视角示范并学习机器人操作](../entities/paper-vrego.md) | [2611.07272](https://arxiv.org/abs/2611.07272) | 新建 |

## 关联页面

- [人形 / 四足 周报索引（同一周）](./senlanke-weekly-2026-11-02-06-humanoid-quadruped-technology-map.md)
- [上一周Manipulation周报索引（10-26–10-30）](./senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)

## 参考来源

- [周报来源归档](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-11-02_06.md)

## 继续阅读

- 各详情节点引用的 arXiv 原文；复用节点沿用既有页面，需要时再逐篇重读。
