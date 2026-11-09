---
type: overview
tags: [overview, survey, vla, self-improvement, contact-manipulation, benchmark, technology-map]
status: complete
updated: 2026-11-09
related:
  - embodied-station-2026-11-06-20-papers-technology-map.md
  - embodied-station-2026-11-05-43-papers-technology-map.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/wechat_embodied_station_29_papers_sla_flexrsi_mr0_2026-11-09.md
original:
  - ../../sources/blogs/wechat_embodied_station_29_papers_sla_flexrsi_mr0_2026-11-09.md
summary: "具身智能小站 2026-11-09 的 29 篇论文阅读坐标：空间表征与 VLA、智能体自我改进与评测、接触操作与触觉、人形协作与仿真工具；每条指向唯一详情节点（新建 19 / 复用 10），本库逐篇通读新建节点并标出小站说法与论文的出入。"
---

# 空间表征、自我改进评测与接触操作：29 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 29 篇](https://mp.weixin.qq.com/s/WohVGx_d-Q6LkBmaEZ2-uQ)（2026-11-09）提供阅读坐标，并把 29 条论文逐条指向站内唯一详情节点。

## 一句话定义

具身智能小站 11-09 的 29 篇论文更新（1 篇深读、2 篇跟进、26 篇扫读）的阅读坐标页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| RL | Reinforcement Learning | 强化学习 |
| RSI | Recursive Self-Improvement | 递归自我改进 |
| MPC | Model Predictive Control | 模型预测控制 |
| OOD | Out-of-Distribution | 分布外 |

## 为什么重要

- 小站每条只有一句话判断；本页让每条落到可追溯的详情节点，节点里有逐篇通读论文后的核对结论。
- **29 篇里真正属于「机器人强化学习」的只是少数**：新建节点读过的有 Q-OIL（RLPD 加人工接管）、Semantics-to-Contact（残差 SAC 加人在回路）、TangramBench（MAPPO）、ForPC（RL 残差策略），环境与训练器边界的静默故障研究则是检测 RL 训练里的数据故障；其余多是评测、仿真器、经典规划、控制框架、模仿学习或可解释性工作。归类时不能只看「具身」二字。

## 核心原理

按小站分栏与本库新建节点的读后观察（以下归纳是本库判断）：

- **空间表征与 VLA 泛化：** SLA（深读，复用既有节点）；形状教师对齐是纯模仿学习、训练期需要物体掩码与深度；VLA 目标表示研究只覆盖一个 OpenVLA 模型、一个仿真场景。
- **自我改进与评测：** Flex-RSI（24 任务、18 模型，改写的是智能体的提示与流程，不训练策略）、RSIJudge（暴露判据会混入适配验证器的分数）、RoboARCade、美团 MR0（纠错用的是模仿学习，不是 RL 后训练）、环境与训练器边界的静默故障。
- **接触操作与触觉：** TactileLab、Semantics-to-Contact（视觉语言模型只管粗定位，接触靠真机 RL）、分形阻抗统一框架（不含学习）、Lodos 仿真器（速度优势限于所测网格场景）、ForPC（定量全在仿真）、HOC2R（真机每方法仅 6 条序列）。
- **人形协作与工具：** PAVE（价值函数一次性单向交换，不是联合训练 MARL）、TangramBench、H-TRIP（经典算法）、SCALE（通信调度，不训练）、dynamicalnodes、TurtleBot3 的 Webots 库（6 名学生、无对照）、康复轨迹处方（只有健康者数据）。

## 工程实践

- 查证某篇论文的细节时，以详情节点和其引用的 arXiv 原文为准，不以小站一句话判断为准。
- 新增小站条目时先按 arXiv ID 去重（同一 ID 只允许一个 canonical 节点）。

## 局限

- 小站只给一句话判断；新建的 19 个节点逐篇通读了论文，复用的 10 个节点沿用既有页面、**没有重新读论文**。
- 小站对若干论文的说法与论文不符，已在各详情节点的核对表里逐条列出（例如 Flex-RSI 主结果只覆盖 20 个任务、RSIJudge 的迁移是「不再出现」而非「缩小」、ForPC 的「推 30 kg」只在真机演示里）。
- 上面的归类是本库判断，只基于新建节点读到的内容。

## 论文索引

| # | 档 | 论文节点 | arXiv | 状态 |
|---|----|----------|-------|------|
| 01 | 深读 | [SLA：把空间理解前移到预训练](../entities/paper-sla-spatial-language-action.md) | [2611.08373](https://arxiv.org/abs/2611.08373) | 复用 |
| 02 | 跟进 | [Flex-RSI：让智能体改进自己的工作流程](../entities/paper-flex-rsi.md) | [2611.08946](https://arxiv.org/abs/2611.08946) | 新建 |
| 03 | 跟进 | [美团MR0：把纠错接进桌面操作](../entities/paper-meituan-robotics-0.md) | [2611.08577](https://arxiv.org/abs/2611.08577) | 新建 |
| 04 | 扫读 | [Lodos：加速精细网格接触仿真](../entities/paper-lodos.md) | [2611.08872](https://arxiv.org/abs/2611.08872) | 新建 |
| 05 | 扫读 | [TactileLab：用触觉流学习接触变化](../entities/paper-tactilelab.md) | [2611.07881](https://arxiv.org/abs/2611.07881) | 复用 |
| 06 | 扫读 | [RSIJudge：高分来自能力，还是评分规则](../entities/paper-rsijudge-criteria-exposure.md) | [2611.08318](https://arxiv.org/abs/2611.08318) | 新建 |
| 07 | 扫读 | [RoboARCade：检验会推理的机器人能否做准](../entities/paper-roboarcade.md) | [2611.08209](https://arxiv.org/abs/2611.08209) | 复用 |
| 08 | 扫读 | [VFlash：把视觉轨迹一次成组解码](../entities/paper-vflash.md) | [2611.08449](https://arxiv.org/abs/2611.08449) | 复用 |
| 09 | 扫读 | [PAVE：给下一台机器人留下更好的现场](../entities/paper-pave.md) | [2611.08075](https://arxiv.org/abs/2611.08075) | 新建 |
| 10 | 扫读 | [RoboImagine：同时校正未来画面与任务进展](../entities/paper-roboimagine.md) | [2611.07850](https://arxiv.org/abs/2611.07850) | 复用 |
| 11 | 扫读 | [训练曲线正常，也可能传错了数据](../entities/paper-silent-faults-env-trainer-boundary.md) | [2611.08144](https://arxiv.org/abs/2611.08144) | 新建 |
| 12 | 扫读 | [ForPC：让人形机器人稳稳推车](../entities/paper-forpc-cart-pushing.md) | [2611.08664](https://arxiv.org/abs/2611.08664) | 新建 |
| 13 | 扫读 | [S2C：先找到目标，再靠力反馈做精细接触](../entities/paper-semantics-to-contact.md) | [2611.07532](https://arxiv.org/abs/2611.07532) | 新建 |
| 14 | 扫读 | [统一FIC框架：兼顾跟踪与接触柔顺](../entities/paper-fractal-impedance-control-framework.md) | [2611.08009](https://arxiv.org/abs/2611.08009) | 新建 |
| 15 | 扫读 | [H-TRIP：地图扩展后，批量距离也及时更新](../entities/paper-htrip-distance-index.md) | [2611.08139](https://arxiv.org/abs/2611.08139) | 新建 |
| 16 | 扫读 | [Q-OIL：把人工接管变成探索线索](../entities/paper-q-oil.md) | [2611.08576](https://arxiv.org/abs/2611.08576) | 新建 |
| 17 | 扫读 | [OopsieRecover：让少量纠错跨场景复用](../entities/paper-oopsierecover.md) | [2611.08462](https://arxiv.org/abs/2611.08462) | 复用 |
| 18 | 扫读 | [GPWM：用三维粒子预测物理变化](../entities/paper-gpwm.md) | [2611.08338](https://arxiv.org/abs/2611.08338) | 复用 |
| 19 | 扫读 | [说“不要做”，VLA听得懂吗](../entities/paper-vla-goal-representation-editing.md) | [2611.08895](https://arxiv.org/abs/2611.08895) | 新建 |
| 20 | 扫读 | [Social Humanoid：让全身动作回应人](../entities/paper-social-humanoid.md) | [2611.08407](https://arxiv.org/abs/2611.08407) | 复用 |
| 21 | 扫读 | [MOMASPLAT：为移动操作补齐新视角示范](../entities/paper-momasplat.md) | [2611.08372](https://arxiv.org/abs/2611.08372) | 复用 |
| 22 | 扫读 | [HOC2R：从人类视频提取接触轨迹](../entities/paper-hoc2r.md) | [2611.08920](https://arxiv.org/abs/2611.08920) | 新建 |
| 23 | 扫读 | [SCALE：带宽有限时，先传有用信息](../entities/paper-scale-realtime-latent-communication.md) | [2611.08390](https://arxiv.org/abs/2611.08390) | 新建 |
| 24 | 扫读 | [HITCH：给视觉语言模型一套全身操作接口](../entities/paper-hitch.md) | [2611.08039](https://arxiv.org/abs/2611.08039) | 复用 |
| 25 | 扫读 | [TangramBench：先协调零件，再执行装配](../entities/paper-tangrambench-goma.md) | [2611.08814](https://arxiv.org/abs/2611.08814) | 新建 |
| 26 | 扫读 | [dynamicalnodes：从控制框图生成ROS2节点](../entities/paper-dynamicalnodes.md) | [2611.07565](https://arxiv.org/abs/2611.07565) | 新建 |
| 27 | 扫读 | [形状教师：训练时学几何，部署时只看图像](../entities/paper-shape-teacher-alignment.md) | [2611.08275](https://arxiv.org/abs/2611.08275) | 新建 |
| 28 | 扫读 | [康复机器人：从目标肌肉激活反推轨迹](../entities/paper-transformer-trajectory-prescription-semg-rehab.md) | [2611.08295](https://arxiv.org/abs/2611.08295) | 新建 |
| 29 | 扫读 | [TurtleBot3：把仿真控制器带进课堂真机](../entities/paper-turtlebot3-webots-noros-library.md) | [2611.08142](https://arxiv.org/abs/2611.08142) | 新建 |

## 关联页面

- [上一期（11-06）20 篇阅读坐标](./embodied-station-2026-11-06-20-papers-technology-map.md)
- [前一期（11-05）43 篇阅读坐标](./embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [小站 11-09 来源归档](../../sources/blogs/wechat_embodied_station_29_papers_sla_flexrsi_mr0_2026-11-09.md)

## 继续阅读

- 各详情节点引用的 arXiv 原文；复用节点需要时再逐篇重读。
