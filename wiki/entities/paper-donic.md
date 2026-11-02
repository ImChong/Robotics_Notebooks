---
type: entity
tags: [paper, dexterous-manipulation, bimanual, hand-object-tracking, imitation-learning, reinforcement-learning, ropedia]
status: complete
updated: 2026-11-01
project_id: donic-generalist-dexterous-bimanual-tracking
arxiv: "2610.29997"
project: https://ropedia.github.io/DONIC/
code: https://github.com/Ropedia/DONIC
related:
  - ../concepts/contact-rich-manipulation.md
  - ../tasks/manipulation.md
  - ./paper-dexmimic.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
  - ../../sources/papers/donic_arxiv_2610_29997.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "DONIC jointly trains one bimanual dexterous reference-tracking policy across operation categories and human motion sources, using BiHOI data, object-centered per-hand rewards, and verified demonstrations."
---

# DONIC：通用灵巧双手跟踪控制

**一句话定义：** DONIC 用一套共享策略跟踪不同操作类别与人类动作来源的双手物体交互，并在未见轨迹和留出物体上测试零微调泛化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DONIC | Generalist Dexterous Bimanual Tracking Control from Human Demonstrations | 从人类演示学习双手灵巧跟踪控制 |
| BiHOI | Bimanual Hand-Object Interaction | DONIC 配套的多视角双手手-物交互数据集 |
| PPO | Proximal Policy Optimization | 与行为克隆联合使用的强化学习更新 |
| BC | Behavior Cloning | 将经过验证的成功执行作为示范监督 |
| CCA | Clip-Centered Advantages | 去除每条轨迹的平均优势偏置、保留轨迹内差异的方法 |

## 为什么重要

双手操作常要求两只手分工，接触时序也可能不同。仅追踪手部姿态并不能确保物体按目标移动。DONIC 把物体运动、逐手接触角色和多来源参考轨迹结合起来，尝试让单一策略掌握更多双手协调行为。

## 核心原理

BiHOI 以 32 个校准 RGB 视角采集 1,575 段、覆盖 229 个物体和 14.77 小时的交互。研究者重建双手与物体，再将人手动作重定向到 Wuji 机器人手参考轨迹，与公开数据源共同训练。

策略使用 object-centered、per-hand reward：整个片段都关注物体运动，各只手则在参考要求接触时受相应监督。PPO 更新与 BC 示范监督形成迭代回路，验证成功的共享策略执行会重新加入示范库。CCA 先减去片段平均优势，缓解不同参考难度带来的偏移，同时保留片段内部动作差异。

## 方法栈

`BiHOI + 公开手-物交互参考 → 手/物重建与机器人重定向 → PPO + BC 联合训练 → 验证成功轨迹回流 → 单策略双手跟踪`

## 实验与评测

- 未见轨迹、物体仍在训练集内：成功率 **86.79%**。
- 留出物体：成功率 **57.32%**。
- 相对双手化 DexTrack、Dexplore 基线，两个拆分分别高 **77.36** 和 **45.73** 个百分点。
- 以上是论文所定义的模拟跟踪评测。轨迹拆分和留出物体结果不能证明未见操作类别或任意任务目标上的泛化。

## 与其他工作对比

| 路线 | 策略粒度 | 主要特点 |
|------|----------|----------|
| DONIC | 单一双手共享策略 | 跨操作类别和人类动作源联合训练，包含物体运动与逐手接触监督 |
| [DexMimic](./paper-dexmimic.md) | 通用灵巧手-物体跟踪策略 | 以价值引导的重置分配训练单手轨迹跟踪 |
| DexTrack / Dexplore 双手适配 | 双手参考跟踪基线 | 论文用于同数据池、同模拟器的对比 |

## 工程实践

- 双手操作评估需同时衡量物体运动和每只手的接触角色，单看末端姿态误差不足以判断任务完成。
- 若复用 BiHOI 处理 pipeline，需阅读官方仓库的第三方依赖许可；仓库注明其中包含非商业条款组件。
- 截至 2026-11-01 复核，官方仓库仍只发布 BiHOI 处理 pipeline；DONIC 训练/评测代码、策略权重和 BiHOI 数据集仍列为待发布。处理代码与各第三方依赖的许可条件需分别核对。

## 结论

**DONIC 的主要贡献是让一套双手策略跨多种参考轨迹联合训练，并用物体运动反馈和逐手接触监督保持操作目标。**

1. BiHOI 扩充了持续双手接触的多视角手-物交互参考。
2. 86.79% 与 57.32% 分别对应不同泛化拆分，解释时应保留各自测试定义。
3. 官方训练代码、权重和完整数据仍待发布；目前可见的开源入口主要是数据处理 pipeline。

## 局限与风险

- 已报告的拆分不等同于跨操作类别、跨机器人形态或开放式指令泛化。
- 人手重建与重定向、对象跟踪和模拟器物理都会影响参考质量与控制表现。
- 训练与数据资产未完整公开，当前复现能力有限。

## 关联页面

- [灵巧操作中的接触丰富交互](../concepts/contact-rich-manipulation.md)
- [Manipulation](../tasks/manipulation.md)
- [DexMimic](./paper-dexmimic.md)

## 参考来源

- [DONIC 论文归档](../../sources/papers/donic_arxiv_2610_29997.md)
- [DONIC 官方项目页](https://ropedia.github.io/DONIC/)
- [DONIC 官方代码仓库](https://github.com/Ropedia/DONIC)
- [arXiv:2610.29997](https://arxiv.org/abs/2610.29997)

## 推荐继续阅读

- [DONIC 项目页](https://ropedia.github.io/DONIC/) — BiHOI 数据和双手评测演示
- [DONIC 代码仓库](https://github.com/Ropedia/DONIC) — 当前开放的 pipeline 与复现状态
- [DexMimic](./paper-dexmimic.md) — 灵巧手-物体通用跟踪对照路线
