---
type: entity
project_id: imitation-learning-contact-rich-tasks-survey
doi: "10.1177/02783649261417694"
venue: "The International Journal of Robotics Research (OnlineFirst, 2026)"
tags: [paper, survey, imitation-learning, contact-rich, manipulation, robot-learning]
status: complete
updated: 2026-10-25
related:
  - ../methods/imitation-learning.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
sources:
  - ../../sources/papers/contact_rich_imitation_learning_survey_doi_10_1177_02783649261417694.md
summary: "这篇 IJRR 综述从示教采集、感知模态、学习方法与应用领域系统梳理接触丰富机器人任务中的模仿学习，并指出数据、可观测性、泛化和 Sim2Real 等挑战。"
---

# A survey on imitation learning for contact-rich tasks in robotics（接触丰富任务中的模仿学习综述）

**一句话定义：** 本文系统组织接触丰富机器人任务的模仿学习研究，沿着示教数据如何采集、算法如何学习、方法如何进入应用来梳理技术路线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 从专家演示学习接触操作技能与策略 |
| DMP | Dynamic Movement Primitive | 用结构化动力系统表示并复现示教轨迹 |
| IRL | Inverse Reinforcement Learning | 从演示推断奖励，再用于策略优化 |
| RL | Reinforcement Learning | 通过环境交互和回报优化动作策略 |
| Sim2Real | Simulation to Reality | 将仿真学习结果迁移到真实机器人 |

## 为什么重要

接触丰富任务包括插入、装配、擦拭、工具使用和人机辅助等。接触力、摩擦和接触状态会随微小位姿误差而改变，纯粹依赖预设轨迹或视觉观察容易在真实执行中失效。综述将模仿学习放在这一现实约束下讨论：专家示教可携带难以显式建模的力调节与接触经验，而有效学习又依赖能否采到足够丰富、可观测的演示数据。

## 方法脉络

论文按实现接触丰富 IL 的自然流程展开：先界定接触任务与挑战，再讨论演示数据采集，随后比较学习范式，最后梳理应用案例。其覆盖范围可读作一条从数据到技能的路线：

1. **定义接触问题：** 关注非线性接触动力学、状态可观测性和学习稳定性。
2. **采集演示：** 结合关节/末端位置、视觉、力/力矩和触觉；示教方式包括手把手引导、遥操作、VR 以及观察式采集，也讨论合成数据。
3. **选择学习方法：** 从行为克隆等监督学习基线，延伸到 DMP、VAE/扩散生成方法、基础模型、IRL、多模态 IL、离线 RL 等。
4. **按应用条件评估：** 对照工业、家庭与医疗任务所需的接触传感、数据来源和执行约束。

## 工程实践

把这篇综述用于项目选型时，可按以下顺序检查：

1. **先定义接触信号：** 明确任务是否需要位置、力/力矩、触觉或视觉，以及接触切换和滑移能否被观测。
2. **再选示教方式：** 直接手把手或遥操作可获得机器人视角下的数据；观察式示教更易扩展，但常缺少动作和作用力标签。
3. **对齐数据与算法：** 轨迹映射清楚时可用行为克隆；动作具有多模态分布时可评估生成式策略；任务/奖励可推断或可交互时，再考虑 IRL、RL 或离线方法。
4. **分开评估泛化维度：** 报告物体、接触几何、传感器、视角和环境变化下的表现，并明确仿真数据与真机数据比例。

本文是综述，没有提供可运行算法实现、训练入口或统一的新基准；以上是按其分类框架整理的工程检查项，不是论文的实验结果。

## 评测与证据

综述通过文献组织和计量分析描述研究主题的演变，重点覆盖 2019–2025 年多模态学习、基础模型和生成方法的发展。它没有提出新的机器人系统，也没有自己的统一任务成功率或算法对照实验。因此，文章中的方法目录适合定位与查找原始工作，不能当作跨方法的统一性能排名。

## 与其他工作对比

| 资料 | 主要范围 | 适合用途 |
|------|----------|----------|
| 本文 | 聚焦接触丰富任务中的 IL，贯穿数据采集、学习范式和工业/家庭/医疗应用 | 从任务约束反推示教、模态和算法选择 |
| [Contact-Rich Manipulation 概念页](../concepts/contact-rich-manipulation.md) | 接触动力学、力控、阻抗和全身协调 | 建立任务与执行层的工程基础 |
| [Tactile-Based Robotic Manipulation 综述](./paper-tactile-based-robotic-manipulation-survey.md) | 从力学驱动与数据驱动角度组织触觉操作 | 补充触觉策略与力学方法分类 |

本文的边界是 IL 方法及其接触任务应用；接触控制和触觉硬件本身仍需结合专门的控制与传感综述理解。

## 结论

**一句话总判：接触丰富任务中的 IL 成败，往往先由演示能否捕获动作以外的接触信息决定，再由算法和数据覆盖共同影响泛化。**

1. 采集演示时同时考虑位姿与作用力/触觉；纯视觉在遮挡和接触瞬态下存在观测盲区。
2. 手把手、遥操作、VR 和观察式示教提供不同成本与信号完整度，选型应与机器人接口和任务接触阶段匹配。
3. 行为克隆、生成式方法、基础模型、IRL、多模态 IL 和离线 RL 是不同学习范式，综述没有给出统一 benchmark 下的直接胜负。
4. 将仿真到真实、传感器差异、物体变化和视角变化纳入评估，避免只报告同分布成功率。

## 局限与风险

- 接触动力学高度非线性，模型误差会扩大仿真到真实的落差；真实机器人反复采样还受安全和硬件磨损限制。
- 力/触觉数据与可穿戴采集技术仍有限；图像不直接给出接触力，接触区域也常被遮挡。
- 第三人称人类视频到第一人称机器人动作存在视角、外观、坐标和动作信息差，无法简单把视觉轨迹当作可执行示教。
- 综述覆盖的研究进展主要截至 2025 年，具体算法和数据开放状态需回到各原始论文/项目页核实。
- SAGE 文章页未列配套项目主页、代码仓库或数据集；当前资料适合做研究导航，不提供直接复现入口。

## 关联页面

- [Imitation Learning](../methods/imitation-learning.md) — IL 的算法谱系、数据管线和机器人应用。
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 接触建模、力控和任务分段。
- [Tactile Sensing](../concepts/tactile-sensing.md) — 触觉传感与接触状态观测。
- [具身数据采集四层术语地图](../concepts/embodied-data-collection-four-layers-taxonomy.md) — 示教来源、设备、模态与 episode 的层级关系。

## 参考来源

- [SAGE 原文：A survey on imitation learning for contact-rich tasks in robotics](../../sources/papers/contact_rich_imitation_learning_survey_doi_10_1177_02783649261417694.md)
- [期刊 DOI](https://doi.org/10.1177/02783649261417694)

## 推荐继续阅读

- [Imitation Learning 论文导航](../../references/papers/imitation-learning.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Tactile-Based Robotic Manipulation 综述](./paper-tactile-based-robotic-manipulation-survey.md)