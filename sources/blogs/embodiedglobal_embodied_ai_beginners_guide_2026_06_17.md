# What is Embodied AI? A Complete Beginner's Guide 2026｜2026 具身智能完整入门指南

> 来源归档（blog / Embodied Global 编辑部综述；**非论文、非实验报告、非官方产品资料**）

- **标题：** What is Embodied AI? A Complete Beginner's Guide 2026
- **中文标题：** 2026 具身智能完整入门指南：定义、技术与落地场景
- **类型：** blog / beginner guide
- **作者：** Embodied Global Team
- **原文发表：** 2026-06-17
- **原始链接：** https://embodiedglobal.com/en/article/what-is-embodied-ai-beginners-guide-2026
- **入库日期：** 2026-11-19
- **核对范围：** 已核对全文的 Introduction、§1 Definition/Traditional AI/2026、§2 Perception–Cognition–Action、§3 Sim2Real/VLA/RL/IL/World Models、§4 Applications、§5 Companies、§6 Market/Trends/Challenges、Conclusion。网站直连在部分抓取接口返回 403，检索结果可提供同 URL 的完整正文；未核对文中每项产业统计的独立原始数据来源。
- **通读：** 已完成
- **性质：** 2026 年行业观察与零基础教程，不包含可运行的代码入口、基准测试或可验证的论文实验。
- **映射：** [具身智能 2026 入门导读](../../wiki/overview/embodied-ai-beginners-guide-2026.md)

## 全文结构与核心摘录（意译归纳，非原文复制）

1. **定义与范围（§1）：** 文章将 Embodied AI / Physical AI / Embodied Intelligence 放在同一叙述下，与主要处理数字信息的模型作对比，强调机器人通过物理身体、传感器、动作与环境持续耦合。这里是作者采用的入门口径，三者在具体技术语境不总是严格同义。
2. **连续反馈环（§2）：** 多模态传感（相机、深度/激光、力与触觉、IMU 等）→ 环境/任务理解与规划 → 运动、操作、力控 → 感知物理反馈，实时重复。
3. **技术栈（§3）：** Sim2Real（仿真与域随机化）、VLA（视觉+语言条件的动作模型）、RL（奖励驱动试错）、IL（示范驱动学习）、World Models（预测未来状态以辅助规划）。这是不同模块和训练范式的清单，**并非**要求所有系统同时采用所有技术。
4. **应用（§4）：** 制造、仓储物流、家政、医疗、零售；文中部分产品与部署案例仍需分别追溯一手资料。
5. **产业版图（§5）：** Figure、Boston Dynamics、Tesla、1X、Agility、Google DeepMind、NVIDIA，及中国厂商；**只是作者选取的示例，不是权威排名或完整厂商清单**。
6. **趋势与障碍（§6）：** 成本、规模化、VLA 普及与产业竞争并行；泛化、安全、数据、成本、法规、长期可靠性仍是挑战。

## 有必要区分的三类表述

| 表述 | 证据地位 | 本库处理 |
|---|---|---|
| 感知—认知—行动与反馈闭环 | 文章明确给出的解释框架 | 可作为科普结构，落地需区分高/低频控制 |
| VLA、Sim2Real、RL、IL、World Model | 文章列出的技术类别 | 与本库方法页互链，不把列表当作统一架构 |
| 市场预测、企业估值/出货量/机器人价格、已解决泛化 | 文章转述或预测，未见本文提供完整可复核原始口径 | **不记作独立验证事实**；不直接加入 wiki 量化结论 |

## 代码 / 数据开放核查

本条是编辑部科普文章，不是具体研究项目；文章未提供可复现的官方训练入口或配套数据集。因此**源码运行时序图不适用**，相关产品是否开源应分别查其官方项目资料。

## 对 wiki 的映射

- [入门阅读节点](../../wiki/overview/embodied-ai-beginners-guide-2026.md)
- [VLA 方法](../../wiki/methods/vla.md)
- [具身 Infra](../../wiki/overview/embodied-infra-2026-panorama.md)
- [运动控制 → 物理智能全栈路线](../../roadmap/motion-control.md)
