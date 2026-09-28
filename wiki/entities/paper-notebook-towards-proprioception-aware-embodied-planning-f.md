---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, bimanual, embodied-planning, mllm, simulation-benchmark, pku, beingbeyond, baai, casia, ucas]
status: complete
updated: 2026-09-28
arxiv: "2510.07882"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./ai2-thor.md
  - ../tasks/bimanual-manipulation.md
  - ../concepts/humanoid-policy-observation-inputs.md
  - ./paper-notebook-endowing-gpt-4-with-a-humanoid-body-building-the.md
  - ./paper-notebook-hierarchical-vision-language-planning-for-multi.md
sources:
  - ../../sources/papers/humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md
summary: "近年多模态大模型（MLLM）能做高层规划，让机器人遵从复杂人类指令。但在涉及双臂人形的长时程任务上效果仍有限——原因是仿真平台不足与当前 MLLM 的具身感知（embodiment awareness）欠缺。本文用一个新的双臂人形模拟器 DualTHOR（带连续过渡与意外机制），并提出 Proprio-MLLM：一个融合本体感受信息、基于运动的位置嵌入、跨空间编码器（cross-spatial encoder）的增强模型，以提升具身感知。在 DualTHOR 环境中，Proprio-MLLM 的规划性能平均提升 19.75%（相比现有 MLLM）。"
---

# Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots

**Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

近年多模态大模型（MLLM）能做高层规划，让机器人遵从复杂人类指令。但在涉及双臂人形的长时程任务上效果仍有限——原因是仿真平台不足与当前 MLLM 的具身感知（embodiment awareness）欠缺。本文用一个新的双臂人形模拟器 DualTHOR（带连续过渡与意外机制），并提出 Proprio-MLLM：一个融合本体感受信息、基于运动的位置嵌入、跨空间编码器（cross-spatial encoder）的增强模型，以提升具身感知。在 DualTHOR 环境中，Proprio-MLLM 的规划性能平均提升 19.75%（相比现有 MLLM）。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| MLLM | Multimodal Large Language Model |
| Proprio-MLLM | 本文的本体感受感知 MLLM |
| Embodiment Awareness | 具身感知，模型对自身身体状态的理解 |
| DualTHOR | 双臂人形模拟器 |
| Position Embedding | 位置嵌入（基于运动） |
| Cross-Spatial Encoder | 跨空间编码器 |

## 为什么重要

- **高层规划需要"具身感知"**：纯语义 MLLM 不够，要注入本体状态；
- **仿真平台是 MLLM 规划研究的前提**（与 DualTHOR 平台论文同源）；
- **本体感受 + 跨空间编码**是把语言规划接到物理身体的桥；
- 与 BiBo（现成 VLM 控人形）形成"增强 MLLM vs 借现成 VLM"的对照。

## 解决什么问题

MLLM 做双臂人形长时程规划受限： - **仿真平台不足**（缺连续过渡/意外）； - MLLM **缺具身感知**，不"知道"自己身体状态，规划脱离物理。

论文要：① 更好的双臂人形仿真（DualTHOR）；② 让 MLLM **感知本体状态**以改进规划。

## 核心机制

1. **DualTHOR 双臂人形模拟器**：连续过渡 + 意外机制；
2. **Proprio-MLLM**：注入本体感受、运动位置嵌入、跨空间编码器；
3. **增强具身感知**：让高层规划"知道身体状态"；
4. **+19.75% 规划性能**：相比现有 MLLM。

方法拆解（深读笔记小节）：DualTHOR 双臂人形模拟器；Proprio-MLLM：注入本体感受；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2510.07882> |
| 源码 | **未开源**：论文仅给出匿名仓库链接 anonymous.4open.science/r/DualTHOR-5F3B（审稿用），截至 2026-09-28 未见正式 GitHub 仓库 |
| 作者 | Boyu Li、Siyuan He、Hang Xu、Haoqi Yuan、Börje F. Karlsson、Zongqing Lu 等 |
| 发表 | 2025 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：DualTHOR 中 359 个任务、10 个房间、68 种物体，分三类：双臂必需、双臂可选、单臂；每个任务在 X1 与 H1 两种人形上各测 50 次。所有基线温度 0、最多 2048 token、图像 500×500、高层规划最多 50 步。Proprio-MLLM 基于 Qwen2.5-VL，加入本体感受、基于运动的位置嵌入（MPE）与跨空间编码器（CSE，融合 CUT3R 单帧 3D 点图特征）。

| 方法 | 双臂必需 X1 / H1 | 双臂可选 X1 / H1 | 单臂 X1 / H1 |
|------|------|------|------|
| GPT-4o | 23.31 / 27.07 | 39.76 / 40.96 | 51.67 / 56.67 |
| Qwen2.5-VL-7B | 18.05 / 17.29 | 21.08 / 19.28 | 23.33 / 25.00 |
| LLM-Planner | 28.57 / 31.58 | 43.37 / 45.78 | 55.00 / 56.67 |
| DAG-Plan | 36.09 / 41.53 | 51.20 / 52.41 | 55.00 / 58.33 |
| **Proprio-MLLM** | **59.39 / 63.16** | **71.69 / 70.48** | **73.33 / 75.00** |

（成功率 %。）

- 相对 DAG-Plan 平均 **+19.75%**，X1 双臂必需任务最高 +23.30%；各方法在双臂任务上都明显弱于单臂，作者认为现有模拟器缺少双臂人形规划数据。
- **意外与重规划**（H1 双臂必需，低层技能成功率 100% / 50% / 20% 分为易 / 中 / 难）：Proprio-MLLM 63.16% / 51.63% / 36.51%，去掉反思提示 56.39% / 38.67% / 26.17%；DAG-Plan 41.53% / 20.33% / 15.53%。
- **消融**（H1）：完整 63.16%；去掉 CSE 45.17%（导航失败 1225 → 2205 次）；去掉 MPE 52.33%（身体调整与逻辑失败增加）；两者都去掉 41.53%；Qwen2.5-VL 原模型 17.29%。
- 全部结果在仿真中得到，论文未报告真机实验。

## 与其他工作对比

| 方法 | 规划输入 | 与 Proprio-MLLM 的差异 |
|------|------|------|
| GPT-4o / Gemini 等通用 MLLM | 图像 + 语言 | 不知道机器人身体状态，双臂必需任务仅 23–27% |
| DAG-Plan | 以图结构提示表示技能 | 能改善选哪只手，但无法利用本体感受减少导航与身体调整失败 |
| [BiBo](./paper-notebook-endowing-gpt-4-with-a-humanoid-body-building-the.md) | 现成 VLM 驱动人形 | 不改模型；Proprio-MLLM 改造 MLLM 并配套仿真平台 |
| [AI2-THOR](./ai2-thor.md) 类离散状态规划基准 | 离散状态跳转 | DualTHOR 加入连续过渡与意外机制，可评测重规划 |

## 结论

**这篇工作的判断是：双臂人形长时程规划做不好，瓶颈不在语言能力，而在「没有够用的仿真平台」和「MLLM 不知道自己有身体」——所以它同时补了环境（DualTHOR）和模型（Proprio-MLLM）两块。**

- 真正起作用的是把 **本体感受状态显式注入 MLLM**：基于运动的位置嵌入 + 跨空间编码器，让高层规划与身体状态对齐，而不是停留在纯语义层面。
- 关键量化是在 DualTHOR 上 **规划性能平均 +19.75%**（H1 双臂必需任务 41.53% → 63.16%）；这是 **仿真内相对现有方法的提升**，论文没有真机结果，不应外推为实机可用性。
- DualTHOR 的差异化在于 **连续过渡与意外机制**——这意味着评测本身就把"计划执行会失败"计入，比离散状态跳转的规划基准更接近物理现实。
- 适用边界：面向 **双臂人形的高层长时程规划**，处理的是"下一步做什么"，不是底层全身控制；能力上限仍受底层执行器与技能库限制。
- 与 BiBo 形成一组对照：后者直接借现成 VLM 驱动人形，本页则选择 **改造 MLLM 本身以获得具身感知**，代价是需要配套的仿真平台与训练管线。

## 局限与风险

- **只在仿真中验证**：没有真机结果，不能外推为实机可用性。
- **只做高层规划**：依赖底层技能库执行，难任务（低层技能成功率 20%）下即便重规划也只有 36.51%。
- **平台仍在完善**：资产生成工具、更多机器人、可控失败模式、多房间环境与多智能体评估都列为后续工作。
- **开源边界**：只有匿名审稿链接，未见正式仓库；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 仿真底座 AI2-THOR：[ai2-thor](./ai2-thor.md)
- 任务语境：双臂操作：[bimanual-manipulation](../tasks/bimanual-manipulation.md)
- 本体感受作为策略输入：[humanoid-policy-observation-inputs](../concepts/humanoid-policy-observation-inputs.md)
- BiBo：现成 VLM 驱动人形的对照：[paper-notebook-endowing-gpt-4-with-a-humanoid-body-building-the](./paper-notebook-endowing-gpt-4-with-a-humanoid-body-building-the.md)
- 人形多步操作的层级视觉–语言规划：[paper-notebook-hierarchical-vision-language-planning-for-multi](./paper-notebook-hierarchical-vision-language-planning-for-multi.md)

## 参考来源

- [humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md](../../sources/papers/humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2510.07882>
- 论文正文（Table II–IV、消融）：<https://arxiv.org/html/2510.07882>

## 推荐继续阅读

- [机器人论文阅读笔记：Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html)
