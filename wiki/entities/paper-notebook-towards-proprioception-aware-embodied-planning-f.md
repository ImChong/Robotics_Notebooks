---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2510.07882"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md
summary: "近年多模态大模型（MLLM）能做高层规划，让机器人遵从复杂人类指令。但在涉及双臂人形的长时程任务上效果仍有限——原因是仿真平台不足与当前 MLLM 的具身感知（embodiment awareness）欠缺。本文用一个新的双臂人形模拟器 DualTHOR（带连续过渡与意外机制），并提出 Proprio-MLLM：一个融合本体感受信息、基于运动的位置嵌入、跨空间编码器（cross-spatial encoder）的增强模型，以提升具身感知。在 DualTHOR 环境中，Proprio-MLLM 的规划性能平均提升 19.75%（相比现有 MLLM）。"
---

# Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots

**Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2510.07882> |
| 作者 | Boyu Li、Siyuan He、Hang Xu、Haoqi Yuan、Börje F. Karlsson、Zongqing Lu 等 |
| 发表 | 2025 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md](../../sources/papers/humanoid_pnb_towards-proprioception-aware-embodied-planning-f.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2510.07882>

## 推荐继续阅读

- [机器人论文阅读笔记：Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots/Towards_Proprioception-Aware_Embodied_Planning_for_Dual-Arm_Humanoid_Robots.html)
