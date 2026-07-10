---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.07407"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_unified-humanoid-fall-safety-policy-from-a-few-d.md
summary: "跌倒是人形移动的固有风险。维持平衡是控制与学习的首要安全焦点，但没有方法能完全杜绝失衡。当失稳真的发生时，已有工作只处理孤立的一段：要么防摔、要么编排受控下降、要么摔后起身——因此人形缺乏整合的策略来在真实跌倒不按剧本时做减损与即时恢复。本文要超越\"保持平衡\"，让整个\"跌倒-恢复\"过程都安全且自主：能防则防、不可避免则减损、摔倒则起身。通过融合稀疏人类示范 + 强化学习 + 一个自适应的扩散式「安全反应记忆」，学出自适应全身行为，把防摔、减损、快速恢复统一进一个策略。在仿真与真机 Unitree G1 上验证，具备稳健 sim-to-real、降低跌落冲击、并在多扰动场景下持续快速恢复。"
---

# Unified Humanoid Fall-Safety Policy from a Few Demonstrations

**Unified Humanoid Fall-Safety Policy from a Few Demonstrations** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

跌倒是人形移动的固有风险。维持平衡是控制与学习的首要安全焦点，但没有方法能完全杜绝失衡。当失稳真的发生时，已有工作只处理孤立的一段：要么防摔、要么编排受控下降、要么摔后起身——因此人形缺乏整合的策略来在真实跌倒不按剧本时做减损与即时恢复。本文要超越"保持平衡"，让整个"跌倒-恢复"过程都安全且自主：能防则防、不可避免则减损、摔倒则起身。通过融合稀疏人类示范 + 强化学习 + 一个自适应的扩散式「安全反应记忆」，学出自适应全身行为，把防摔、减损、快速恢复统一进一个策略。在仿真与真机 Unitree G1 上验证，具备稳健 sim-to-real、降低跌落冲击、并在多扰动场景下持续快速恢复。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Fall-Safety | 跌落安全，覆盖防摔/减损/起身全过程 |
| Few Demonstrations | 少量示范，稀疏人类演示 |
| Diffusion Memory | 扩散式记忆，存取安全反应模式 |
| Impact Mitigation | 冲击减损，降低落地受力 |
| Recovery | 恢复，摔后起身 |
| Unified Policy | 统一策略，一策略覆盖多阶段 |

## 为什么重要

- **"全过程"视角胜过"单点"**：把防摔/减损/起身一体化，更贴合真实不按剧本的跌倒；
- **扩散式记忆做"安全反应库"**是新颖思路，可按情境检索合适反应；
- **少示范 + RL**降低对大规模安全数据的依赖；
- **与 SafeFall、自保护跌落、Robot Crash Course、VIGOR 共同构成人形跌落安全研究簇**，本文强调"统一"。

## 解决什么问题

跌落安全此前被**割裂**处理： - 防摔 / 受控下降 / 起身**各管一段**； - 真实跌倒常**不按剧本**，缺乏**整合策略**做减损 + 即时恢复。

论文要：把**防摔 + 减损 + 快速恢复**统一进**一个自适应策略**，并能从**少量示范**学到。

## 核心机制

1. **统一跌落安全策略**：把防摔、减损、快速恢复整合进一个自适应策略；
2. **三融合学习**：稀疏示范 + RL + 自适应扩散安全记忆；
3. **少示范即可**：从少量演示学到鲁棒安全行为；
4. **真机验证**：Unitree G1 稳健 sim-to-real、降冲击、多扰动快速恢复。

方法拆解（深读笔记小节）：三融合：稀疏示范 + RL + 扩散记忆；一个策略统一三阶段；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations.html> |
| arXiv | <https://arxiv.org/abs/2511.07407> |
| 作者 | Zhengjie Xu、Ye Li、Kwan-yee Lin、Stella X. Yu |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_unified-humanoid-fall-safety-policy-from-a-few-d.md](../../sources/papers/humanoid_pnb_unified-humanoid-fall-safety-policy-from-a-few-d.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations.html>
- 论文：<https://arxiv.org/abs/2511.07407>

## 推荐继续阅读

- [机器人论文阅读笔记：Unified Humanoid Fall-Safety Policy from a Few Demonstrations](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations/Unified_Humanoid_Fall-Safety_Policy_from_a_Few_Demonstrations.html)
