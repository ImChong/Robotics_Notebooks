---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2512.06571"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-agile-striker-skills-for-humanoid-socce.md
summary: "学又快又稳的踢球技能是人形足球机器人的核心能力，但很难：需要快速摆腿、单脚支撑下的姿态稳定，还要在含噪感知与外部扰动（如对手）下鲁棒。本文用四阶段训练流水线：① 长距离追球（教师，用真值）；② 定向踢球（教师，用真值）；③ 教师蒸馏给只用含噪感知的学生；④ 学生自适应/精修（约束 RL）。配合真实噪声建模与定制奖励，系统在多样球-门配置下取得强射门精度与进球成功率，并成功真机部署，为全身控制中的视觉运动技能学习树立基准。"
---

# Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input

**Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

学又快又稳的踢球技能是人形足球机器人的核心能力，但很难：需要快速摆腿、单脚支撑下的姿态稳定，还要在含噪感知与外部扰动（如对手）下鲁棒。本文用四阶段训练流水线：① 长距离追球（教师，用真值）；② 定向踢球（教师，用真值）；③ 教师蒸馏给只用含噪感知的学生；④ 学生自适应/精修（约束 RL）。配合真实噪声建模与定制奖励，系统在多样球-门配置下取得强射门精度与进球成功率，并成功真机部署，为全身控制中的视觉运动技能学习树立基准。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Striker Skill | 射门/踢球技能 |
| Teacher-Student | 教师-学生，教师用特权/真值，学生用真实观测蒸馏 |
| Noisy Sensing | 含噪感知，真实传感的噪声输入 |
| Constrained RL | 约束强化学习，带约束的策略优化 |
| Single Support | 单脚支撑，踢球时一脚支撑的稳定挑战 |
| Distillation | 蒸馏，把教师策略压进学生 |

## 为什么重要

- **教师-学生 + 含噪建模**是「特权训练→真机部署」的经典而有效配方；
- **运动技能（踢球）= 敏捷 + 稳定 + 感知鲁棒三难并存**，是检验全身控制的好任务；
- **约束 RL 做学生精修**有助于在保持安全/稳定约束下提升性能；
- **与人形球类运动线（羽毛球、拳击、足球门将）** 互为补充，体育任务正成为高动态全身控制的试金石。

## 解决什么问题

人形踢球同时要： - **快速摆腿**产生球速； - **单脚支撑下保持姿态稳定**； - 在**含噪感知**与**对手扰动**下仍鲁棒。

直接用含噪观测端到端学很难收敛，纯真值训练又无法部署。论文要：一套能把「真值教师的本领」迁到「含噪感知学生」并真机可用的训练配方。

## 核心机制

1. **四阶段教师-学生流水线**：追球/踢球教师 → 含噪感知学生 → 约束 RL 精修；
2. **含噪感知鲁棒**：真实噪声建模 + 定制奖励，弥合真值训练与真机部署；
3. **真机敏捷踢球**：多样球-门配置下高精度、可进球；
4. **方法基准**：为全身控制中的视觉运动技能学习提供参考。

方法拆解（深读笔记小节）：四阶段教师-学生流水线；真实噪声建模 + 定制奖励；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input.html> |
| arXiv | <https://arxiv.org/abs/2512.06571> |
| 作者 | Zifan Xu、Myoungkyu Seo、Dongmyeong Lee、Hao Fu、Jiaheng Hu、Jiaxun Cui、Yuqian Jiang 等（UT Austin，Peter Stone 组） |
| 发表 | 2025 年 12 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-agile-striker-skills-for-humanoid-socce.md](../../sources/papers/humanoid_pnb_learning-agile-striker-skills-for-humanoid-socce.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input.html>
- 论文：<https://arxiv.org/abs/2512.06571>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input.html)
