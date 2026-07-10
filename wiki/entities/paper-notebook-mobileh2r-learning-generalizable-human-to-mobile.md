---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2501.04595"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_mobileh2r.md
summary: "MobileH2R 是一个学习泛化的、基于视觉的「人到移动机器人（H2MR）递交」技能的框架。不同于传统固定底座递交，该任务要求移动机器人借移动性在大工作空间里可靠接物。MobileH2R 完全用可扩展、多样的合成数据学习，开发了三类技术：① 可扩展地生成多样的全身人体运动数据；② 自动造安全、易模仿的演示；③ 高效的 4D 模仿学习，协调机器人底盘与机械臂的运动。在仿真与真实世界评测中，相比基线，各情形成功率至少 +15%。"
---

# MobileH2R

**MobileH2R: Learning Generalizable Human to Mobile Robot Handover Exclusively from Scalable and Diverse Synthetic Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

MobileH2R 是一个学习泛化的、基于视觉的「人到移动机器人（H2MR）递交」技能的框架。不同于传统固定底座递交，该任务要求移动机器人借移动性在大工作空间里可靠接物。MobileH2R 完全用可扩展、多样的合成数据学习，开发了三类技术：① 可扩展地生成多样的全身人体运动数据；② 自动造安全、易模仿的演示；③ 高效的 4D 模仿学习，协调机器人底盘与机械臂的运动。在仿真与真实世界评测中，相比基线，各情形成功率至少 +15%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| H2MR | Human-to-Mobile-Robot 人到移动机器人 |
| Handover | 递交，把物体交给机器人 |
| Synthetic Data | 合成数据（全身人体运动） |
| 4D Imitation | 4D 模仿学习（含时间的轨迹） |
| Base-Arm Coordination | 底盘-机械臂协调 |
| Mobile Robot | 移动机器人 |

## 为什么重要

- **移动递交需底盘-臂协调**，对人形（移动 + 操作）直接相关；
- **完全合成数据**是绕开真实采集的可扩展路线，呼应 DexMimicGen/DreamGen；
- **自动造安全演示**降低数据工程；
- 人机递交是人形服务场景的高频交互。

## 解决什么问题

**移动机器人接人递来的物体**比固定底座难： - 需在**大工作空间**移动接物，**底盘 + 臂**要协调； - 真实递交数据**难采**； - 要对**多样人类递交动作**泛化。

MobileH2R 要：**仅用合成数据**学出泛化的视觉 H2MR 递交。

## 核心机制

1. **H2MR 递交框架**：移动机器人大工作空间可靠接物；
2. **完全合成数据**：可扩展生成全身人体运动 + 自动安全演示；
3. **4D 模仿学习**：协调底盘与机械臂；
4. **≥ +15%**：仿真与真机均超基线。

方法拆解（深读笔记小节）：可扩展合成全身人体运动数据；自动造安全易模仿的演示；高效 4D 模仿学习（底盘-臂协调）；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data.html> |
| arXiv | <https://arxiv.org/abs/2501.04595> |
| 作者 | Zifan Wang、Ziqing Chen、Junyu Chen、Yunze Liu、Xueyi Liu、He Wang、Li Yi 等（清华等） |
| 发表 | 2025 年 1 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_mobileh2r.md](../../sources/papers/humanoid_pnb_mobileh2r.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data.html>
- 论文：<https://arxiv.org/abs/2501.04595>

## 推荐继续阅读

- [机器人论文阅读笔记：MobileH2R](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data/MobileH2R__Learning_Generalizable_Human_to_Mobile_Robot_Handover_from_Synthetic_Data.html)
