---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_climber-force-and-motion-estimation-from-video.md
summary: "本工作研究从视频同时估计攀岩者的运动与受力：在攀岩这一强接触、离地、全身的运动下，恢复攀岩者的 3D 运动以及其作用于岩点（holds）的接触受力。相较于需要在岩壁/岩点上布置力传感器的侵入式方案，本方法仅凭视频就能推断人-岩交互力，从而非侵入地分析攀岩动作与发力模式。这属于\"从视频做力 + 运动联合估计\"的范式——把人体姿态估计与接触/力推断结合，对运动科学、教练辅助乃至机器人攀爬都有参考价值。"
---

# Climber Force and Motion Estimation from Video

**Climber Force and Motion Estimation from Video** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本工作研究从视频同时估计攀岩者的运动与受力：在攀岩这一强接触、离地、全身的运动下，恢复攀岩者的 3D 运动以及其作用于岩点（holds）的接触受力。相较于需要在岩壁/岩点上布置力传感器的侵入式方案，本方法仅凭视频就能推断人-岩交互力，从而非侵入地分析攀岩动作与发力模式。这属于"从视频做力 + 运动联合估计"的范式——把人体姿态估计与接触/力推断结合，对运动科学、教练辅助乃至机器人攀爬都有参考价值。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Force Estimation | 受力（接触力）估计 |
| Motion Estimation | 3D 运动/姿态估计 |
| Hold | 岩点（攀岩抓握/踩踏点） |
| Contact | 人-岩接触 |
| Monocular Video | 单目（普通）视频 |
| Non-invasive | 非侵入（免力传感器） |

## 为什么重要

- **"从视频估接触力"对强接触全身任务很有价值**：攀岩、搬运、推压等都需理解接触力；
- **运动 + 受力联合估计**把运动学与动力学连接，呼应人形 loco-manip 对"外力来源建模"的诉求；
- 攀岩这类**离地多接触**运动可为人形攀爬/跑酷提供参考与受力先验；
- 非侵入受力估计有望低成本扩充"带力标注"的人类数据。

## 解决什么问题

攀岩的**受力分析**通常要**侵入式力传感**（岩点/岩壁布传感器），成本高、难普及；而： - 攀岩是**强接触、离地、全身**运动，姿态与受力都难估； - 想**仅凭视频**就同时得到**运动 + 受力**。

本工作要：从普通视频**非侵入地**联合估计攀岩者的**3D 运动与接触受力**。

## 核心机制

1. **从视频联合估计攀岩运动 + 受力**：非侵入；
2. **强接触/离地全身**场景下的姿态 + 接触力推断；
3. **免力传感器**：仅凭视频分析发力模式；
4. **面向运动科学/教练/机器人攀爬**的分析工具。

方法拆解（深读笔记小节）：视频 → 3D 攀岩运动；接触与受力推断；非侵入分析；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Climber_Force_and_Motion_Estimation_from_Video/Climber_Force_and_Motion_Estimation_from_Video.html> |
| 发表 | 2025 年 4 月（arXiv，详见项目页） |
| 项目主页 | [rihat99.github.io/climb_force](https://rihat99.github.io/climb_force/) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_climber-force-and-motion-estimation-from-video.md](../../sources/papers/humanoid_pnb_climber-force-and-motion-estimation-from-video.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Climber_Force_and_Motion_Estimation_from_Video/Climber_Force_and_Motion_Estimation_from_Video.html>

## 推荐继续阅读

- [机器人论文阅读笔记：Climber Force and Motion Estimation from Video](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Climber_Force_and_Motion_Estimation_from_Video/Climber_Force_and_Motion_Estimation_from_Video.html)
