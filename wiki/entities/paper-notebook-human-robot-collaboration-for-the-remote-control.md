---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.05773"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_human-robot-collaboration-for-remote-control-of.md
summary: "越来越多人形被部署到医院、养老等场所，常由人远程操控。本文针对运动学冗余的移动人形的躯干-手臂协调难题，提出在自主与人控之间取平衡的人机协作方法：① 人发起（human-initiated）——操作者手动控制躯干运动；② 机器人发起（robot-initiated）——机器人依据可达性、任务目标与推断的人类意图做自主协调。围绕躯干-手臂宏-微（macro-micro）结构设计协调机制。通过 N=17 的用户研究，在任务表现、可操作度（manipulability）与能效等多指标上比较，分析参与者偏好，给出\"如何平衡自主与人输入以提升效率与任务执行\"的结论。已被 ICRA 2025 接收。"
---

# Human-Robot Collaboration for the Remote Control of Mobile Humanoid Robots with Torso-Arm Coordination

**Human-Robot Collaboration for the Remote Control of Mobile Humanoid Robots with Torso-Arm Coordination** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

越来越多人形被部署到医院、养老等场所，常由人远程操控。本文针对运动学冗余的移动人形的躯干-手臂协调难题，提出在自主与人控之间取平衡的人机协作方法：① 人发起（human-initiated）——操作者手动控制躯干运动；② 机器人发起（robot-initiated）——机器人依据可达性、任务目标与推断的人类意图做自主协调。围绕躯干-手臂宏-微（macro-micro）结构设计协调机制。通过 N=17 的用户研究，在任务表现、可操作度（manipulability）与能效等多指标上比较，分析参与者偏好，给出"如何平衡自主与人输入以提升效率与任务执行"的结论。已被 ICRA 2025 接收。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Torso-Arm Coordination | 躯干-手臂协调 |
| Kinematic Redundancy | 运动学冗余，自由度多于任务所需 |
| Human/Robot-Initiated | 人发起 / 机器人发起的协调 |
| Macro-Micro | 宏-微结构（躯干大范围 + 手臂精细） |
| Manipulability | 可操作度，末端灵活性度量 |
| User Study | 用户研究（N=17） |

## 为什么重要

- **"自主 ↔ 人控"平衡**是辅助/医护场景遥操作的核心设计问题；
- **意图推断驱动的机器人发起协调**减轻操作者负担，是共享自主的方向；
- **以人为本的用户研究**对遥操作系统评价不可或缺（不止任务成功率）；
- 躯干-手臂冗余协调对所有移动人形操作都有借鉴。

## 解决什么问题

远程操控**移动人形**（医院/养老场景）时： - 机器人**运动学冗余**（躯干 + 手臂），**协调难**； - 全自主不够灵活、全手动负担重； - 不清楚**哪种自主/人控平衡**操作者更偏好、更高效。

论文要：设计并比较**躯干-手臂协调**的人机协作策略，找到好的"自主 ↔ 人控"平衡。

## 核心机制

1. **躯干-手臂协调的人机协作方法**：人发起 + 机器人发起两类策略；
2. **宏-微结构利用运动学冗余**：躯干大范围、手臂精细；
3. **机器人发起协调**：依可达性、任务目标与推断意图自主协调；
4. **N=17 用户研究**：多指标比较、给出自主/人控平衡建议（ICRA 2025）。

方法拆解（深读笔记小节）：两类协调策略；躯干-手臂宏-微结构；用户研究（N=17）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2505.05773> |
| 作者 | Nikita Boguslavskii、Lorena Maria Genua、Zhi Li（WPI） |
| 发表 | 2025 年 5 月 |
| 会议 | ICRA 2025 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_human-robot-collaboration-for-remote-control-of.md](../../sources/papers/humanoid_pnb_human-robot-collaboration-for-remote-control-of.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2505.05773>

## 推荐继续阅读

- [机器人论文阅读笔记：Human-Robot Collaboration for the Remote Control of Mobile Humanoid Robots with Torso-Arm Coordination](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots/Human-Robot_Collaboration_for_Remote_Control_of_Mobile_Humanoid_Robots.html)
