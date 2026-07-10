---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.19530"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_heavy-lifting-tasks-via-haptic-teleoperation-of.md
summary: "人形可在体力要求高的环境里支援人类，做需要全身协调的任务（抬运重物）。本文称之为动态移动操作（Dynamic Mobile Manipulation, DMM）——需要在动态交互力下同时控制行走、操作与姿态。论文提出在可调高度的轮式人形上做 DMM 的遥操作框架：一个人机接口（HMI）通过捕捉人体动作并施加力触觉反馈，把人体动作全身重定向到机器人。操作者用身体姿态调节机器人姿态与移动、用手臂引导操作；实时力反馈传递末端力旋量与平衡线索，闭合\"人感知 ↔ 机器人环境交互\"的回路。论文还比较了提供不同平衡辅助程度的遥行走映射，让操作者手动或自动调节机器人对负载扰动的倾斜。最终在抬举至多 2.5 kg（机器人质量 21%）的杠铃/箱子实验中验证了协调全身控制、变高度与抗扰。"
---

# Heavy lifting tasks via haptic teleoperation of a wheeled humanoid

**Heavy lifting tasks via haptic teleoperation of a wheeled humanoid** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形可在体力要求高的环境里支援人类，做需要全身协调的任务（抬运重物）。本文称之为动态移动操作（Dynamic Mobile Manipulation, DMM）——需要在动态交互力下同时控制行走、操作与姿态。论文提出在可调高度的轮式人形上做 DMM 的遥操作框架：一个人机接口（HMI）通过捕捉人体动作并施加力触觉反馈，把人体动作全身重定向到机器人。操作者用身体姿态调节机器人姿态与移动、用手臂引导操作；实时力反馈传递末端力旋量与平衡线索，闭合"人感知 ↔ 机器人环境交互"的回路。论文还比较了提供不同平衡辅助程度的遥行走映射，让操作者手动或自动调节机器人对负载扰动的倾斜。最终在抬举至多 2.5 kg（机器人质量 21%）的杠铃/箱子实验中验证了协调全身控制、变高度与抗扰。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| DMM | Dynamic Mobile Manipulation，动态移动操作 |
| HMI | Human-Machine Interface，人机接口 |
| Haptic Feedback | 力触觉反馈 |
| Wrench | 力旋量（力 + 力矩） |
| Telelocomotion | 遥行走，远程控制机器人行走 |
| Balance Assistance | 平衡辅助，帮助操作者维持机器人平衡 |

## 为什么重要

- **力触觉反馈是重载遥操作的关键**：让操作者"感受到"负载与平衡才能稳；
- **平衡辅助可调**兼顾操作者掌控与系统稳定；
- **轮式人形**在重载移动操作上是务实平台（与同组的物体参数估计工作互补）；
- DMM 的"同时控行走+操作+姿态"是全身控制的硬命题。

## 解决什么问题

搬重物的**动态移动操作**要**同时**控制行走、操作、姿态，且面对**负载扰动**： - 纯自主难、纯手动遥操作又难维持平衡； - 操作者需要**感知**末端力与平衡状态才能稳。

论文要：一套**带力触觉反馈、可调平衡辅助**的轮式人形遥操作框架。

## 核心机制

1. **DMM 遥操作框架**：力触觉 HMI 把人体动作全身重定向到轮式人形；
2. **分工映射**：身体管姿态/移动、手臂管操作；
3. **实时力旋量 + 平衡反馈**：闭合感知-交互回路；
4. **可调平衡辅助**：手动/自动抗负载扰动，2.5 kg（21% 自重）重物验证。

方法拆解（深读笔记小节）：HMI 全身重定向 + 力反馈；身体管姿态/移动、手臂管操作；实时力反馈闭环；可调平衡辅助 + 评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid.html> |
| arXiv | <https://arxiv.org/abs/2505.19530> |
| 作者 | Amartya Purushottam、Jack Yan、Christopher Yu、Joao Ramos（UIUC） |
| 发表 | 2025 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_heavy-lifting-tasks-via-haptic-teleoperation-of.md](../../sources/papers/humanoid_pnb_heavy-lifting-tasks-via-haptic-teleoperation-of.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid.html>
- 论文：<https://arxiv.org/abs/2505.19530>

## 推荐继续阅读

- [机器人论文阅读笔记：Heavy lifting tasks via haptic teleoperation of a wheeled humanoid](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid/Heavy_Lifting_Tasks_via_Haptic_Teleoperation_of_a_Wheeled_Humanoid.html)
