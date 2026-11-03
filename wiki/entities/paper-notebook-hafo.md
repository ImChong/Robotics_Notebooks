---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2511.20275"
venue: "2025.11"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_hafo.md
summary: "HAFO：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# HAFO

**HAFO: A Force-Adaptive Control Framework for Humanoid Robots in Intense Interaction Environments** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

HAFO 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里保留分类位置与原文入口，方法流程见下方「流程总览」，其余结论不给出。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练：双智能体 PPO（共享全身状态 s_t）"]
        lc["下身指令 c^l<br/>线速度/角速度 + 站姿相位"] --> pl["下身策略 π_lower<br/>外扰：上身对抗扰动 + 高频随机推力"]
        amass["AMASS 人体动作<br/>PHC 重定向 + 可行性过滤"] --> cur["渐进模仿课程<br/>q_curr = q_0 + α_i·(q_tar − q_0)"]
        cur --> pu["上身策略 π_upper<br/>残差动作 θ_target = θ_ref + a_u，|a_u| ≤ Δ<br/>外扰：末端随机 3D 力"]
        spring["虚拟弹簧-阻尼模型<br/>模拟拉力：m·ẍ = Kp(x_des − x) + Kd(ẋ_des − ẋ) + f_ext"] --> pl
        spring --> pu
        dr["域随机化"] --> pl
        dr --> pu
    end
    pl --> joint["下身关节目标"]
    pu --> joint2["上身关节目标"]
    subgraph DEPLOY["部署：遥操作"]
        tele["遥操作输入"] --> ik["高效 IK 实时解算关节角<br/>作为上身指令"]
    end
    ik --> pu
    joint --> robot["人形机器人全身控制"]
    joint2 --> robot
```

图依据：论文 §3.1-§3.4 / Fig.1-2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/HAFO__A_Force-Adaptive_Control_Framework_for_Humanoid_Robots_in_Intense_Interact` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是待读条目而非论文归纳：它只登记了「HAFO 属于 04_Loco-Manipulation_and_WBC 这一类」这一件事。**

- 页面现状：status 为 planned，正文只有分类归属与 progress.json 待深读状态，没有可供提炼的机制、指标或实验内容。
- 从标题能读出的定位仅限于「人形在强交互环境下的力自适应控制框架」，属于 WBC / loco-manipulation 谱系；具体做法与效果在本页无依据，不应据此下判断。
- 使用方式：把本页当检索入口——分类页 [paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) 与 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)；深读笔记完成后再回填实质结论。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hafo.md](../../sources/papers/humanoid_pnb_hafo.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2511.20275)：已通读方法部分（实验与附录以略读为准）：§3.1-§3.4 与 Fig.1 图注


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
