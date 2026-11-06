---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2411.03532"
venue: "2024.11"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_a-behavior-architecture-for-fast-humanoid-robot.md
summary: "A Behavior Architecture for Fast Humanoid Robot Door Traversals：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# A Behavior Architecture for Fast Humanoid Robot Door Traversals

**A Behavior Architecture for Fast Humanoid Robot Door Traversals** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

A Behavior Architecture for Fast Humanoid Robot Door Traversals 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论（论文系统模块与数据流见下方「流程总览」）。

## 流程总览

```mermaid
flowchart TB
    subgraph PERC["感知: 门把手检测"]
        ZED["ZED 2i 彩色 + 深度<br/>(朝前并下倾 43°)"]
        YOLO["YOLOv8 分割门开启机构<br/>(推杠 / 压杆 / 旋钮 / 免手拉手)"]
        POS["掩膜腐蚀 + 深度点下采样<br/>取质心 + alpha 滤波 = 位置"]
        PLN["Planar Region Extraction<br/>找门板平面, 法向 = 朝向"]
        STB["稳定检测滤波<br/>(1 秒窗内 ≥60% 帧 YOLO 与平面关联成功)"]
        ZED --> YOLO --> POS
        ZED --> PLN
        POS --> STB
        PLN --> STB
    end
    STB --> KG["Door Handle Node / 场景知识图"]
    subgraph BEH["行为协调: 树形运行时"]
        ROOT["Door Traversal 根节点<br/>(按门类型分流, 失败时重试/换策略)"]
        SUB["子树: 动作序列 (Action Sequence)<br/>'execute after' 字段实现分层并发"]
        PRIM["Action Primitives (叶节点)<br/>脚步 / 手臂 / 手 / 胸 / 骨盆轨迹<br/>含 on-entry / on-exit 条件"]
        ROOT --> SUB --> PRIM
    end
    KG --> ROOT
    OP["操作员界面<br/>数字孪生 + 遥操作 + 行为在线编写与执行"] <--> ROOT
    PRIM -->|"轨迹 / 脚步请求"| WBC
    subgraph CTRL["全身控制"]
        WBC["IHMC 动量式全身控制器<br/>(QP 协调运动任务与接触约束)"]
        WLK["步行控制器<br/>(站立/转移/摆动状态机, ICP 平衡)"]
        WBC --- WLK
    end
    CTRL -->|"执行状态反馈"| ROOT
    WBC --> HW["Nadia 人形机器人"]
```

图依据：论文 §2 Architecture Overview（Fig.2）、§2.1–2.4（Fig.3–5、9–10）、§4 Door Handle Detection（Fig.17）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/A_Behavior_Architecture_for_Fast_Humanoid_Robot_Door_Traversals` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是一条待读记录，而不是对这篇论文的归纳——它记录的是「这篇该读、归在哪一类、还没读」。**

- 页面能给出的确定信息只有三项：分类为 04_Loco-Manipulation_and_WBC、深读状态为待撰写、计划文件夹路径已预留。
- 论文本身的机制、量化 benchmark 与实机结论 **本页均未涵盖**，不要把这里的任何表述当作对该工作的评价。
- 它的作用是让你能从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到它到该实体；深读笔记完成后本页应链向笔记站并补成完整摘要。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_a-behavior-architecture-for-fast-humanoid-robot.md](../../sources/papers/humanoid_pnb_a-behavior-architecture-for-fast-humanoid-robot.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2411.03532)：已通读方法部分（实验与附录以略读为准）：§2 Architecture Overview（2.1–2.4）、§4 Door Handle Detection

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
