---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2509.14935"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_cad-driven-co-design-for-flight-ready-jet-powere.md
summary: "CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids

**CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，不含实验结论。

## 流程总览

```mermaid
flowchart TD
    A["iRonCub-Mk3 CAD 基线<br/>8 个几何参数：喷气发动机角度/偏移/高度、前臂长、肩宽、髋距、踝连杆高、脚长"] --> B["Uniform Latin Hypercube 采样<br/>生成 5000 个设计实例"]
    B --> C["CAD 自动提取质量/质心/惯量<br/>写入 URDF 与 MuJoCo XML；FEM 结构校验，不合格者剔除"]
    C --> D["K-means 聚类（几何参数空间）<br/>得到 N 个代表性质心模型（50/100/250）"]
    E["最小加加速度轨迹<br/>由空间航点插值，输出 CoM 位置/速度参考"] --> F
    D --> F["NSGA-II 候选解：质心模型索引 + 8 个 MPC 代价权重"]
    F --> G["仿真：所选模型 + 线性化 MPC 跟踪固定轨迹"]
    G --> H["目标 1：CoM 跟踪 MSE<br/>目标 2：推力能耗 E"]
    G -->|"QP 失败或误差 > 2.5 m"| X["惩罚 / 剔除"]
    H --> I["Pareto 支配排序，迭代进化"]
    I --> F
    I --> J["输出 Pareto 最优的设计-控制组合"]
```

图依据：论文 §III-A–D / Algorithm 1，§II-C。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/cad-driven-co-design-for-flight-ready-jet-powere` |
| arXiv | <https://arxiv.org/abs/2509.14935> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前提供的是「待读条目」而不是论文解读：它把这条 12_Hardware_Design 条目挂进知识库以保持检索连通，实质技术判断要等深读笔记落地。**

- 页面已确认的信息仅有分类（12_Hardware_Design）、arXiv 编号与计划文件夹路径；CAD 驱动的协同设计流程与喷气动力人形的量化指标均尚未记录。
- 适用边界很窄：可用于按分类检索，不能作为该论文技术结论的引用依据。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_cad-driven-co-design-for-flight-ready-jet-powere.md](../../sources/papers/humanoid_pnb_cad-driven-co-design-for-flight-ready-jet-powere.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2509.14935>
- [arXiv PDF](https://arxiv.org/pdf/2509.14935)：已通读方法部分（实验与附录以略读为准）：§II-C 几何参数与控制权重、§III Co-Design Framework（III-A–D、Algorithm 1、Table II）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
