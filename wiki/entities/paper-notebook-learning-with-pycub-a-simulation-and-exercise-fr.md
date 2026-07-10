---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.01756"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-with-pycub.md
summary: "pyCub 是一个开源、基于物理的 iCub 人形仿真，并配套练习用于教学生人形机器人学基础。相比已有 iCub 仿真器（iCub SIM、iCub Gazebo）需要 C++ 代码与 YARP 中间件，pyCub 无需 YARP、用 Python 即可。它完整仿真了带全部关节的机器人，含眼部双相机与 iCub 独特的敏感皮肤（体表 4000 个感受器）。练习从速度/关节/笛卡尔空间的基础控制，到注视、抓取、反应式控制等更复杂任务；整套用 Python 编写与控制，即使编程基础很少的人也能用，练习可分级到不同难度。作者在两轮人形机器人课程中验证了该框架。"
---

# Learning with pyCub

**Learning with pyCub: A Simulation and Exercise Framework for Humanoid Robotics** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

pyCub 是一个开源、基于物理的 iCub 人形仿真，并配套练习用于教学生人形机器人学基础。相比已有 iCub 仿真器（iCub SIM、iCub Gazebo）需要 C++ 代码与 YARP 中间件，pyCub 无需 YARP、用 Python 即可。它完整仿真了带全部关节的机器人，含眼部双相机与 iCub 独特的敏感皮肤（体表 4000 个感受器）。练习从速度/关节/笛卡尔空间的基础控制，到注视、抓取、反应式控制等更复杂任务；整套用 Python 编写与控制，即使编程基础很少的人也能用，练习可分级到不同难度。作者在两轮人形机器人课程中验证了该框架。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| pyCub | 本文的 Python iCub 仿真框架 |
| iCub | 一款知名人形研究平台 |
| YARP | iCub 传统中间件（本文免去） |
| Cartesian Control | 笛卡尔空间控制 |
| Tactile Skin | 触觉皮肤（4000 感受器） |
| Exercise Framework | 练习框架，分级教学任务 |

## 为什么重要

- **降低教学/上手门槛**对扩大人形研究社区很重要；
- **触觉皮肤仿真**为接触/灵巧研究提供少见的教学资源；
- **纯 Python 生态**契合当下 RL/IL 工具链，利于教学到科研衔接；
- 与重型仿真（Isaac、MuJoCo）互补，定位"教学友好"。

## 解决什么问题

人形机器人学**教学门槛高**： - 现有 iCub 仿真需 **C++ + YARP**，对学生不友好； - 缺**纯 Python、低门槛**且涵盖触觉皮肤等特性的教学仿真与练习。

pyCub 要：一个**纯 Python、免 YARP、带分级练习**的 iCub 教学仿真框架。

## 核心机制

1. **纯 Python、免 YARP 的 iCub 仿真**：大幅降低使用门槛；
2. **完整仿真含触觉皮肤**：双相机 + 4000 感受器，覆盖 iCub 特色；
3. **分级练习框架**：从基础控制到注视/抓取/反应式；
4. **教学验证 + 开源**：两轮课程实测，开源含 Docker。

方法拆解（深读笔记小节）：纯 Python、免 YARP 的 iCub 物理仿真；分级练习；低门槛 + 课程验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics.html> |
| arXiv | <https://arxiv.org/abs/2506.01756> |
| 作者 | Lukas Rustler、Matej Hoffmann（捷克理工大学 CTU） |
| 发表 | 2025 年 6 月 |
| 源码 | 开源，含文档与 Docker 支持 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-with-pycub.md](../../sources/papers/humanoid_pnb_learning-with-pycub.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics.html>
- 论文：<https://arxiv.org/abs/2506.01756>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning with pyCub](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics.html)
