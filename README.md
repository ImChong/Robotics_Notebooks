# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://imchong.github.io/Robotics_Notebooks/)
[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)
[![License](https://img.shields.io/github/license/ImChong/Robotics_Notebooks)](./LICENSE)

---

## 适合谁

如果你：
- 想系统学人形机器人运动控制 / 强化学习 / 模仿学习
- 想有一条能照着走的学习路线，而不是到处找散资源
- 有一定编程基础（Python / C++）和本科数学基础

这个项目是为你设计的。

如果你不知道从哪开始，直接看 [路线A：运动控制成长路线](roadmap/route-a-motion-control.md)。

---

## 这个项目是什么

`Robotics_Notebooks` 是一个**技术栈导航项目**，不是资源收集箱。

它把机器人工程能力的成长路径拆成模块，然后告诉你：
- 每个模块在解决什么问题
- 模块之间是什么关系
- 应该先学什么、再学什么
- 每一步学完能拿出什么

一句话：

> 论文项目负责点，技术栈项目负责线和面，个人主页负责展示。

---

## 这个项目不是什么

- 不是教科书——它不系统讲理论，它告诉你去哪儿找、怎么串
- 不是笔记堆——它有结构、有依赖关系、有成长路线
- 不是工具文档——工具在实体页（后续会补）

---

## 从哪里开始

### 如果你是第一次来
**推荐从这里进入：**

1. [路线A：运动控制算法工程师成长路线](roadmap/route-a-motion-control.md) — 照着走就对了

如果你的目标更具体：
- [如果想用强化学习做 locomotion](roadmap/learning-paths/if-goal-locomotion-rl.md)
- [如果想学模仿学习与技能迁移](roadmap/learning-paths/if-goal-imitation-learning.md)

### 如果你想看知识页
按这条主链读：

```
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

从 [LIP / ZMP](wiki/concepts/lip-zmp.md) 开始即可。

### 如果你想看技术栈地图
从 [tech-map/overview.md](tech-map/overview.md) 进入。

---

## 项目结构

| 目录 | 是什么 | 用来做什么 |
|------|--------|-----------|
| `wiki/` | 结构化知识页 | 核心概念和方法 |
| `roadmap/` | 成长路线 | 照着走的学习路径 |
| `tech-map/` | 技术栈地图 | 模块关系与依赖 |
| `sources/` | 原始资源索引 | 论文 / 课程 / 工具入口 |
| `references/` | 论文导航 | 按主题整理的论文列表 |
| `docs/` | 项目文档 | 执行清单 / 变更记录 |

---

## 当前执行清单

项目的下一阶段目标和当前待办，都在这份文件里：
- [技术栈项目下一阶段执行清单 v2](docs/tech-stack-next-phase-checklist-v2.md)

它不是给读者看的，是给我们一起维护项目方向用的。

---

## 和其他项目的边界

| 项目 | 做什么 |
|------|--------|
| [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)（本项目）| 跨模块知识组织、成长路线、技术栈地图 |
| [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) | 单篇论文深读笔记 |
| [`ImChong.github.io`](https://github.com/ImChong/ImChong.github.io) | 个人简历与对外展示 |

---

## 当前阶段

项目已经完成 V1 的主干知识页搭建，正在推进 V2：**结构联动、入口优化、路线执行化**。

具体进展见 [执行清单 v2](docs/tech-stack-next-phase-checklist-v2.md)。

---

## 方法说明

知识组织参考了 Andrej Karpathy 的个人研究系统方法：
- https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

核心思路：不是堆资料，而是把资源、知识、路线、依赖关系分层拆开，形成可持续演进的研究与工程系统。
