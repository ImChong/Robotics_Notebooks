# Robotics_Notebooks

[![Deploy GitHub Pages](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml/badge.svg)](https://github.com/ImChong/Robotics_Notebooks/actions/workflows/pages.yml)

机器人技术栈知识库 / Robotics research and engineering wiki.

这个项目正在从“资源堆叠式思维导图”重构为一个**面向机器人全栈成长路径的技术栈导航项目**。

## 项目定位

- **终极目标：** 全栈机器人工程师
- **当前切入口：** 机器人运动控制算法工程师
- **当前重点：** 运动控制、强化学习、模仿学习、人形机器人相关能力

一句话定位：

> 面向人形机器人运动控制、强化学习与模仿学习的技术栈地图与学习路线图，最终通向机器人全栈工程能力。

## 方法说明

本项目的知识组织与迭代方式，参考并借鉴了 Andrej Karpathy 提出的个人知识库/研究组织方法：
- https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

核心思路不是简单堆资料，而是把原始资源、结构化知识、路线图、依赖关系和可执行入口逐步拆层，形成一个可持续演进的研究与工程知识系统。

## v1 目录结构

- `wiki/`：结构化知识页（核心知识层）
- `sources/`：原始资料与资源索引
- `resources/`：课程、训练笔记、know-how 等沉淀资源
- `roadmap/`：成长路线与阶段导航
- `tech-map/`：技术栈地图与依赖关系
- `references/`：论文导航、开源生态、项目索引
- `docs/homepage-copy-v1.md`：首页文案与导航文案草稿
- `schema/`：知识库维护规则
- `exports/`：未来给网页/思维导图导出的数据层
- `index.md`：知识入口索引
- `STRUCTURE_v1.md`：项目目录结构蓝图 v1
- `log.md`：重构变更日志

## 从哪里开始看

### 结构蓝图
- [STRUCTURE_v1.md](STRUCTURE_v1.md)
- [index.md](index.md)

### 知识入口
- [Robot Learning Overview](wiki/overview/robot-learning-overview.md)
- [Humanoid Control Roadmap](wiki/roadmaps/humanoid-control-roadmap.md)

### 核心概念 / 方法
- [Sim2Real](wiki/concepts/sim2real.md)
- [Whole-Body Control](wiki/concepts/whole-body-control.md)
- [Reinforcement Learning](wiki/methods/reinforcement-learning.md)
- [Imitation Learning](wiki/methods/imitation-learning.md)
- [Locomotion](wiki/tasks/locomotion.md)
- [Manipulation](wiki/tasks/manipulation.md)
- [WBC vs RL](wiki/comparisons/wbc-vs-rl.md)

## 资源入口

旧版 README 中的大量资源链接已从首页移出，避免继续把 README 堆成资源垃圾场。

请改从这里进入：
- [sources/README.md](sources/README.md)
- [旧版 README 资源归档](sources/notes/legacy-readme-resource-map.md)

## 当前状态

- 当前思维导图网页渲染暂时不改
- 当前重点是先把知识结构、成长路线和模块边界搭对
- 后续再逐步把 `wiki/`、`roadmap/`、`tech-map/` 导出为网页可消费的数据
