# Robotics_Notebooks

机器人技术栈知识库 / Robotics research and engineering wiki.

这个项目正在从“资源堆叠式思维导图”重构为“可持续维护的机器人知识库”。

## 当前结构

- `wiki/`：结构化知识页（核心）
- `sources/`：原始资料与资源索引
- `schema/`：知识库维护规则
- `exports/`：未来给网页/思维导图导出的数据层
- `index.md`：知识入口索引
- `log.md`：重构变更日志

## 从哪里开始看

### 知识入口
- [index.md](index.md)
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

## 与其他项目的边界

- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)：单篇论文深读
- [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)：跨主题、跨论文、跨模块的机器人知识组织

一句话：

> `Humanoid_Robot_Learning_Paper_Notebooks` 负责点，`Robotics_Notebooks` 负责线和面。

## 当前状态

- 当前思维导图网页渲染暂时不改
- 当前重点是重构知识层与资料层
- 后续会把 `wiki/` 逐步导出为网页可消费的数据
