---
type: entity
tags: [course, education, machine-learning, visualization, interactive, notre-dame]
status: complete
updated: 2026-09-14
related:
  - ../methods/vla.md
  - ../concepts/foundation-policy.md
  - ./paper-notebook-robot-crash-course.md
sources:
  - ../../sources/sites/williamtheisen-ai-exploring.md
summary: "William Theisen AI 学习枢纽：Notre Dame 交互式 AI 可视化站（30+ 主题）+ CSE 30124/10124 课程入口；适合 ML/RL/Transformer 基础补课。"
---

# Exploring Artificial Intelligence（William Theisen 学习枢纽）

**Exploring Artificial Intelligence**（[ai.williamtheisen.com](https://ai.williamtheisen.com/)，William Theisen，圣母大学 University of Notre Dame）是面向本科与自学者的 **交互式 AI 教学站点**：在浏览器中可视化经典与深度学习概念，并链到两门课程官网与开源讲义仓库。

## 一句话定义

**Notre Dame 教师维护的静态交互教具集**——用可拖拽、可调参的 HTML 可视化讲清熵、KNN、神经网络、CNN、RL、Transformer 等基础，并挂接 CSE 30124（AI 导论）与 CSE 10124（Building ChatGPT）课程材料。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 站点主题：人工智能通识与可视化 |
| ML | Machine Learning | 监督/无监督经典算法可视化 |
| RL | Reinforcement Learning | `reinforcement-learning.html` 交互演示 |
| CNN | Convolutional Neural Network | 卷积与 MNIST 演示 |
| PCA | Principal Component Analysis | 降维探索器 |
| RLHF | Reinforcement Learning from Human Feedback | 对齐专题页 |
| ND | University of Notre Dame | 主办院校 |

## 为什么重要

- **降低抽象概念门槛：** 信息论熵、交叉熵、反向传播、注意力等可在浏览器即时实验，适合机器人研究者补课 ML 基础。
- **与课程闭环：** [CSE 30124](https://30124.williamtheisen.com/) 与 [CSE 10124](https://10124.williamtheisen.com/) 提供结构化进度；GitHub 仓库公开 notebook 预览。
- **覆盖现代专题：** 含 Diffusion、BPE、RLHF、Transformer 等近年热点可视化，不仅是传统 ML 博物馆。
- **零安装：** 纯静态页，适合课堂演示与自学；不替代动手训练但补足直觉。

## 核心信息

| 字段 | 内容 |
|------|------|
| 维护者 | William Theisen |
| 机构 | 圣母大学（University of Notre Dame） |
| 主站 | https://ai.williamtheisen.com/ |
| 课程仓 | [nd-cse-30124](https://github.com/wtheisen/nd-cse-30124)、[nd-cse-10124](https://github.com/wtheisen/nd-cse-10124) |
| 开源状态 | **已开源/可访问** 静态站与课程仓库 |

## 核心结构

### 交互可视化（部分）

| 类别 | 页面示例 |
|------|----------|
| 搜索与博弈 | `astar.html`、`minimax.html` |
| 经典 ML | `knn.html`、`dtree.html`、`svm.html`、`naive-bayes.html`、`logistic-regression.html` |
| 神经网络 | `perceptron.html`、`neural-network.html`、`backpropagation.html`、`cnn.html` |
| 序列与生成 | `rnn.html`、`transformer.html`、`attention.html`、`diffusion.html` |
| 概率与信息论 | `entropy.html`、`cross-entropy.html`、`markov.html` |
| 强化学习 | `reinforcement-learning.html`、`rlhf.html` |

### 课程入口

| 课程 | 站点 | 主题 |
|------|------|------|
| CSE 30124 | [30124.williamtheisen.com](https://30124.williamtheisen.com/) | Introduction to Artificial Intelligence |
| CSE 10124 | [10124.williamtheisen.com](https://10124.williamtheisen.com/) | Building ChatGPT |

## 局限与风险

- **非机器人专用：** 不含操纵仿真、VLA 训练或真机部署内容；与 [RoboLab](./robolab.md) 等基准无直接关系。
- **静态前端：** 大规模实验仍需 Python/Jupyter 或课程 notebook，可视化本身不训练模型。
- **课程材料边界：** 部分 lab 话题可能对访客静音；以公开页与 GitHub 为准。

## 与其他页面的关系

- [VLA](../methods/vla.md) — 理解 Transformer/RL 基础后再读 VLA 架构更顺。
- [foundation-policy](../concepts/foundation-policy.md) — CSE 10124 与 LLM 构建主题相邻。
- 机器人操纵评测主线见 [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)。

## 推荐继续阅读

- [主站索引](https://ai.williamtheisen.com/)
- [CSE 30124 课程站](https://30124.williamtheisen.com/)
- [nd-cse-30124 仓库](https://github.com/wtheisen/nd-cse-30124)

## 参考来源

- [站点归档 williamtheisen-ai-exploring](../../sources/sites/williamtheisen-ai-exploring.md)
