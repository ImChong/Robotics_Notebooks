---
type: entity
project_id: deeprobotics-evolvingnav
tags: [paper, robot-navigation, 4d-memory, belief, benchmark, deep-robotics]
status: complete
updated: 2026-10-24
arxiv: "2609.39166"
project: "https://developer.deeprobotics.cn/research/articles/evolving-nav/index.html"
related:
  - ../concepts/hierarchical-quadruped-navigation-stack.md
sources:
  - ../../sources/sites/deeprobotics-evolvingnav.md
summary: "EvolvingNav（arXiv:2609.39166；云深处 + 浙大）：把带时间戳的 3D 历史转成对「到达时刻」世界的预测式 4D 信念，闭环：行动→可见性限定证据→信念修正→记忆更新；配套 EvoWorld-Bench（54 场景、803.68K 可执行任务）。"
---

# EvolvingNav（arXiv:2609.39166）

**EvolvingNav**（*Beyond the Remembered World: Predictive 4D Belief for Persistent Navigation in Evolving Worlds*，[arXiv:2609.39166](https://arxiv.org/abs/2609.39166)，Mingjian Gao、Zhaocheng Li、Haoyang Huang 等；Deeprobotics、浙江大学、UCSD、CUHK；[官方项目页](https://developer.deeprobotics.cn/research/articles/evolving-nav/index.html)）。

## 一句话定义

**EvolvingNav 把已记住的世界当作假设：预测到达时仍成立的状态，只检查必要之处，用证据回路持续更新 4D 记忆。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EvolvingNav | Evolving-world Navigation | 本文方法（命名依标题） |
| 4D | 3D space + time | 带时间的三维信念 / 记忆 |
| EvoWorld-Bench | Evolving World Benchmark | 本文配套基准（命名依项目页） |
| HSSD / HM3D | Habitat 场景数据集 | 评测场景来源（项目页列出） |

## 核心思路

- 预测式导航、信念引导搜索、证据重规划、预测式时间 4D 记忆四个环节。
- EvoWorld-Bench 覆盖预测导航、搜索、重规划、在线动态、迁移与具身问答。

## 评测要点（项目页自报）

| 项 | 数值 |
|----|------|
| EvoWorld-Bench | 54 场景；803.68K 可执行任务实例 |
| 首次检查成功率 | 61.32% |
| 预算搜索成功率 | 86.18% |
| 真机 | 室内巡逻、送文件两段演示 |

## 开源状态

项目页提供 GitHub 代码链接；本页未逐文件核验。

## 结论

**数字为项目页汇总，对比基线与协议请读论文；真机仅展示演示。**

## 关联页面

- [DR-WM](./paper-dr-wm.md) — 云深处同批研究
- [云深处 rl_training](./deeprobotics-rl-training.md) — 同公司 RL 训练工程
- [hierarchical-quadruped-navigation-stack](../concepts/hierarchical-quadruped-navigation-stack.md)

## 参考来源

- [项目页归档](../../sources/sites/deeprobotics-evolvingnav.md)
- [arXiv 原文](https://arxiv.org/abs/2609.39166)
- [官方项目页](https://developer.deeprobotics.cn/research/articles/evolving-nav/index.html)
- [云深处前沿探索](https://developer.deeprobotics.cn/research/)
