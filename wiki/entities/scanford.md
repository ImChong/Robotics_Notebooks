---
type: entity
tags:
  - hardware
  - mobile-manipulation
  - data-collection
  - vlm
  - library
  - stanford
  - toyota-research
status: complete
updated: 2026-09-07
summary: "Scanford：Stanford+TRI 的 RPDF 图书馆盘点机器人（Franka FR3+TidyBot++），东亚图书馆两周扫 2103 书架；VLM+RAG 自动标注；未开源。"
related:
  - ./paper-scanford-robot-powered-data-flywheel.md
  - ../concepts/data-flywheel.md
  - ../tasks/manipulation.md
  - ../queries/humanoid-robot-data-collection-landscape.md
sources:
  - ../../sources/sites/scanford.md
  - ../../sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md
---

# Scanford（图书馆盘点机器人）

**Scanford** 是 [Robot-Powered Data Flywheel](./paper-scanford-robot-powered-data-flywheel.md) 框架在 **斯坦福东亚图书馆** 的野外实例：移动操作平台沿书架通道扫描，用 **VLM + 图书馆目录 RAG** 识别中/日/韩书脊标题与索书号，并借 catalog 自动策展训练数据以持续微调感知模型。

## 一句话定义

**带相机的移动机械臂在真实图书馆过道里扫书架——既帮馆员盘点，又把 VLM 最缺的「脏、多语言、破损标签」图像自动标好喂回模型。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RPDF | Robot-Powered Data Flywheel | 本文所属的持续数据–模型闭环框架 |
| VLM | Vision-Language Model | 书脊识别与标注的核心感知模型 |
| RAG | Retrieval-Augmented Generation | 用 sectional catalog 约束 VLM 书候选 |
| RGB-D | RGB + Depth | 腕部 RealSense D435 观测 |
| ADA | Americans with Disabilities Act | 过道宽度约束（≥36 in）影响平台选型 |
| LiDAR | Light Detection and Ranging | 底座 L2 用于过道居中与漂移校正 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Stanford；Toyota Research Institute |
| **平台** | Franka FR3 + TidyBot++ 移动底座（宽约 21 in） |
| **传感** | 腕部 Intel RealSense D435；底座 Unitree L2 LiDAR |
| **控制** | 预定义货架高度序列；每停前进 0.3 m；LiDAR 平面拟合校正漂移 |
| **部署** | 2 周、10 天 × 4 h；**2103** 书架；**26** 次人工干预（<5 min/次） |
| **开源** | **未开源**（[项目页](https://scanford-robot.github.io/) 无代码仓） |

## 与其他页面的关系

- [RPDF 论文页](./paper-scanford-robot-powered-data-flywheel.md) — 框架、实验与 OCR 外溢
- [Data Flywheel](../concepts/data-flywheel.md) — 概念层飞轮
- [人形数据采集地图](../queries/humanoid-robot-data-collection-landscape.md) — 产业对照（遥操作/可穿戴等）

## 参考来源

- [Scanford 项目页](../../sources/sites/scanford.md)
- [arXiv:2511.19647 摘录](../../sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md)

## 推荐继续阅读

- <https://scanford-robot.github.io/>
