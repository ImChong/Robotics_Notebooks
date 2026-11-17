---
type: entity
project_id: paper-touchtwin-2611-17619
tags: [paper,tactile,real2sim,dexterous-manipulation]
status: complete
updated: 2026-11-17
arxiv: "2611.17619"
project: https://touchtwin.github.io/
original:
  - ../../sources/papers/touchtwin-2611-17619.md
sources:
  - ../../sources/papers/touchtwin-2611-17619.md
related:
  - ./xhand-1-pro-u.md
  - ./paper-agentic-teacher-2611-17394.md
  - ./paper-pear-2611-18562.md
summary: "结合人类 RGB-D 视觉与 880 税素触觉手套记录，在 MuJoCo 中以真实接触力约束手—物体回放，生成可扩增的接触数据。"
---

# TouchTwin: Scaling Human Touch Data via Real-to-Sim-to-Real｜TouchTwin：触觉约束的 Real-to-Sim-to-Real 数据扩增

## 一句话定义

**TouchTwin：触觉约束的 Real-to-Sim-to-Real 数据扩增**：结合人类 RGB-D 视觉与 880 税素触觉手套记录，在 MuJoCo 中以真实接触力约束手—物体回放，生成可扩增的接触数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RGB-D | Red Green Blue and Depth | 彩色与深度视觉输入 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触动力学仿真平台 |
| R2S2R | Real-to-Sim-to-Real | 真实到仿真再返回策略应用的数据转移 |

## 为什么重要

纯视觉重建可能在几何上合理，却产生错误抓握力或接触事件；真实触觉可为物理回放和扰动增广提供监督。

## 核心信息

| 项目 | 核查来源 |
|---|---|
| 作者或机构 | 原文作者机构，见 arXiv v1 |
| 原文 | [一手全文/原始新闻稿](https://arxiv.org/html/2611.17619v1) |
| 代码状态 | 未核实与本研究/设备对应的可运行代码仓库 |
| 核查版本 | arXiv:2611.17619v1 |

## 核心原理

同步获取第一视角 RGB-D 与高密度手套压力阵列；构建可仿真的物体与个体化手模型，用真实触觉曲线校准 MuJoCo 抓握回放中的接触时刻与力度。随后扰动抓取姿态、物体属性、力与运动轨迹，生成带接触标签的数据，并以新视角视频作为训练输入。

## 流程总览

```mermaid
flowchart LR
  A[RGB-D 视频与 880-taxel 实测触觉]
  B[可仿真物体与手部模型]
  C[真实触觉约束 MuJoCo 回放]
  D[抓姿与物体物理扰动]
  E[接触力标注与新视角视频]
  F[抓握策略学习]
  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
```

上述流程图是基于原始资料整理的方法主干，不代表已复现作者模型。

## 源码运行时序图

**不适用：** 已核对官方项目页与论文，但尚未确认可运行的官方训练/仿真代码仓库及执行入口，不能由算法图臆造调用链。

## 工程实践

1. 校验 RGB-D、手套各 taxel 与关节状态的时间同步和力单位。
2. 复现时分别记录接触漏检率、抓握力比例、抓握任务成功率和滑落失败模式。
3. 明确区分仿真接触标签评测、仿真策略评测和实体机器人的执行指标。

## 实验与评测

论文报告学习策略的抓握力是对应真实记录的 0.99 倍，在留出示范记录上成功 20/20 次、未见扰动条件下成功 35/36 次；这些统计出自作者的仿真实验。

## 与其他工作对比

与 [XHAND 1 Pro-U](./xhand-1-pro-u.md) 的示教硬件定位不同，TouchTwin 聚焦如何把触觉记录转换成仿真物理约束和可复用训练数据。

## 结论

**结合人类 RGB-D 视觉与 880 税素触觉手套记录，在 MuJoCo 中以真实接触力约束手—物体回放，生成可扩增的接触数据。**

1. 纯视觉轨迹不能直接确定接触力曲线。
2. 应同时报告抓力比例、接触事件一致性与任务成功率。
3. 触觉约束使物理增广样本保留有意义的接触监督。

## 局限与风险

接触参数辨识、手套受力标定、物体形状误差与仿真域随机化会影响真实机器人泛化；公布指标不能当作真机成功率。

## 关联页面

- [xhand 1 pro u](./xhand-1-pro-u.md)
- [paper agentic teacher 2611 17394](./paper-agentic-teacher-2611-17394.md)
- [paper pear 2611 18562](./paper-pear-2611-18562.md)
- [触觉阻抗控制](../methods/tactile-impedance-control.md)
- [数据增广](../methods/generative-data-augmentation.md)
- [embodied-data 路线](../../roadmap/depth-embodied-data.md)

## 参考来源

- [一手原文核对记录](../../sources/papers/touchtwin-2611-17619.md)

## 推荐继续阅读

- [一手正文](https://arxiv.org/html/2611.17619v1)
- [作者/扩展资料](https://touchtwin.github.io/)

