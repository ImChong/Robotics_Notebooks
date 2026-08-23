---
type: entity
tags: ['paper', 'dexterous-manipulation', 'retargeting', 'force-control', 'teleoperation']
status: complete
updated: 2026-08-23
arxiv: "2608.15560"
related:
  - ../concepts/motion-retargeting-pipeline.md
  - ../tasks/manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../overview/vla-robustness-9-papers-technology-map.md
sources:
  - ../../sources/papers/reforce_arxiv_2608_15560.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md
  - ../../sources/sites/reforce.md
summary: "ReForce（arXiv:2608.15560，UCSD 等）：力觉重定向——运动学 residual + 仿真力跟踪器；纸杯/夹钳接触任务；截至入库日未开源。"
---

# ReForce

**ReForce: Learning Force-aware Retargeting for Dexterous Manipulation**（[arXiv:2608.15560](https://arxiv.org/abs/2608.15560)，[项目页](https://wuyuhang-eai.github.io/reforce/)）——UC San Diego（Xiaolong Wang 组）等。

## 一句话定义

**灵巧操作的数据迁移核心不只是像人动，还要像人接触。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACT | Action Chunking Transformer | 模仿学习策略基线 |
| FSR | Force Sensing Resistor | 指尖力传感 |
| GMR | General Motion Retargeting | 运动学重定向基线 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-23 九篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md) 的「长视野 VLA → 动作分布 → 真实交互」主线。
- 开源状态（入库日）：**未开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | UC San Diego（Xiaolong Wang 组）等 |
| **出处** | arXiv:2608.15560（2026-08） |
| **开源** | **未开源** |

### 流程总览

```mermaid
flowchart LR
  obs[观测/指令] --> core[核心方法模块]
  core --> act[动作/规划输出]
  act --> rob[仿真或真机闭环]
```

## 评测

| 项 | 内容 |
|----|------|
| **主结果** | 仿真与真机纸杯抓取、夹钳操作；降低力跟踪误差并增强多指接触参与。 |

- 数据出处：[ingest 摘录](../../sources/papers/reforce_arxiv_2608_15560.md)。

## 结论

**力觉重定向应作为人类示范到机器人执行之间的独立接口层。**

- 在运动学重定向上预测力 residual
- 大规模仿真交互训练通用 force tracker
- 支持在线力觉遥操作与离线数据翻译
- Manus 手套 + FSR 真机栈
- 项目页截至入库日无 GitHub

## 源码运行时序图

**不适用**（截至 **2026-08-23**）：项目页未列可运行代码仓库。

## 与其他页面的关系

- [motion-retargeting-pipeline](../concepts/motion-retargeting-pipeline.md)
- [manipulation](../tasks/manipulation.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [vla-robustness-9-papers-technology-map](../overview/vla-robustness-9-papers-technology-map.md)

## 参考来源

- [reforce_arxiv_2608_15560](../../sources/papers/reforce_arxiv_2608_15560.md)
- [reforce](../../sources/sites/reforce.md)
- [wechat_embodied_station_9_papers_open_source_2026-08-23](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md)

## 推荐继续阅读

- [arXiv:2608.15560](https://arxiv.org/abs/2608.15560)
- [项目页](https://wuyuhang-eai.github.io/reforce/)
