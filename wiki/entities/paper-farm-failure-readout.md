---
type: entity
tags: [paper, world-model, safety, deployment]
status: complete
updated: 2026-09-11
arxiv: "2609.11445"
code: https://github.com/HaoranPei-casia/FARM
related:
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/dexterous-wm-humanoid-14-papers-technology-map.md
sources:
  - ../../sources/papers/farm-failure-readout_arxiv_2609_11445.md
  - ../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md
summary: "冻结世界模型预测状态 + 轻量 readout 做失败检测；7 任务五折 OOF pooled AUROC 85.68。"
---

# FARM（arXiv:2609.11445）

**FARM**（[FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model](https://arxiv.org/abs/2609.11445)）来自 [具身智能小站 14 篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)。冻结世界模型预测状态 + 轻量 readout 做失败检测；7 任务五折 OOF pooled AUROC 85.68。

## 一句话定义

**不另训监控器，从冻结机器人世界模型的内部预测态读出失败信号。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **灵巧手 / 世界模型 / 人形控制 / VLA** 主线之一。
- 开源状态：**已开源**（步骤 2.5 核查，2026-09-11）。
- 与 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.11445](https://arxiv.org/abs/2609.11445) |
| **项目页** | https://github.com/HaoranPei-casia/FARM |
| **代码/资源** | https://github.com/HaoranPei-casia/FARM |
| **开源** | **已开源** |
| **文内指标** | 33,985 参数 readout；pooled AUROC/AUPRC 85.68/88.59；跨 PIPER X / SO-101 / Franka 测试迁移。 |


## 源码运行时序图

```mermaid
sequenceDiagram
  participant U as 用户/脚本
  participant R as 官方仓库入口
  participant M as 模型/规划器
  participant E as 仿真或真机环境
  U->>R: clone + 安装依赖
  U->>M: 加载配置/权重
  U->>E: rollout / 规划 / 控制
  M-->>E: 动作或轨迹
  E-->>U: 成功率/指标日志
```


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 33,985 参数 readout；pooled AUROC/AUPRC 85.68/88.59；跨 PIPER X / SO-101 / Franka 测试迁移。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准（[参考来源](#参考来源)）。

## 结论

**FARM 适合作为本期「已开源」边界下的快速索引页，部署前请核对仓库/README 可运行性。**

1. 核心贡献：不另训监控器，从冻结机器人世界模型的内部预测态读出失败信号。
2. 开源结论：**已开源** — 以项目页实际链接为准（入库日 2026-09-11）。
3. 横向对照见 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [farm-failure-readout_arxiv_2609_11445.md](../../sources/papers/farm-failure-readout_arxiv_2609_11445.md)
- [wechat 14篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)
- [arXiv:2609.11445](https://arxiv.org/abs/2609.11445)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.11445)
- [项目页/资源](https://github.com/HaoranPei-casia/FARM)
- [代码/资源](https://github.com/HaoranPei-casia/FARM)
