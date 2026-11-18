---
type: entity
project_id: driving-sgdrive
tags: [paper, autonomous-driving, vla]
status: complete
updated: 2026-11-18
arxiv: "2601.05640"
code: https://github.com/LogosIntelligenceLab/SGDrive
summary: "SGDrive：场景、关键交通参与者和短期目标的分层认知。在VLM中引入场景/agent/goal分层世界知识查询，通过block-wise attention避免信息泄漏，世界表征作为条件送入DiT扩散规划器，并可进行强化微调。"
original:
  - ../../sources/papers/ad_top10_2026_sgdrive.md
sources:
  - ../../sources/papers/ad_top10_2026_sgdrive.md
  - ../../sources/repos/ad_top10_2026_sgdrive.md
  - ../../sources/blogs/wechat_shenlan_ai_autonomous_driving_top10_2026.md
related:
  - ../overview/autonomous-driving-open-source-top10-2026.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
---

# SGDrive：场景、关键交通参与者和短期目标的分层认知

**英文：** SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving

## 一句话定义

在VLM中引入场景/agent/goal分层世界知识查询，通过block-wise attention避免信息泄漏，世界表征作为条件送入DiT扩散规划器，并可进行强化微调。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLM | Vision-Language Model | 场景理解骨干 |
| E2E | End-to-End | 从观测到规划的端到端模型 |
| RL | Reinforcement Learning | 奖励信号驱动的优化 |
| VLA | Vision-Language-Action | 视觉语言动作联合学习 |

## 为什么重要

场景、关键交通参与者和短期目标的分层认知专门针对单纯模仿驾驶轨迹难以解释或处理长尾交互的限制，提供可核查的论文、权重或代码入口。

## 核心原理

在VLM中引入场景/agent/goal分层世界知识查询，通过block-wise attention避免信息泄漏，世界表征作为条件送入DiT扩散规划器，并可进行强化微调。

## 流程总览

```mermaid
flowchart LR
  Image[驾驶视觉与导航] --> VLM[InternVL3]
  VLM --> Scene[Scene查询]
  VLM --> Agent[Agent查询]
  VLM --> Goal[Goal查询]
  Scene --> Mask[分块注意力]
  Agent --> Mask
  Goal --> Mask
  Mask --> DiT[扩散轨迹规划]
  DiT --> Traj[未来轨迹]
```

此流程根据 [原始论文记录](../../sources/papers/ad_top10_2026_sgdrive.md)，非跨项目通用套图。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  actor User as 开发者
  participant Script as scripts/training/run_sgdrive_train_multi_node_2b.sh
  participant First as World queries
  participant Second as DiT planner
  participant Output as NAVSIM evaluation
  User->>Script: 使用官方README所列入口
  Script->>First: 加载样本/模型与配置
  First->>Second: 提供训练/推理输入
  Second->>Output: 生成轨迹或评价数据
  Output-->>User: 输出结果
```

按官方README和 [源码入口记录](../../sources/repos/ad_top10_2026_sgdrive.md) 核对；**未在本次 ingest 执行完整 GPU 推理或训练**。

## 评测与指标

官方NAVSIM v1 navtest报告VLM 85.5、IL 87.4、RL 91.1 PDMS；v2另用EPDMS指标。

## 工程实践

官方仓库：https://github.com/LogosIntelligenceLab/SGDrive

1. ```sh scripts/training/run_sgdrive_train_multi_node_2b.sh```
2. ```sh scripts/training/run_sgdrive_train_multi_node_2b_rl.sh```
3. ```sh scripts/evaluation/run_sgdrive_agent_pdm_score_evaluation.sh```

## 结论

**场景、关键交通参与者和短期目标的分层认知** 是驾驶研究中一条具有独立问题设定的复现路线。

1. 先使用官方公开checkpoint与现有评测脚本建立基线。
2. 明确划分论文提出的完整系统和当前代码实际已开放的部分。
3. 记录数据、版本、硬件与指标，避免不同评测设定的数字错比。

## 局限与风险

部分权重仍有coming soon标记，需核对HF可得资产；训练缓存需求可达1–2TB；无法由PDMS推导真实道路安全。

真实上路安全和合规均不能由离线分数、仿真结果或开源状态单独证明。

## 关联页面

- [2026 自动驾驶开源项目技术地图](../overview/autonomous-driving-open-source-top10-2026.md)
- [VLA](../methods/vla.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [论文核对记录](../../sources/papers/ad_top10_2026_sgdrive.md)
- [官方源码入口](../../sources/repos/ad_top10_2026_sgdrive.md)
- [深蓝AI策展原文](../../sources/blogs/wechat_shenlan_ai_autonomous_driving_top10_2026.md)

## 推荐继续阅读

- [原论文](https://arxiv.org/html/2601.05640)
- [官方代码](https://github.com/LogosIntelligenceLab/SGDrive)
