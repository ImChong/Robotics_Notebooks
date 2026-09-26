---
type: entity
tags: [paper, vla, agent, physical-ai, long-horizon]
status: complete
updated: 2026-09-26
arxiv: "2609.29166"
code: https://github.com/Darwin-Agent/HarnessPAI
related:
  - ../overview/embodied-research-12-papers-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/harnesspai_arxiv_2609_29166.md
  - ../../sources/repos/harnesspai.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md
summary: "HarnessPAI（arXiv:2609.29166）：单次执行固定程序、跨轮改写 harness 并积累技能；Darwin-Agent/HarnessPAI 已开源。"
---

# HarnessPAI

**HarnessPAI**（*An Evolving Harness for Physical AI*，[arXiv:2609.29166](https://arxiv.org/abs/2609.29166)，[代码](https://github.com/Darwin-Agent/HarnessPAI)，[项目页](https://darwin-agent.github.io/HarnessPAI)）收录自 [具身智能小站 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)。

## 一句话定义

**Physical AI 把高层推理预算挪到多轮试验：执行时跑固定程序，失败后再演化 harness 与技能库。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| IL | Imitation Learning | 模仿学习 |
| SR | Success Rate | 任务成功率 |
| WM | World Model | 世界模型 |

## 为什么重要

- 纳入 [12 篇具身研究清单](../../wiki/overview/embodied-research-12-papers-technology-map.md) 主线，与同期 VLA / 接触 / 规划 / 安全论文可横向对照。
- 公众号强调的可操作读法：先看 **任务信息需求**（如 PolyUMI 旋灯泡仍以视觉最优）与 **评测口径**（如 Self-Adaptive 多 trial、BeyondRetarget 仿真片段非真机 SR）。
- 开源状态（步骤 2.5）：**已开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.29166](https://arxiv.org/abs/2609.29166) |
| **项目页** | https://darwin-agent.github.io/HarnessPAI |
| **代码** | https://github.com/Darwin-Agent/HarnessPAI |
| **开源** | **已开源** |

## 实验与评测（公众号口径）

- 指标与消融以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md) 与 **原文 PDF** 为准；本页不复制整表。
- 读复现前先核对：样本规模、是否仿真/真机、是否允许多次 attempt。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户
  participant R as 官方仓库
  participant M as 训练/推理入口
  participant E as 仿真或真机
  U->>R: clone + 依赖安装
  U->>M: 配置与权重
  M->>E: rollout / 控制
  E-->>U: 指标日志
```


## 结论

**总判：HarnessPAI 适合作为「Physical AI 把高层推理预算挪到多轮试验：执行时跑固定程序，失败后再演…」方向的入口页；细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md) 对照选型，避免与同名不同 arXiv 的工作混淆（如 RAPID vs RAPID-VLM-RL）。
2. 开源为 **已开源** 时优先从项目页 Code 区核实，再写复现计划。
3. 长程 / 部署类条目（AdaHVLA、HarnessPAI、Self-Adaptive VLA）同时记录 **成功率定义** 与 **失败恢复预算**。

## 关联页面

- [具身研究 12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md)
- [Manipulation](../tasks/manipulation.md)
- [VLA](../methods/vla.md)

## 参考来源

- [harnesspai 论文归档](../../sources/papers/harnesspai_arxiv_2609_29166.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)

## 推荐继续阅读

- [arXiv:2609.29166](https://arxiv.org/abs/2609.29166)
- [项目页](https://darwin-agent.github.io/HarnessPAI)
