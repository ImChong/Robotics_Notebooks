---
type: entity
tags:
  - paper
  - code-as-policy
  - vla
  - manipulation
  - local-llm
status: complete
updated: 2026-09-23
arxiv: "2609.26499"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../entities/paper-pai-2209-07753-codeaspolicies.md
  - ./paper-industrialvla-bench.md
  - ../overview/collab-wm-12-papers-technology-map.md
sources:
  - ../../sources/papers/agentic-coding-manipulation_arxiv_2609_26499.md
  - ../../sources/sites/agentic-coding-manipulation.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md
summary: "Agentic Coding Agent（arXiv:2609.26499）：本地 Qwen3.8-27B + coding-agent harness 控制 UR3e，自行编写/调试代码；9 玩具任务 45 次试验完成 30 次泛化。"
---

# Agentic Coding Agent（arXiv:2609.26499）

**Agentic Coding Agent**（*Generalizing Manipulation Skills with a Local Coding Agent*，[arXiv:2609.26499](https://arxiv.org/abs/2609.26499)，[项目页](https://rtalwar2.github.io/agentic-coding-for-robot-manipulation/)）来自 [具身智能小站 12 篇盘点](../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)。

## 一句话定义

**本地 Qwen3.8-27B + coding-agent harness 控制 UR3e，自行编写/调试代码；9 玩具任务 45 次试验完成 30 次泛化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉–语言模型 |
| SDK | Software Development Kit | 机器人控制软件开发包 |
| TCP | Tool Center Point | 工具中心点 |
| HSV | Hue Saturation Value | 颜色空间阈值分割 |

## 为什么重要

- 固定动作接口或训练策略换任务成本高；探本地 VLM 能否用已文档化 procedure 泛化到新颜色/尺寸/组合。
- 开源结论：**未开源**（步骤 2.5，2026-09-23）。
- 与 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.26499](https://arxiv.org/abs/2609.26499) |
| **开源** | **未开源** |
| **要点** | Platform HTTP 服务（9 个运动/感知原语 + 安全包络）+ Skills 文档 + agent workspace 写脚本执行。 |
| **文内指标** | 15.8h 机器人时间；成功复做后时长与 token 约减半；平台/agent 代码明确不发布。 |

## 源码运行时序图

**不适用**（入库日模型/训练权重未公开，或仅有 API/CLI 封装；无可运行官方训练/推理入口。）

## 实验与评测

- 15.8h 机器人时间；成功复做后时长与 token 约减半；平台/agent 代码明确不发布。
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**Agentic coding 证明 procedure+skills 可泛化，但感知误读与无 body schema 是主要失败源；非开箱部署方案。**

1. 开源边界：**未开源** — 以项目页实际链接为准（入库日 2026-09-23）。
2. 核心机制：Platform HTTP 服务（9 个运动/感知原语 + 安全包络）+ Skills 文档 + agent workspace 写脚本执行。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [paper-pai-2209-07753-codeaspolicies](../entities/paper-pai-2209-07753-codeaspolicies.md)
- [paper-industrialvla-bench](./paper-industrialvla-bench.md)

## 参考来源

- [agentic-coding-manipulation_arxiv_2609_26499.md](../../sources/papers/agentic-coding-manipulation_arxiv_2609_26499.md)
- [wechat_embodied_station_12_papers_collab_wm_2026-09-23.md](../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)
- [arXiv:2609.26499](https://arxiv.org/abs/2609.26499)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.26499)
- [项目页](https://rtalwar2.github.io/agentic-coding-for-robot-manipulation/)

