---
type: entity
tags: [paper, open-vocabulary-navigation, active-perception, spatial-semantics, zero-shot]
status: complete
updated: 2026-08-19
arxiv: "2608.12707"
related:
  - ../tasks/vision-language-navigation.md
  - ./paper-humanoidvln.md
  - ../methods/vla.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ./paper-seeker.md
sources:
  - ../../sources/papers/sap_nav_arxiv_2608_12707.md
  - ../../sources/repos/sap-nav.md
  - ../../sources/blogs/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md
summary: "SAP-Nav（arXiv:2608.12707）：在线 zero-shot 层级 OVON；Queryable Spatial-Semantic Representation + Active Viewpoint Verification；LangMap/HM3D-OVON SOTA。实现待发布。"
---

# SAP-Nav：开词汇导航要主动补空间证据

**SAP-Nav**（*Spatial Semantic Representation Meets Active Perception for Hierarchical Open-Vocabulary Object Navigation*；[arXiv:2608.12707](https://arxiv.org/abs/2608.12707)，[项目页](https://xuetongpei.github.io/SAP-Nav/)，[仓库](https://github.com/XuetongPei/SAP-Nav)）是完全 **在线、zero-shot** 的层级 **开词汇物体导航** 框架：不靠离线地图，边走边建 **Queryable Spatial-Semantic Representation**。

## 一句话定义

**开词汇导航的瓶颈常不是认词，而是当前视角有没有足够空间证据——不够就主动换视点再验证。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OVON | Open-Vocabulary Object Navigation | 自由语言指定目标的物体导航 |
| QSSR | Queryable Spatial-Semantic Representation | 可查询的增量空间–语义表征 |
| AVV | Active Viewpoint Verification | 主动视点验证目标 |
| SR | Success Rate | 导航成功率 |
| HM3D | Habitat-Matterport 3D | 常用 OVON 仿真数据集 |

## 为什么重要

- **层级语言：** 场景 / 房间 / 区域 / 实例线索同时出现，需要结构化 spatial grounding。
- **部分观测：** 单帧认词不够，必须 **主动感知** 补证据。
- **真机验证：** 摘要报告真实机器人实验，不是纯 sim 刷榜。

## 核心信息

| 项 | 内容 |
|----|------|
| **出处** | arXiv:2608.12707（2026-08） |
| **设定** | 完全在线、zero-shot、无离线地图 |
| **结果** | LangMap / HM3D-OVON 整体最好；region-level SR **+12.2%** vs 训练式方法 |
| **开源（截至 2026-08-19）** | **待发布**（仓内 README：code will be released soon） |

## 核心原理

```mermaid
flowchart LR
  explore["主动探索房间视角"]
  qssr["Queryable Spatial-Semantic Map"]
  avv["Active Viewpoint Verification"]
  goal["语言目标确认"]
  explore --> qssr --> avv --> goal
  avv -->|"证据不足"| explore
```

## 工程实践

| 项 | 建议 |
|----|------|
| 源码运行时序图 | **不适用**（导航代码未发布） |
| 读表 | 分开看 region-level 与 instance-level SR |
| 对照 | 与 [HumanoidVLN](./paper-humanoidvln.md) 的 **执行层** 问题不同：本文在 **证据采集** |

## 结论

**SAP-Nav 把 OVON 从「识别词表」拉回「在线空间推理 + 主动看」。**

1. **Queryable map** — 已探索位置也能发起语义查询。
2. **AVV 是核心** — 视角不够就移动，而不是硬猜。
3. **+12.2% region SR** — 说明层级 grounding 收益在区域级最明显。
4. **代码未开** — 真机细节需等官方实现。

## 局限与风险

- 实现未发布，无法复现 LangMap/HM3D 数字。
- Zero-shot 在线建图算力与延迟未在实体页量化。
- 与需要全局先验地图的方法公平性需读原文协议。

## 实验与评测

LangMap / HM3D-OVON **整体最好**；region-level SR 相对训练式方法 **+12.2%**；摘要含真实机器人可行性实验。

## 与其他工作对比

相对离线语义地图方法：本文 **完全在线 zero-shot**。相对 [HumanoidVLN](./paper-humanoidvln.md)：SAP-Nav 在 **物体级 OVON**，HumanoidVLN 在 **类人 VLN 协议**。

## 关联页面

- [世界模型与真实执行 10 篇技术地图](../overview/world-model-exec-10-papers-technology-map.md)
- [Vision-Language Navigation](../tasks/vision-language-navigation.md)
- [HumanoidVLN](./paper-humanoidvln.md)
- [Seeker](./paper-seeker.md)
- [VLA](../methods/vla.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — ③ 在线语义建图层：可查询空间–语义表征 + 主动视点验证

## 参考来源

- [SAP-Nav 论文摘录](../../sources/papers/sap_nav_arxiv_2608_12707.md)
- [仓库归档](../../sources/repos/sap-nav.md)
- [具身智能小站 10 篇盘点（2026-08-19）](../../sources/blogs/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md)

## 推荐继续阅读

- [SAP-Nav 项目页](https://xuetongpei.github.io/SAP-Nav/)
- [arXiv:2608.12707](https://arxiv.org/abs/2608.12707)
