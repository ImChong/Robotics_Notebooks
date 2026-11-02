---
type: entity
project_id: structure-over-reasoning
tags:
  - paper
  - embodied-agent
  - navigation
status: complete
updated: 2026-11-01
arxiv: "2610.25130"
project: https://anonymous.4open.science/r/structure-over-reasoning-PAGE
related:
  - ../concepts/llm-robotics-control-interfaces.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../tasks/vision-language-navigation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/structure_over_reasoning_arxiv_2610_25130.md
  - ../../sources/sites/structure-over-reasoning.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "Structure over Reasoning（arXiv:2610.25130）：把小VLM的方向判断累积到语义地图，再用图搜索和停止门控决定行动；统一预算仿真对照支持结构化设计，最强粗粒度基线差异的区间仍包含零。"
---

# Structure over Reasoning（arXiv:2610.25130）

**Structure over Reasoning**（*Structure over Reasoning: Unified Zero-Shot Navigation with a Small Vision-Language Model*，[arXiv:2610.25130](https://arxiv.org/abs/2610.25130)，[项目页](https://anonymous.4open.science/r/structure-over-reasoning-PAGE)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「具身智能体与导航」组。

## 一句话定义

**把小VLM的方向判断累积到语义地图，再用图搜索和停止门控决定行动；统一预算仿真对照支持结构化设计，最强粗粒度基线差异的区间仍包含零。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Structure over Reasoning | Structure over Reasoning: Unified Zero-Shot Navigation with a Small Vision-Language Model | 本文方法 / 基准的简称 |

## 为什么重要

- 小模型导航，把感知证据交给结构处理。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-01）。项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A["每步输入: 12 视角 RGB + 深度全景, 位姿, 语言指令"] --> B["深度 → 距离扫描 → 占据栅格 (鸟瞰地图)<br/>可达自由空间 St 与测地距离"]
    B --> C["自由空间提取骨架图<br/>节点为走廊/房间中心, 边界节点为 frontier"]
    A --> D["Qwen3-VL-4B 问题一: 逐视角打 0-4 分<br/>该方向看起来像目标吗"]
    D --> E["路线无关语义场 (belief field)<br/>中心化得分投影到 30° 楔形, 对数域回火更新 + 小均匀泄漏"]
    C --> F{"指令含路线子句?<br/>(精细粒度 kr > 0)"}
    F -- "是" --> G["门控路线查询 (CoT): 给出 frontier 列表<br/>模型选出路线所指的 frontier 作为提案"]
    G --> H["测地核重加权场, 按 λ̃ 收缩耦合<br/>(提案只重分配场内质量)"]
    F -- "否 (粗粒度)" --> I["门关闭, 仅语义场驱动"]
    E --> H
    E --> I
    H --> J["动作效用: 场质量 + 提案牵引 + 探索奖励<br/>− 行程代价 − 折返惩罚"]
    I --> J
    J --> K["在基准航点预测器给出的候选中取 argmax 并移动"]
    A --> L["Qwen3-VL-4B 问题二: 最佳视角 + 指令 + 轨迹<br/>此处是否应停止 (一词回答)"]
    L --> M{"终止器阶段一: 轨迹门<br/>当前轨迹得分不低于历史最佳 − δ?"}
    M -- "接受" --> N["停止, 结束 episode"]
    M -- "拒绝" --> K
    K --> O{"终止器阶段二: 剩余预算 T−t > 回到最佳位置所需步数 + 1?"}
    O -- "是" --> A
    O -- "否" --> P["回走到轨迹得分最佳位置并结束"]
```

图依据：论文 §III Method (A-D) / Fig.1 / 式(1)-(9)

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.25130](https://arxiv.org/abs/2610.25130) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 把小VLM的方向判断累积到语义地图，再用图搜索和停止门控决定行动；统一预算仿真对照支持结构化设计，最强粗粒度基线差异的区间仍包含零。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**小模型导航，把感知证据交给结构处理：把小VLM的方向判断累积到语义地图，再用图搜索和停止门控决定行动；统一预算仿真对照支持结构化设计，最强粗粒度基线差异的区间仍包含零。**

1. 开源边界：**项目页未列 GitHub** — 项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。
2. 核心机制：把小VLM的方向判断累积到语义地图，再用图搜索和停止门控决定行动；统一预算仿真对照支持结构化设计，最强粗粒度基线差异的区间仍包含零。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [structure_over_reasoning_arxiv_2610_25130.md](../../sources/papers/structure_over_reasoning_arxiv_2610_25130.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.25130](https://arxiv.org/abs/2610.25130)
- [arXiv PDF](https://arxiv.org/pdf/2610.25130)：已通读方法部分（实验与附录以略读为准）：§III 方法（鸟瞰图与骨架图、路线无关语义场、路线通道与门控耦合、两阶段终止器），§IV 实验与消融

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.25130)
- [项目页](https://anonymous.4open.science/r/structure-over-reasoning-PAGE)
