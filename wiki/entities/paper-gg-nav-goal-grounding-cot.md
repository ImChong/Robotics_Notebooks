---
type: entity
project_id: gg-nav-goal-grounding-cot
tags:
  - paper
  - navigation
  - chain-of-thought
  - foundation-model
  - unitree-go2
status: complete
updated: 2026-10-14
arxiv: "2610.14541"
project: https://duangzhu.github.io/GG-Nav-Project-Page/
code: https://github.com/DuangZhu/GG_Nav
related:
  - ../tasks/zero-shot-object-navigation.md
  - ../tasks/vision-language-navigation.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/gg_nav_goal_grounding_cot_arxiv_2610_14541.md
  - ../../sources/repos/gg_nav_goal_grounding_cot.md
  - ../../sources/sites/gg-nav-goal-grounding-cot.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "GG-Nav（arXiv:2610.14541）：结构化推理先把中间帧转为目标证据，再结合当前观察决定行动或追问；覆盖目标导航基准与 Go2 真机。"
---

# GG-Nav（arXiv:2610.14541）

**GG-Nav**（*GG-Nav: Goal-oriented Grounding Chain-of-Thought Elicits Reasoning in Navigation Foundation Models*，[arXiv:2610.14541](https://arxiv.org/abs/2610.14541)，[项目页](https://duangzhu.github.io/GG-Nav-Project-Page/)，[代码](https://github.com/DuangZhu/GG_Nav)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「导航、足式与规划」组。

## 一句话定义

**结构化推理先把中间帧转为目标证据，再结合当前观察决定行动或追问；覆盖目标导航基准与 Go2 真机。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CoT | Chain-of-Thought | 思维链 |
| ObjNav | Object Navigation | 目标导航 |
| NFM | Navigation Foundation Model | 导航基础模型 |

## 为什么重要

- 导航基础模型常漏看目标或停错位置；把沿途观察积累成证据可减少此类失败。
- 开源结论：**已开源**（步骤 2.5，2026-10-14）。仓库可达。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14541](https://arxiv.org/abs/2610.14541) |
| **开源** | **已开源** |
| **要点** | goal-grounding CoT；行动 / 追问两种输出。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

以下为按仓库入口（`DuangZhu/GG_Nav`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as DuangZhu/GG_Nav
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 GG-Nav 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**导航推理的证据链应显式化，便于诊断漏看与停止错误。**

1. 开源边界：**已开源** — 仓库可达。
2. 核心机制：goal-grounding CoT；行动 / 追问两种输出。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [zero-shot-object-navigation](../tasks/zero-shot-object-navigation.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [gg_nav_goal_grounding_cot_arxiv_2610_14541.md](../../sources/papers/gg_nav_goal_grounding_cot_arxiv_2610_14541.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14541](https://arxiv.org/abs/2610.14541)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14541)
- [项目页](https://duangzhu.github.io/GG-Nav-Project-Page/)
- [https://github.com/DuangZhu/GG_Nav](https://github.com/DuangZhu/GG_Nav)
