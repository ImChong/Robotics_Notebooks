---
type: entity
project_id: natural-language-to-nav2
tags:
  - paper
  - navigation
  - planning
status: complete
updated: 2026-10-29
arxiv: "2610.32633"
project: https://a56239637-dot.github.io/Natural-Language-to-Nav2-Project-Page/
code: https://github.com/a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation
related:
  - ../tasks/locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../concepts/sim2real.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/natural_language_to_nav2_arxiv_2610_32633.md
  - ../../sources/repos/natural_language_to_nav2.md
  - ../../sources/sites/natural-language-to-nav2.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "Natural-Language to Nav2（arXiv:2610.32633）：将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。"
---

# Natural-Language to Nav2（arXiv:2610.32633）

**Natural-Language to Nav2**（*Natural-Language to Nav2: Operational Policies for Costmap-Based Navigation*，[arXiv:2610.32633](https://arxiv.org/abs/2610.32633)，[项目页](https://a56239637-dot.github.io/Natural-Language-to-Nav2-Project-Page/)，[代码](https://github.com/a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「导航、规划与定位」组。

## 一句话定义

**将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Natural-Language to Nav2 | Natural-Language to Nav2: Operational Policies for Costmap-Based Navigation | 本文方法 / 基准的简称 |

## 为什么重要

- 把运营规则写进Nav2代价地图。
- 开源结论：**已开源**（步骤 2.5，2026-10-29）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.32633](https://arxiv.org/abs/2610.32633) |
| **开源** | **已开源** |
| **要点** | 将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Natural-Language to Nav2 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把运营规则写进Nav2代价地图：将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [sim2real](../concepts/sim2real.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [natural_language_to_nav2_arxiv_2610_32633.md](../../sources/papers/natural_language_to_nav2_arxiv_2610_32633.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.32633](https://arxiv.org/abs/2610.32633)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.32633)
- [项目页](https://a56239637-dot.github.io/Natural-Language-to-Nav2-Project-Page/)
- [https://github.com/a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation](https://github.com/a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation)
