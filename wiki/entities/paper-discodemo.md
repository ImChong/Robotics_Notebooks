---
type: entity
project_id: discodemo
tags:
  - paper
  - data-collection
  - demonstration
status: complete
updated: 2026-10-29
arxiv: "2610.32093"
project: https://davian-robotics.github.io/DiscoDemo/
code: https://github.com/DAVIAN-Robotics/DiscoDemo
related:
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../methods/imitation-learning.md
  - ../tasks/teleoperation.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/discodemo_arxiv_2610_32093.md
  - ../../sources/repos/discodemo.md
  - ../../sources/sites/discodemo.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "DiscoDemo（arXiv:2610.32093）：模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。"
---

# DiscoDemo（arXiv:2610.32093）

**DiscoDemo**（*DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning*，[arXiv:2610.32093](https://arxiv.org/abs/2610.32093)，[项目页](https://davian-robotics.github.io/DiscoDemo/)，[代码](https://github.com/DAVIAN-Robotics/DiscoDemo)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「示范数据与数据引擎」组。

## 一句话定义

**模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiscoDemo | DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning | 本文方法 / 基准的简称 |

## 为什么重要

- 让强化学习生成多种成功示范。DiscoDemo把强化学习用于数据生成，以逆向课程和多样性目标扩展覆盖，再训练同一个视觉策略；四项仿真和真机任务提供了较清楚的对照。
- 开源结论：**已开源**（步骤 2.5，2026-10-29）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.32093](https://arxiv.org/abs/2610.32093) |
| **开源** | **已开源** |
| **要点** | 模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。 |
| **文内指标** | 公众号转述：新增机制是让成功门控的解法多样性参与生成器训练，再把收益落实到下游视觉模仿，而非仅优化RL控制器自身回报；过强多样性也会损害任务能力。 |

## 核心流程图

以下概括本页所述的核心关系；具体实现与评测条件以原论文为准。

```mermaid
flowchart TD
  A["逆向课程与成功门控多样性"] --> B["强化学习训练示范生成器"]
  B --> C["多样成功示范"]
  C --> D["训练视觉模仿策略"]
  D --> E["下游任务评估"]
```

## 源码运行时序图

以下为按仓库入口（`DAVIAN-Robotics/DiscoDemo`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as DAVIAN-Robotics/DiscoDemo
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 DiscoDemo 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：新增机制是让成功门控的解法多样性参与生成器训练，再把收益落实到下游视觉模仿，而非仅优化RL控制器自身回报；过强多样性也会损害任务能力。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让强化学习生成多种成功示范：新增机制是让成功门控的解法多样性参与生成器训练，再把收益落实到下游视觉模仿，而非仅优化RL控制器自身回报；过强多样性也会损害任务能力。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [imitation-learning](../methods/imitation-learning.md)
- [teleoperation](../tasks/teleoperation.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [discodemo_arxiv_2610_32093.md](../../sources/papers/discodemo_arxiv_2610_32093.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.32093](https://arxiv.org/abs/2610.32093)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.32093)
- [项目页](https://davian-robotics.github.io/DiscoDemo/)
- [https://github.com/DAVIAN-Robotics/DiscoDemo](https://github.com/DAVIAN-Robotics/DiscoDemo)
