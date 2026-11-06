---
type: entity
project_id: astra-on-robomme
tags:
  - paper
  - manipulation
  - imitation-learning
status: complete
updated: 2026-11-06
arxiv: "2611.06184"
project: https://bingaochen.github.io/Astra-on-RoboMME/
code: https://github.com/bingaochen/Astra-on-RoboMME
related:
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/astra_on_robomme_arxiv_2611_06184.md
  - ../../sources/repos/astra_on_robomme.md
  - ../../sources/sites/astra-on-robomme.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
summary: "Astra on RoboMME（arXiv:2611.06184）：小模型检查子任务进度，必要时才请 Astra 看历史、派发下一步。"
---

# Astra on RoboMME（arXiv:2611.06184）

**Astra on RoboMME**（*Can Astra Solve RoboMME without Breaking the Bank?*，[arXiv:2611.06184](https://arxiv.org/abs/2611.06184)，[项目页](https://bingaochen.github.io/Astra-on-RoboMME/)，[代码](https://github.com/bingaochen/Astra-on-RoboMME)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「操作策略与数据」组。

## 一句话定义

**小模型检查子任务进度，必要时才请 Astra 看历史、派发下一步。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Astra on RoboMME | Can Astra Solve RoboMME without Breaking the Bank? | 本文方法 / 基准的简称 |

## 为什么重要

- 记忆任务里，什么时候值得再问一次大模型。RoboMME 的 800 次仿真试验里，每局平均调用 3.63 次，成功率为 79.13%。
- 开源结论：**已开源**（步骤 2.5，2026-11-06）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.06184](https://arxiv.org/abs/2611.06184) |
| **开源** | **已开源** |
| **要点** | 小模型检查子任务进度，必要时才请 Astra 看历史、派发下一步。 |
| **文内指标** | 公众号转述：RoboMME 的 800 次仿真试验里，每局平均调用 3.63 次，成功率为 79.13%。 |

## 源码运行时序图

以下为按仓库入口（`bingaochen/Astra-on-RoboMME`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as bingaochen/Astra-on-RoboMME
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Astra on RoboMME 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：RoboMME 的 800 次仿真试验里，每局平均调用 3.63 次，成功率为 79.13%。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**记忆任务里，什么时候值得再问一次大模型：RoboMME 的 800 次仿真试验里，每局平均调用 3.63 次，成功率为 79.13%。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：小模型检查子任务进度，必要时才请 Astra 看历史、派发下一步。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [imitation-learning](../methods/imitation-learning.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [astra_on_robomme_arxiv_2611_06184.md](../../sources/papers/astra_on_robomme_arxiv_2611_06184.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.06184](https://arxiv.org/abs/2611.06184)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.06184)
- [项目页](https://bingaochen.github.io/Astra-on-RoboMME/)
- [https://github.com/bingaochen/Astra-on-RoboMME](https://github.com/bingaochen/Astra-on-RoboMME)
