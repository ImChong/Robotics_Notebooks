---
type: entity
project_id: modex
tags:
  - paper
  - manipulation
  - contact-rich
status: complete
updated: 2026-11-01
arxiv: "2610.23888"
code: https://github.com/Aurora-Xiong/MoDex
related:
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/dexterous-kinematics.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/modex_arxiv_2610_23888.md
  - ../../sources/repos/modex.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "MoDex（arXiv:2610.23888）：MoDex学习参考轨迹的残差，按物体距离选择一次抓取或顺序交互；未见物体及真机迁移展示潜力，目前主要验证双物体，现实表现仍落后于仿真。"
---

# MoDex（arXiv:2610.23888）

**MoDex**（*MoDex: Towards Universal Multi-Object Dexterous Grasping*，[arXiv:2610.23888](https://arxiv.org/abs/2610.23888)，[代码](https://github.com/Aurora-Xiong/MoDex)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「灵巧、接触与精细操作」组。

## 一句话定义

**MoDex学习参考轨迹的残差，按物体距离选择一次抓取或顺序交互；未见物体及真机迁移展示潜力，目前主要验证双物体，现实表现仍落后于仿真。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoDex | MoDex: Towards Universal Multi-Object Dexterous Grasping | 本文方法 / 基准的简称 |

## 为什么重要

- 物体靠近或分散，抓取策略分别处理。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23888](https://arxiv.org/abs/2610.23888) |
| **开源** | **已开源** |
| **要点** | MoDex学习参考轨迹的残差，按物体距离选择一次抓取或顺序交互；未见物体及真机迁移展示潜力，目前主要验证双物体，现实表现仍落后于仿真。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`Aurora-Xiong/MoDex`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Aurora-Xiong/MoDex
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 MoDex 模型
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

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**物体靠近或分散，抓取策略分别处理：MoDex学习参考轨迹的残差，按物体距离选择一次抓取或顺序交互；未见物体及真机迁移展示潜力，目前主要验证双物体，现实表现仍落后于仿真。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：MoDex学习参考轨迹的残差，按物体距离选择一次抓取或顺序交互；未见物体及真机迁移展示潜力，目前主要验证双物体，现实表现仍落后于仿真。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [modex_arxiv_2610_23888.md](../../sources/papers/modex_arxiv_2610_23888.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23888](https://arxiv.org/abs/2610.23888)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23888)
- [https://github.com/Aurora-Xiong/MoDex](https://github.com/Aurora-Xiong/MoDex)
