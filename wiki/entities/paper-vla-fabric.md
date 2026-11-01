---
type: entity
project_id: vla-fabric
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-11-01
arxiv: "2610.23018"
project: https://nebulis-lab.com/VLA-Fabric/
code: https://github.com/NEBULIS-Lab/VLA-Fabric
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../tasks/bimanual-manipulation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/vla_fabric_arxiv_2610_23018.md
  - ../../sources/repos/vla_fabric.md
  - ../../sources/sites/vla-fabric.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "VLA-Fabric（arXiv:2610.23018）：VLA-Fabric保留各臂独立感知和动作输出，在模型内部学习跨策略通信；两种架构及三、四臂任务验证协作，强调通信机制需要随骨干重新实现。"
---

# VLA-Fabric（arXiv:2610.23018）

**VLA-Fabric**（*VLA-Fabric: Learning Interaction for Composable Multi-Arm VLA Agents*，[arXiv:2610.23018](https://arxiv.org/abs/2610.23018)，[项目页](https://nebulis-lab.com/VLA-Fabric/)，[代码](https://github.com/NEBULIS-Lab/VLA-Fabric)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「VLA 与策略学习」组。

## 一句话定义

**VLA-Fabric保留各臂独立感知和动作输出，在模型内部学习跨策略通信；两种架构及三、四臂任务验证协作，强调通信机制需要随骨干重新实现。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA-Fabric | VLA-Fabric: Learning Interaction for Composable Multi-Arm VLA Agents | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 每只机械臂一个VLA，显式交换协作信息。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23018](https://arxiv.org/abs/2610.23018) |
| **开源** | **已开源** |
| **要点** | VLA-Fabric保留各臂独立感知和动作输出，在模型内部学习跨策略通信；两种架构及三、四臂任务验证协作，强调通信机制需要随骨干重新实现。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`NEBULIS-Lab/VLA-Fabric`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as NEBULIS-Lab/VLA-Fabric
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 VLA-Fabric 模型
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

**每只机械臂一个VLA，显式交换协作信息：VLA-Fabric保留各臂独立感知和动作输出，在模型内部学习跨策略通信；两种架构及三、四臂任务验证协作，强调通信机制需要随骨干重新实现。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：VLA-Fabric保留各臂独立感知和动作输出，在模型内部学习跨策略通信；两种架构及三、四臂任务验证协作，强调通信机制需要随骨干重新实现。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [vla_fabric_arxiv_2610_23018.md](../../sources/papers/vla_fabric_arxiv_2610_23018.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23018](https://arxiv.org/abs/2610.23018)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23018)
- [项目页](https://nebulis-lab.com/VLA-Fabric/)
- [https://github.com/NEBULIS-Lab/VLA-Fabric](https://github.com/NEBULIS-Lab/VLA-Fabric)
