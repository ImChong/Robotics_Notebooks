---
type: entity
project_id: route-guided-cbf-qp-repair
tags:
  - paper
  - reward-model
  - policy-learning
status: complete
updated: 2026-11-04
arxiv: "2611.01713"
project: https://ckck12.github.io/Route-Guied-CBF-QP/
code: https://github.com/Ckck12/Route-Guied-CBF-QP
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/route_guided_cbf_qp_repair_arxiv_2611_01713.md
  - ../../sources/repos/route_guided_cbf_qp_repair.md
  - ../../sources/sites/route-guided-cbf-qp-repair.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Route-Guided CBF-QP Repair（arXiv:2611.01713）：检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。"
---

# Route-Guided CBF-QP Repair（arXiv:2611.01713）

**Route-Guided CBF-QP Repair**（*Route-Guided CBF-QP Repair for Safe Vision-Language-Action Models*，[arXiv:2611.01713](https://arxiv.org/abs/2611.01713)，[项目页](https://ckck12.github.io/Route-Guied-CBF-QP/)，[代码](https://github.com/Ckck12/Route-Guied-CBF-QP)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「奖励学习与策略后训练」组。

## 一句话定义

**检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Route-Guided CBF-QP Repair | Route-Guided CBF-QP Repair for Safe Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 避开障碍后，回到原来的操作路线。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.01713](https://arxiv.org/abs/2611.01713) |
| **开源** | **已开源** |
| **要点** | 检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`Ckck12/Route-Guied-CBF-QP`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Ckck12/Route-Guied-CBF-QP
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Route-Guided CBF-QP Repair 模型
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

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**避开障碍后，回到原来的操作路线：检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [route_guided_cbf_qp_repair_arxiv_2611_01713.md](../../sources/papers/route_guided_cbf_qp_repair_arxiv_2611_01713.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.01713](https://arxiv.org/abs/2611.01713)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.01713)
- [项目页](https://ckck12.github.io/Route-Guied-CBF-QP/)
- [https://github.com/Ckck12/Route-Guied-CBF-QP](https://github.com/Ckck12/Route-Guied-CBF-QP)
