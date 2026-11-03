---
type: entity
project_id: contact-imitation-interface
tags:
  - paper
  - humanoid
  - dexterous
status: complete
updated: 2026-11-03
arxiv: "2610.41028"
project: https://nus-lins-lab.github.io/CIMIWeb/
code: https://github.com/ariszxxu/CIMI
related:
  - ../tasks/locomotion.md
  - ../concepts/motion-retargeting.md
  - ../concepts/dexterous-kinematics.md
  - ../concepts/contact-rich-manipulation.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/contact_imitation_interface_arxiv_2610_41028.md
  - ../../sources/repos/contact_imitation_interface.md
  - ../../sources/sites/contact-imitation-interface.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "Contact Imitation Interface（arXiv:2610.41028）：CIMI 通过共享接触几何、指尖视图和优先接触对齐的重定向，连接人类示范与灵巧手控制。"
---

# Contact Imitation Interface（arXiv:2610.41028）

**Contact Imitation Interface**（*Contact Imitation Interface:Transferring Dexterous Skills by Showing How Contact Unfolds*，[arXiv:2610.41028](https://arxiv.org/abs/2610.41028)，[项目页](https://nus-lins-lab.github.io/CIMIWeb/)，[代码](https://github.com/ariszxxu/CIMI)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「人形与灵巧控制」组。

## 一句话定义

**CIMI 通过共享接触几何、指尖视图和优先接触对齐的重定向，连接人类示范与灵巧手控制。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Contact Imitation Interface | Contact Imitation Interface:Transferring Dexterous Skills by Showing How Contact Unfolds | 本文方法 / 基准的简称 |

## 为什么重要

- 示范迁移要保留接触过程。分层策略分别规划接触表面轨迹和腕指协调。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.41028](https://arxiv.org/abs/2610.41028) |
| **开源** | **已开源** |
| **要点** | CIMI 通过共享接触几何、指尖视图和优先接触对齐的重定向，连接人类示范与灵巧手控制。 |
| **文内指标** | 公众号转述：分层策略分别规划接触表面轨迹和腕指协调。 |

## 源码运行时序图

以下为按仓库入口（`ariszxxu/CIMI`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as ariszxxu/CIMI
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Contact Imitation Interface 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：分层策略分别规划接触表面轨迹和腕指协调。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**示范迁移要保留接触过程：分层策略分别规划接触表面轨迹和腕指协调。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：CIMI 通过共享接触几何、指尖视图和优先接触对齐的重定向，连接人类示范与灵巧手控制。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [contact_imitation_interface_arxiv_2610_41028.md](../../sources/papers/contact_imitation_interface_arxiv_2610_41028.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2610.41028](https://arxiv.org/abs/2610.41028)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.41028)
- [项目页](https://nus-lins-lab.github.io/CIMIWeb/)
- [https://github.com/ariszxxu/CIMI](https://github.com/ariszxxu/CIMI)
