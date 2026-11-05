---
type: entity
project_id: pyrua-datafarm
tags:
  - paper
  - dexterous
  - imitation-learning
status: complete
updated: 2026-11-05
arxiv: "2611.05320"
project: https://dagroup-pku.github.io/PyRUA-Lean/DataFarm/
code: https://github.com/DAGroup-PKU/PyRUA-Lean/tree/DataFarm
related:
  - ../methods/imitation-learning.md
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../tasks/teleoperation.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/pyrua_datafarm_arxiv_2611_05320.md
  - ../../sources/repos/pyrua_datafarm.md
  - ../../sources/sites/pyrua-datafarm.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "PyRUA-DataFarm（arXiv:2611.05320）：每项任务录一条校准示范，再由智能体搭仿真、渲染多视角视频。"
---

# PyRUA-DataFarm（arXiv:2611.05320）

**PyRUA-DataFarm**（*PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data*，[arXiv:2611.05320](https://arxiv.org/abs/2611.05320)，[项目页](https://dagroup-pku.github.io/PyRUA-Lean/DataFarm/)，[代码](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/DataFarm)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「灵巧操作与示范学习」组。

## 一句话定义

**每项任务录一条校准示范，再由智能体搭仿真、渲染多视角视频。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PyRUA-DataFarm | PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data | 本文方法 / 基准的简称 |

## 为什么重要

- 用一次真机校准生成多视角示范。五项真机任务换桌布和灯光后，合成数据组成功 87/100 次，等量遥操作示范组为 46/100。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05320](https://arxiv.org/abs/2611.05320) |
| **开源** | **已开源** |
| **要点** | 每项任务录一条校准示范，再由智能体搭仿真、渲染多视角视频。 |
| **文内指标** | 公众号转述：五项真机任务换桌布和灯光后，合成数据组成功 87/100 次，等量遥操作示范组为 46/100。 |

## 源码运行时序图

以下为按仓库入口（`DAGroup-PKU/PyRUA-Lean/tree/DataFarm`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as DAGroup-PKU/PyRUA-Lean/tree/DataFarm
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 PyRUA-DataFarm 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：五项真机任务换桌布和灯光后，合成数据组成功 87/100 次，等量遥操作示范组为 46/100。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**用一次真机校准生成多视角示范：五项真机任务换桌布和灯光后，合成数据组成功 87/100 次，等量遥操作示范组为 46/100。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：每项任务录一条校准示范，再由智能体搭仿真、渲染多视角视频。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [imitation-learning](../methods/imitation-learning.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [teleoperation](../tasks/teleoperation.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [pyrua_datafarm_arxiv_2611_05320.md](../../sources/papers/pyrua_datafarm_arxiv_2611_05320.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05320](https://arxiv.org/abs/2611.05320)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05320)
- [项目页](https://dagroup-pku.github.io/PyRUA-Lean/DataFarm/)
- [https://github.com/DAGroup-PKU/PyRUA-Lean/tree/DataFarm](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/DataFarm)
