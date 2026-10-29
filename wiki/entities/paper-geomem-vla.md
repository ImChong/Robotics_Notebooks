---
type: entity
project_id: geomem-vla
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-10-29
arxiv: "2610.29866"
project: https://geomem-vla.github.io/
code: https://github.com/GeoMem-VLA/GeoMem-VLA
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/geomem_vla_arxiv_2610_29866.md
  - ../../sources/repos/geomem_vla.md
  - ../../sources/sites/geomem-vla.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "GeoMem-VLA（arXiv:2610.29866）：GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。"
---

# GeoMem-VLA（arXiv:2610.29866）

**GeoMem-VLA**（*GeoMem-VLA: Internalizing 3D Scene Understanding into Temporal Memory for Vision-Language-Action Models*，[arXiv:2610.29866](https://arxiv.org/abs/2610.29866)，[项目页](https://geomem-vla.github.io/)，[代码](https://github.com/GeoMem-VLA/GeoMem-VLA)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「VLA、智能体与策略后训练」组。

## 一句话定义

**GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GeoMem-VLA | GeoMem-VLA: Internalizing 3D Scene Understanding into Temporal Memory for Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| 3D | 3D | 三维 |

## 为什么重要

- 把三维理解内化成可持续更新的记忆。
- 开源结论：**已开源**（步骤 2.5，2026-10-29）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.29866](https://arxiv.org/abs/2610.29866) |
| **开源** | **已开源** |
| **要点** | GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`GeoMem-VLA/GeoMem-VLA`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as GeoMem-VLA/GeoMem-VLA
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 GeoMem-VLA 模型
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

**把三维理解内化成可持续更新的记忆：GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [geomem_vla_arxiv_2610_29866.md](../../sources/papers/geomem_vla_arxiv_2610_29866.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.29866](https://arxiv.org/abs/2610.29866)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.29866)
- [项目页](https://geomem-vla.github.io/)
- [https://github.com/GeoMem-VLA/GeoMem-VLA](https://github.com/GeoMem-VLA/GeoMem-VLA)
