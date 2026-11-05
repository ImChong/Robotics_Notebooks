---
type: entity
project_id: shape-servoing-tutorial-part-ii
tags:
  - paper
  - humanoid
  - tutorial
status: complete
updated: 2026-11-05
arxiv: "2611.03825"
project: https://nachocz.github.io/tutorial_on_shape_servoing_of_deformable_objects/
code: https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects
related:
  - ../tasks/locomotion.md
  - ../concepts/motion-retargeting.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/shape_servoing_tutorial_part_ii_arxiv_2611_03825.md
  - ../../sources/repos/shape_servoing_tutorial_part_ii.md
  - ../../sources/sites/shape-servoing-tutorial-part-ii.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "Shape Servoing Tutorial Part II（arXiv:2611.03825）：从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。"
---

# Shape Servoing Tutorial Part II（arXiv:2611.03825）

**Shape Servoing Tutorial Part II**（*A Tutorial on Shape Servoing of Deformable Objects—Part II: Advanced Concepts*，[arXiv:2611.03825](https://arxiv.org/abs/2611.03825)，[项目页](https://nachocz.github.io/tutorial_on_shape_servoing_of_deformable_objects/)，[代码](https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「人形运动与教程」组。

## 一句话定义

**从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Shape Servoing Tutorial Part II | A Tutorial on Shape Servoing of Deformable Objects—Part II: Advanced Concepts | 本文方法 / 基准的简称 |

## 为什么重要

- 柔性物体变形控制为什么会失效。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03825](https://arxiv.org/abs/2611.03825) |
| **开源** | **已开源** |
| **要点** | 从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`nachocz/tutorial_on_shape_servoing_of_deformable_objects`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as nachocz/tutorial_on_shape_servoing_of_deformable_objects
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Shape Servoing Tutorial Part II 模型
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

**柔性物体变形控制为什么会失效：从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [shape_servoing_tutorial_part_ii_arxiv_2611_03825.md](../../sources/papers/shape_servoing_tutorial_part_ii_arxiv_2611_03825.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03825](https://arxiv.org/abs/2611.03825)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03825)
- [项目页](https://nachocz.github.io/tutorial_on_shape_servoing_of_deformable_objects/)
- [https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects](https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects)
