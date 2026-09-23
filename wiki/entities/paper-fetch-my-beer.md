---
type: entity
tags:
  - paper
  - manipulation
  - liquid-transport
  - hierarchical-policy
  - diffusion
  - sim2real
status: complete
updated: 2026-09-23
arxiv: "2609.18119"
related:
  - ../tasks/manipulation.md
  - ../concepts/sim2real.md
  - ../methods/imitation-learning.md
  - ./paper-dreaming-sound-of-contact.md
  - ../overview/constraint-control-11-papers-technology-map.md
sources:
  - ../../sources/papers/fetch-my-beer_arxiv_2609_18119.md
  - ../../sources/sites/fetch-my-beer.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md
summary: "Fetch My Beer（arXiv:2609.18119）：流体仿真筛选稳定轨迹 + VLM 过滤不稳定姿态；高层语言视觉给 SE(3) 目标，潜扩散控制器生成平滑动作块；仅合成示范零样本 sim-to-real 液体运输。"
---

# Fetch My Beer（arXiv:2609.18119）

**Fetch My Beer**（*Fetch My Beer: Synthetic-to-real Hierarchical Policy for Smooth Pick-and-place*，[arXiv:2609.18119](https://arxiv.org/abs/2609.18119)，[项目页](https://fetch-my-beer.github.io/)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)（2026-09-20）。

## 一句话定义

**流体仿真筛选稳定轨迹 + VLM 过滤不稳定姿态；高层语言视觉给 SE(3) 目标，潜扩散控制器生成平滑动作块；仅合成示范零样本 sim-to-real 液体运输。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SE(3) | Special Euclidean Group | 刚体位姿空间 |
| VLM | Vision-Language Model | 视觉–语言模型筛选姿态 |
| RA-L | IEEE Robotics and Automation Letters | 发表期刊 |
| Sim2Real | Simulation to Real | 合成到真机迁移 |

## 为什么重要

- 装满液体的容器即使抓取成功，急停/转向仍可能洒出；需轨迹级动态稳定而非仅到达目标。
- 开源结论：**待发布**（步骤 2.5，2026-09-20）。
- 与 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.18119](https://arxiv.org/abs/2609.18119) |
| **开源** | **待发布** |
| **要点** | 合成抓取+流体仿真验证 → 分层：高层 SE(3) 目标 + 潜空间扩散密集动作块；强调 motion smoothness。 |
| **文内指标** | 液体 pick-and-place；相对此前 SOTA 操作策略在运输平滑度与动态稳定性上更优（项目页/RA-L 2026）。 |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。


## 实验与评测

- 液体 pick-and-place；相对此前 SOTA 操作策略在运输平滑度与动态稳定性上更优（项目页/RA-L 2026）。
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**Fetch My Beer 把「不洒」作为 pick-and-place 的一等目标；代码/数据截至入库日标注 Coming soon。**

1. 开源边界：**待发布** — 以项目页实际链接为准（入库日 2026-09-20）。
2. 核心机制：合成抓取+流体仿真验证 → 分层：高层 SE(3) 目标 + 潜空间扩散密集动作块；强调 motion smoothness。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [sim2real](../concepts/sim2real.md)
- [imitation-learning](../methods/imitation-learning.md)
- [paper-dreaming-sound-of-contact](./paper-dreaming-sound-of-contact.md)

## 参考来源

- [fetch-my-beer_arxiv_2609_18119.md](../../sources/papers/fetch-my-beer_arxiv_2609_18119.md)
- [wechat_embodied_station_11_papers_constraint_control_2026-09-20.md](../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)
- [arXiv:2609.18119](https://arxiv.org/abs/2609.18119)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.18119)
- [项目页](https://fetch-my-beer.github.io/)

