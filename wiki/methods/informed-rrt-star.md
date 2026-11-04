---
type: method
tags: [path-planning, sampling-based, rrt, informed-sampling]
status: complete
updated: 2026-11-04
related:
  - ./rrt.md
  - ./rrt-star.md
  - ./rrt-connect.md
  - ../concepts/configuration-space.md
  - ../entities/ompl.md
  - ../entities/moveit2.md
  - ../methods/a-star.md
  - ../comparisons/robot-path-planning-five-paradigms-taxonomy.md
sources:
  - ../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md
summary: "Informed RRT* 在找到初始解后，把采样集中到可能继续改善当前路径的区域（路径长度代价下为长椭球体），保持与 RRT* 相同的完备性与最优性保证，同时提高收敛速度与终解质量。"
---

# Informed RRT*

> 依据 [深蓝AI《彻底搞懂RRT与RRT*》](https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg) 与一手论文 Gammell、Srinivasa、Barfoot《Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic》（[arXiv:1404.2334](https://arxiv.org/abs/1404.2334)，摘要页已核对）整理。

## 一句话定义

**Informed RRT* 在找到初始解以后，只在可能改善当前解的子集里采样，减少无效探索。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Informed RRT* | Informed Optimal RRT | 带直接采样可改进子集的 RRT* |
| Prolate hyperspheroid | Prolate hyperspheroid | 长椭球体：路径长度代价下可改进状态子集的形状 |

## 为什么重要

- 一手摘要：RRT* 虽能渐近找到起点到规划域内每个状态的最优路径，但这既低效、又与单查询性质不符；对最小化路径长度的问题，可改进当前解的状态子集可描述为长椭球体；若不直接采样该子集，大世界或高维下改进解的概率会变得任意小。
- 论文称该方法保留与 RRT* 相同的完备性与最优性概率保证，并在收敛速度、终解代价、通过困难通道的能力上优于 RRT*，且对状态维数与范围依赖更小（论文实验结论，本页未复现）。

## 主要技术路线

- 在 [RRT*](./rrt-star.md) 上做的简单修改：找到初始解后改为在可改进子集内直接采样（论文称为 exact method，直接采样该子集）。
- 文章的归纳：把更多采样集中在可能继续改善当前路径的区域，以减少无效探索。

## 关联页面

- [RRT](./rrt.md)、[RRT*](./rrt-star.md)、[RRT-Connect](./rrt-connect.md)
- [配置空间](../concepts/configuration-space.md)
- [OMPL](../entities/ompl.md)、[MoveIt 2](../entities/moveit2.md)
- [A*](./a-star.md)
- [路径规划五范式分类](../comparisons/robot-path-planning-five-paradigms-taxonomy.md)

## 参考来源

- [文章归档](../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md)
- Gammell 等，[arXiv:1404.2334](https://arxiv.org/abs/1404.2334)（提交 2014-04-08）
