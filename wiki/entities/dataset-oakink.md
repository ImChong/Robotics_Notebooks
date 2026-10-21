---
type: entity
tags:
  - dataset
  - hand-object
  - affordance
  - knowledge-base
status: draft
updated: 2026-10-20
summary: "知识驱动的手-物交互库：1,800 种家居物体的功能可供性，加上 5 万个意图导向交互。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# OakInk / OakInk2

## 一句话定义

**OakInk**：由物体可供性知识库 Oak 与交互知识库 Ink 组成的手-物交互数据集；OakInk2 扩展到复杂双手日常任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Oak | OakBase（物体可供性库） | 物体知识库 |
| Ink | 交互知识库 | 交互数据部分 |
| Affordance | 可供性 | 物体可被如何使用的属性 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 上海交通大学、上海期智研究院 |
| 时间 | 2022-03（CVPR 2022） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1,800 种物体；100 个物体上的丰富交互；约 5 万个交互 |
| 模态 | 手-物交互位姿（人体、双手、物体）+ 可供性 / 意图标注；OakInk2 另含多视角图像流 |
| OakInk2 | 多视角图像流与人体、双手、物体精确位姿 |
| 层级 | 可供性、原子任务、复杂任务 |

## 为什么重要

- 把可供性（affordance）与交互意图绑定，适合研究“为什么这样抓”。

## 核心原理

先标注物体可供性，再用 Tink 方法把人类交互迁移到虚拟物体上。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | oakink.net 下载页（文章给出链接） |
| 代码 | oakink/OakInk 仓库 |

## 局限与风险

- 交互迁移到虚拟物体的真实性需核对（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://oakink.net/)
