---
type: entity
status: draft
updated: 2026-10-22
project_id: sharpa-d01
tags: [hardware, sharpa-robotics, dexterous-manipulation, tactile]
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ./sharpa-robotics.md
  - ./sharpa-w02-hand.md
  - ./craftnet.md
sources:
  - ../../sources/sites/sharpa-iros-2026-d01-w02-ae01.md
  - ../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md
summary: "Sharpa D01：IROS 2026 发布的整机，Sharpa Wave 手 + 7 DoF 臂 + 全身电子皮肤；末端速度 >10.5 m/s、重复精度 0.2 mm（公司自报）；文章称其把触觉扩到机器人身体。"
---

# Sharpa D01 整机

## 一句话定义

**D01** 是 Sharpa 的集成机器人：灵巧手、手臂与全身电子皮肤一体，天南文章称“把触觉扩到机器人身体”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| E-skin | Electronic skin | 电子皮肤 |
| IROS | Intelligent Robots and Systems | 2026-09 匹兹堡，发布场合 |

## 规格（公司新闻稿自报）

| 项 | 数值 |
|----|------|
| 构成 | Sharpa Wave 手 + 7 DoF 臂 + 全身电子皮肤 |
| 臂负载自重比 | 约 1:1 |
| 末端最高速度 | > 10.5 m/s |
| 通信频率 | 1000 Hz |
| 末端重复精度 | 0.2 mm |
| 电子皮肤 | 100 Hz 采样，0.2 N 力分辨率 |

发布于 IROS 2026（新闻稿 2026-09-29）；未给价格与交付时间，无独立评测。

## 为什么重要

- 对应 System 0 的高频接触控制：全身触觉让接触不再局限于指尖。
- 与 [CraftNet](./craftnet.md) 的软硬协同方向一致（推测）。

## 结论

**D01 是触觉从“手”扩展到“全身”的硬件载体；速度与精度规格亮眼，但无客户或基准结果。**

- 规格全部自报；
- 重量未披露；
- 是否运行 CraftNet 未见说明。

## 常见误区与局限

- 规格均为公司自报；第三方页面出现的 “Dexa D01 / Wave W02 / Avatar AE01” 等名称与官方稿不一致，本库以官方稿为准。
- 开源状态：硬件产品，无开源设计或 SDK 信息（2026-10-22 未见）。

## 关联页面

- [Sharpa Robotics](./sharpa-robotics.md) — 公司页
- [Sharpa 学术路线三条主线（阅读坐标）](../overview/sharpa-academic-route-three-lines.md)
- [触觉感知](../concepts/tactile-sensing.md)

## 参考来源

- [Sharpa IROS 2026 新闻稿归档](../../sources/sites/sharpa-iros-2026-d01-w02-ae01.md)
- [天南具身公园：Sharpa 学术路线](../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md)

## 推荐继续阅读

- [新闻稿原文](https://technode.global/prnasia/sharpa-unveils-three-flagship-products-d01-robot-w02-dexterous-hand-and-ae01-data-glove-at-iros-2026/)
