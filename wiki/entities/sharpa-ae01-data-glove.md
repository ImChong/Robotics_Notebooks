---
type: entity
status: draft
updated: 2026-10-22
project_id: sharpa-ae01
tags: [hardware, sharpa-robotics, dexterous-manipulation, tactile]
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ./sharpa-robotics.md
  - ./sharpa-w02-hand.md
  - ./paper-uvta-human-tactile-dexterous-manipulation.md
sources:
  - ../../sources/sites/sharpa-iros-2026-d01-w02-ae01.md
  - ../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md
summary: "Sharpa AE01：触觉外骨骼数据手套，22 DoF/22 编码器、每指尖 256 级振动反馈，用于遥操作与第一视角数据采集（公司自报）；天南文章称其把操作者动作与触觉反馈带进采集。"
---

# Sharpa AE01 触觉数据手套

## 一句话定义

**AE01** 是带振动触觉反馈的外骨骼数据手套，用于灵巧手遥操作与人类数据采集，直指“操作员感受不到机器手”的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| UVTA | Unified Visual-Tactile-Action | 相近外骨骼采集论文 |
| IROS | Intelligent Robots and Systems | 2026-09 匹兹堡，发布场合 |

## 规格（公司新闻稿自报）

| 项 | 数值 |
|----|------|
| 自由度 | 22 DoF，22 编码器 |
| 触觉反馈 | 每指尖 256 级振动 |
| 指尖位置重复精度 | < 1 mm |
| 标定 | 无需额外用户标定 |

发布于 IROS 2026（新闻稿 2026-09-29）；未给价格与交付时间，无独立评测。

## 为什么重要

- 对应文章的断点 1（遥操作无触觉回传）与路线一的“以人为中心采集”。
- 与 [UVTA](./paper-uvta-human-tactile-dexterous-manipulation.md) 的被动 22 DoF 外骨骼思路相近，但二者是否同一设备未核实。

## 结论

**AE01 是把触觉反馈引入采集链路的硬件，但振动反馈不等于力反馈。**

- 反馈为振动触觉，非真实力反馈；
- 与 UVTA 采集设备的关系未核实。

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
