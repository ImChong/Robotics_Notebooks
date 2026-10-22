---
type: entity
status: draft
updated: 2026-10-22
project_id: sharpa-w02
tags: [hardware, sharpa-robotics, dexterous-manipulation, tactile]
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ./sharpa-robotics.md
  - ./sharpa-d01-robot.md
  - ./paper-ftp-1.md
sources:
  - ../../sources/sites/sharpa-iros-2026-d01-w02-ae01.md
  - ../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md
summary: "Sharpa W02：W01（Sharpa Wave）后继，21 主动 DoF、<750 g，指尖动态触觉阵列 + 指/掌电子皮肤、IP54（公司自报）；文章称其覆盖指尖与掌面。"
---

# Sharpa W02 灵巧手

## 一句话定义

**W02** 是 Sharpa 第二代灵巧手，在 Wave 一代基础上把触觉从指尖扩展到全手。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| IP54 | Ingress Protection 54 | 防尘与防溅水等级 |
| IROS | Intelligent Robots and Systems | 2026-09 匹兹堡，发布场合 |

## 规格（公司新闻稿自报）

| 项 | 数值 |
|----|------|
| 主动自由度 | 21 |
| 重量 | < 750 g |
| 指尖 | Dynamic Tactile Array，5 mN–30 N，1 mm 空间分辨率 |
| 指/掌 | 电子皮肤，0.1–20 N，5 mm 空间分辨率 |
| 其他 | 零间隙抓取（筷子、细绳）；IP54 |

发布于 IROS 2026（新闻稿 2026-09-29）；未给价格与交付时间，无独立评测。

## 为什么重要

- 一代 Wave 为 22 DoF（见公司页），W02 为 21 主动 DoF，二者口径不同，勿直接比较大小。
- 全手触觉为 [FTP-1](./paper-ftp-1.md) 等触觉表征提供更完整输入（推测）。

## 结论

**W02 把触觉覆盖面扩到全手并加防护等级，是面向落地的迭代。**

- 21 vs 22 DoF 的差异原因新闻稿未解释；
- 价格与量产时间未披露。

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
