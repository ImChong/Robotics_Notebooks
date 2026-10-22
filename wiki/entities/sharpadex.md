---
type: entity
status: draft
updated: 2026-10-22
project_id: sharpadex
tags: [dataset, tactile, dexterous-manipulation, data-collection, sharpa-robotics]
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ./sharpa-robotics.md
  - ../concepts/visuo-tactile-fusion.md
  - ./paper-ftp-1.md
  - ./paper-uvta-human-tactile-dexterous-manipulation.md
sources:
  - ../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md
summary: "SharpaDex：Sharpa 发布的首个触觉数据集，天南文章称 300 小时、目前最大的灵巧手真机触觉数据集，至少 64.9% 时间处于密集接触；除该文章外本库未检索到官方页面，细节待核实。"
---

# SharpaDex：Sharpa 灵巧手触觉数据集

## 一句话定义

**SharpaDex** 是 Sharpa 于 2026-10 前后发布的首个触觉数据集；全部信息来自单篇微信文章，**无官方页面或论文可核对**（2026-10-22 检索未见）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SharpaDex | —（文内仅称此名） | 数据集名，文内仅称 SharpaDex |
| 触觉 | Tactile | 指尖/手部接触信号 |
| FTP-1 | —（见 FTP-1 节点） | 跨传感器触觉预训练，同一路线 |

## 文章给出的事实（仅作者转述）

| 项 | 内容 | 性质 |
|----|------|------|
| 规模 | 300 小时，“目前最大的灵巧手真机触觉数据集” | 文章转述，未核对 |
| 接触占比 | 至少 64.9% 时间在密集接触中 | 文章转述 |
| 小指 | 触觉接触时间 < 5% | 文章转述 |
| 估值 | 按 2000–4000 元/小时 ≈ 120 万元 | **作者估算**，非官方 |
| 开放程度 | 标题称“顺手开源”，但无仓库/下载链接 | 待核实 |
| 模态 | 灵巧手真机触觉信号；是否同步视觉、关节状态未披露 | 文章未给出 |
| 许可证 | 未见许可证或使用条款 | 待核实 |
| 重定向就绪度 | 采集手型与传感器布局未披露，能否迁移到其他灵巧手形态未知 | 文章未给出 |

## 为什么重要

- 触觉灵巧数据此前稀缺：库内 T-Rex 数据集约 50 小时量级（据其 Hugging Face 卡片，见 [T-Rex](./paper-trex-tactile-reactive-dexterous-manipulation.md)）；300 小时若属实将是数量级提升。
- 对应天南归纳的「路线三」：为 [FTP-1](./paper-ftp-1.md) 这类跨本体触觉表征提供数据。

## 流程总览

```mermaid
flowchart LR
  A["灵巧手真机触觉采集"] --> B["SharpaDex 300 h"]
  B --> C["触觉表征预训练"]
  C --> D["下游灵巧操作策略"]
```

上图是依据文章语境的推测性示意，非官方管线。

## 结论

**SharpaDex 目前是“有说法、无证据”的节点：值得跟踪，不宜引用其规模作硬数据。**

- 只有 300 h、64.9%、<5% 三个数字，均来自同一篇文章。
- 开源状态未核实；找到官方页面后应补 `sources/sites/` 与许可、格式、传感器规格。
- 价值估算是作者主观推算，勿当市场价。

## 常见误区与局限

- “最大”是转述表述，对比范围未给出（本库未查到更大的同类集合并不等于证实）。
- 采集手型、传感器类型、任务分布均未披露。

## 关联页面

- [Sharpa Robotics](./sharpa-robotics.md)
- [FTP-1](./paper-ftp-1.md) — 跨传感器触觉表征
- [UVTA](./paper-uvta-human-tactile-dexterous-manipulation.md) — 人类触觉数据采集
- [视触觉融合](../concepts/visuo-tactile-fusion.md)
- [Sharpa 学术路线三条主线（阅读坐标）](../overview/sharpa-academic-route-three-lines.md)

## 参考来源

- [天南具身公园：Sharpa 学术路线](../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md)

## 推荐继续阅读

- [Sharpa 官网](https://www.sharpa.com/)（关注数据集发布页）
