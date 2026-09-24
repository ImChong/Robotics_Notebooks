---
type: entity
tags: [patent, humanoid, hardware, xpeng, iron, whole-body-embodied]
status: complete
updated: 2026-09-24
related:
  - ../overview/xpeng-iron-54-patents-technology-map.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../entities/paper-xpace.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/patents/xpeng_iron_patents_cn.md
  - ../../sources/blogs/wechat_xpeng_iron_54_patents_2026-09-24.md
summary: "小鹏 IRON CN118927245A：VR头显与惯性动捕融合的多部位遥操作（2024-11-12 公开/公告）；VR 滤波动捕姿态并传头显位姿，生成腰/臂/手/颈等多部位指令，肘部等约束冗余臂奇异。…"
---

# 小鹏 IRON · VR头显与惯性动捕融合的多部位遥操作（CN118927245A）

专利 **CN118927245A**（*VR头显与惯性动捕融合的多部位遥操作*，小鹏机器人 / IRON 相关布局，2024-11-12）公开 **全身控制·具身智能** 方向的结构或控制方案 — 编译自 [54 项专利盘点](../../sources/blogs/wechat_xpeng_iron_54_patents_2026-09-24.md)（AI工业 / 微信公众号，2026-09-24）。

## 一句话定义

**VR 滤波动捕姿态并传头显位姿，生成腰/臂/手/颈等多部位指令，肘部等约束冗余臂奇异。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CN | China National Patent | 中国国家知识产权局专利公开/授权号 |
| IRON | XPeng Humanoid Robot | 小鹏人形机器人产品系（含 IRON-R01 等） |
| DoF | Degrees of Freedom | 机构自由度或控制维度 |
| WBC | Whole-Body Control | 全身协调控制（步态/遥操作类专利常相关） |

## 为什么重要

- **IRON 硬件栈信号：** 与 [XPACE](../entities/paper-xpace.md) 等公开论文侧重 **学习栈** 不同，本件反映 **小鹏在 全身控制·具身智能 上的机构/控制专利布局**。
- **选型参照：** 读 IRON 肩/膝/手/步态实现时，可将公开专利与 [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md) 对照，理解 **直线执行器、十字轴、连杆传力** 等工程取舍。
- **非等同量产：** 专利披露 **≠** 当前 IRON 量产 BOM；仅作 **技术路线与优先级** 线索。

## 核心结构 / 方法要点

VR 滤波动捕姿态并传头显位姿，生成腰/臂/手/颈等多部位指令，肘部等约束冗余臂奇异。

| 字段 | 内容 |
|------|------|
| 专利号 | CN118927245A |
| 类别 | 全身控制·具身智能 |
| 公开/公告 | 2024-11-12 |
| 权利人 | 小鹏机器人相关主体（以专利局登记为准） |
| 开源 | **不适用** — 专利文本公开，无代码仓库 |

## 常见误区或局限

- **误区：「专利标题 = 已量产功能」。** 申请/授权文本为 **布局与实施例披露**，量产可能迭代或仅部分采用。
- **局限：** 公众号整理为 **摘要级** 解读，力矩/带宽/控制频率等 **未在 ingest 中量化**；细节以 [Google Patents](https://patents.google.com/patent/CN118927245A/zh) 原文为准。

## 关联页面

- [小鹏 IRON 54 项专利技术地图](../overview/xpeng-iron-54-patents-technology-map.md)
- [XPACE（小鹏 WAM）](../entities/paper-xpace.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [小鹏 IRON 54 项专利索引（CN）](../../sources/patents/xpeng_iron_patents_cn.md#cn118927245a)
- [微信公众号盘点归档](../../sources/blogs/wechat_xpeng_iron_54_patents_2026-09-24.md)

## 推荐继续阅读

- Google Patents：[CN118927245A](https://patents.google.com/patent/CN118927245A/zh)
- [小鹏 IRON 54 项专利原文链接](https://mp.weixin.qq.com/s/R7Qi2iv1eNfm3yh2s_PUCg)
