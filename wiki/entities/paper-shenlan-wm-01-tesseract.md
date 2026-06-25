---

type: entity
tags: [paper, world-models, shenlan-survey, open-source, hkust]
status: complete
updated: 2026-06-25
arxiv: "2504.20995"
venue: ICCV 2025
summary: "首个开源通用 4D 机器人 WM：RGB+深度+法线视频与动作预测，跨域泛化优于 2D 视频 WM。"
related:
  - ../overview/world-models-15-open-source-technology-map.md
  - ../overview/world-models-route-01-cascade.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/shenlan_wm_survey_01_tesseract.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
---

# TesserAct

**TesserAct: Learning 4D Embodied World Models** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 01/15** 篇，归类为 **01 级联架构**。

## 一句话定义

首个开源通用 4D 机器人 WM：RGB+深度+法线视频与动作预测，跨域泛化优于 2D 视频 WM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |

## 为什么重要

- 首个开源通用 4D 机器人 WM：RGB+深度+法线视频与动作预测，跨域泛化优于 2D 视频 WM。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 01**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/15 |
| 路线 | 01 级联架构 |
| 出处 | ICCV 2025 |
| 文内引用 | 75（2026-06-02，策展） |
| arXiv | [2504.20995](https://arxiv.org/abs/2504.20995) |

## 核心机制（归纳）

### 1）策展导读要点

先预测未来视觉/潜特征（视频、RGB-D、法线等），再由独立或轻量动作头解码控制指令；误差在级联各段传递，工程上易拆模块复用。

### 2）策展导读要点

首个开源通用 4D 机器人 WM：RGB+深度+法线视频与动作预测，跨域泛化优于 2D 视频 WM。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-01-cascade.md](../overview/world-models-route-01-cascade.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_01_tesseract.md](../../sources/papers/shenlan_wm_survey_01_tesseract.md)

## 参考来源

- [shenlan_wm_survey_01_tesseract.md](../../sources/papers/shenlan_wm_survey_01_tesseract.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 推荐继续阅读

- [arXiv:2504.20995](https://arxiv.org/abs/2504.20995) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)
