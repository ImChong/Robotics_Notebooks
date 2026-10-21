---
type: entity
tags:
  - dataset
  - real-robot
  - bimanual
  - aloha
status: draft
updated: 2026-10-20
summary: "基于开源 ALOHA 主从遥操作采集的双臂数据集：1 万余条示范、100 余项任务，CC BY 4.0。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# AIST Bimanual Manipulation Dataset

## 一句话定义

**AIST Bimanual Manipulation Dataset**：日本 AIST 用开源 ALOHA 双臂系统采集的多任务双臂示范数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ALOHA | A Low-cost Open-source Hardware System | 低成本开源双臂遥操作系统 |
| CC BY | Creative Commons Attribution | 署名型开放许可 |
| MS | Master-Slave（主从） | 主从遥操作 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 产业技术综合研究所（AIST） |
| 时间 | 2025 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1 万余条示范，100 余项任务 |
| 模态 | 多视角同步视频、关节轨迹、文本标注 |
| 许可 | CC BY 4.0 |

## 为什么重要

- 许可开放（CC BY 4.0），并按双臂协作方式、技能类别和采集日期组织，便于分任务选取。

## 核心原理

主从遥操作采集，任务按双臂分工、接触密集程度与技能类别分类。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | 官网按任务预览，官方下载器批量获取（文章给出链接） |

## 局限与风险

- 数据基于 ALOHA 构型，迁移到其他双臂平台需要标定与动作空间对齐（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://aistairc.github.io/aist_bimanip_site/)
