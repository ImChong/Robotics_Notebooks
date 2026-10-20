# ActiveAlign — 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** Look Before You Act: Bridging Active Perception and Vision-Language-Action Models via Observation-Aligned Data Adaptation
- **作者：** Jiawei Geng、Zhenyang Liu、Yuzheng Wu、Yongchong Gu、Yanwei Fu
- **机构：** 复旦大学、上海创智学院
- **预印本：** <https://arxiv.org/abs/2610.23378>
- **版本：** arXiv v1，2026-10-16
- **入库日期：** 2026-10-20
- **一句话说明：** 先移动相机再操作会改变 VLA 的首帧观测与机器人状态分布，ActiveAlign 用成功续接轨迹适配策略训练分布。

## 核心论文摘录

### 问题与方法

主动调整腕部相机可以暴露遮挡目标，但相机轨迹同时改变机器人状态和策略接收到的首帧分布；直接把相机动作接在既有 VLA 前面，可能让其从训练分布之外的状态开始。ActiveAlign 先由视角模块生成相机轨迹，再离线将轨迹终点闭环对齐到既有示范，重放剩余操作轨迹并保留成功续接的数据，用于微调 VLA。部署时沿用同一视角模块，VLA 架构无需修改。

### 评测与结果

- 在 40 个 LIBERO 任务上，ActiveAlign + π₀ 达到 70.7% 成功率，较标准训练基线提高 40.0 个百分点；ActiveAlign + π₀.₅ 达到 72.7%，提高 26.4 个百分点。
- 物理遮挡测试成功次数由 35/90 提升至 83/90。
- 匹配状态与数据量对照用于区分收益是否仅来自前缀缩短或一般视角增强；论文还报告 OpenVLA 与 GR00T N1 上有相同趋势。
- arXiv 页面未列项目页或代码仓库链接；本次未核实到官方公开实现。

## 对 wiki 的映射

- [ActiveAlign 实体页](../../wiki/entities/paper-activealign.md)

## 一手来源

- [论文（arXiv:2610.23378）](https://arxiv.org/abs/2610.23378)
- [arXiv HTML 全文](https://arxiv.org/html/2610.23378v1)
