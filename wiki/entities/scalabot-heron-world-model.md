---
type: entity
tags: [company, noetix, world-model, embodied-ai, humanoid]
status: draft
updated: 2026-10-22
related:
  - ../concepts/world-action-models.md
  - ../entities/cn-os-noetix-sdk-bumi.md
sources:
  - ../../sources/sites/scalabot_noetixrobotics_com.md
summary: "Scalabot（缩放机器人）是松延动力 2026-09-08 发布的通用具身智能技术子品牌；HERON 为技术架构，HERON-World Model 是其首个世界模型（WorldArena 1.0 Track 1 离线综合 75.80，公司口径）。"
institutions:
  - noetix
---

# Scalabot / HERON-World Model（松延动力）

## 一句话定义

**Scalabot（缩放机器人）** 是 [松延动力（Noetix Robotics）](https://github.com/Noetix-Robotics) 于 2026-09-08 发布的通用具身智能 **技术子品牌**（非独立研究实验室，官网 `parentOrganization` 为北京松延动力科技集团股份有限公司）；**HERON** 是其技术架构，**HERON-World Model** 是该架构下首个世界模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HERON | （官方未展开） | Scalabot 的技术架构体系，后续计划发布更多模型 |
| CRA | （官方未展开） | 官网关键词 HERON-CRA，本次未找到可核实细节 |
| MoT | Mixture-of-Transformers | 分离 VLM 与生成参数，单向注意力以保留 VLM 世界知识 |
| MoE | Mixture-of-Experts | DiT 去噪拆为高噪声 / 低噪声两类专家 |
| DiT | Diffusion Transformer | 扩散 Transformer 生成骨干 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- 松延动力此前以本体（N2 / E1 / Bumi）为主，Scalabot 标志其向「本体 + 具身大脑」扩展（Gasgoo 报道 + 创始人访谈转述）。
- 世界模型输入「当前图像 + 动作」，预测后续画面并含多智能体交互；可与本库 [World Action Models](../concepts/world-action-models.md) 对照阅读。

## 核心原理（量子位 2026-09-09 报道归纳）

| 要点 | 内容 |
|------|------|
| 数据 | 第一视角人类视频 + 真实场景重建仿真 + 真机数据；真机作「真实执行锚点」，仿真覆盖罕见 / 危险场景 |
| 动作统一 | 按视频时间戳对齐，转末端位姿，分为自身运动、操作器运动、抓取状态，经轻量编码器映射为定长 Action Token |
| 遮挡 | 用 mask 与置信度避免遮挡样本被误当作「无动作」 |
| 架构 | MoT 隔离知识；DiT 去噪拆高 / 低噪声 MoE（粗结构与运动 vs 局部几何与精细交互） |
| 评测 | WorldArena 1.0 Track 1 离线，EWMScore_P **75.80**（物理交互 95.60、空间视角 99.20、运动动力学 92.61、轨迹精度 56.66） |

## 局限与风险

- 评测分数为 **公司口径**，未见第三方复核。
- 官网为前端渲染，静态页仅含元信息；HERON-CRA 细节、权重 / 代码开放情况 **未核实**。
- 报道称「跨平台、统一动作表示」与「数据闭环」属规划 / 宣称，不等于已交付。

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [noetix_sdk_bumi](./cn-os-noetix-sdk-bumi.md)

## 参考来源

- [Scalabot 官网归档](../../sources/sites/scalabot_noetixrobotics_com.md)（<https://scalabot.noetixrobotics.com/>）
- [Gasgoo：Noetix Robotics Launches Embodied Intelligence Sub-brand Scalabot](https://autonews.gasgoo.com/articles/news/noetix-robotics-launches-embodied-intelligence-sub-brand-scalabot-2097870821795381249)
- [量子位：HERON-World Model 发布](https://www.qbitai.com/2026/09/486291.html)
