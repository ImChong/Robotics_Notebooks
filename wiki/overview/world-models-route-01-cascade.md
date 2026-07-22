---
type: overview
tags: [world-models, category-hub, cascade, video-prediction, latent-action, shenlan-survey]
status: complete
updated: 2026-07-22
summary: "深蓝世界模型 15 项目 · 01 级联架构（6 篇）— 先预测未来视觉/4D 状态，再由逆动力学或动作头解码控制；误差在级联间传递是主要权衡。"
related:
  - ./world-models-15-open-source-technology-map.md
  - ./world-models-route-02-joint.md
  - ./world-models-route-03-virtual-sandbox.md
  - ../methods/mimic-video.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../entities/paper-shenlan-wm-01-tesseract.md
  - ../entities/paper-shenlan-wm-02-vpp.md
  - ../entities/paper-shenlan-wm-03-lapa.md
  - ../entities/paper-shenlan-wm-05-villa-x.md
  - ../entities/paper-shenlan-wm-06-video-gen-robot-policies.md
  - ../entities/paper-masked-visual-actions.md
sources:
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/papers/masked_visual_actions_arxiv_2607_19343.md
---

# 世界模型路线 01：级联架构

> **图谱分类节点**：对应 [深蓝具身智能 · 15 开源世界模型](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) 的 **01 级联架构** 分组；本页汇集该组 **6 篇** 工作的站内实体与 source 索引。总地图见 [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IDM | Inverse Dynamics Model | 由（未来）状态反推所需动作/力矩的模型 |
| VPP | Video Prediction Policy | 以视频预测为中介生成动作的策略架构 |
| VAM | Video-Action Model | 从视频学习并预测动作的模型 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |

## 核心问题

**能否先把未来「脑补」出来，再据此决定动作？** 级联架构将 **世界预测** 与 **动作解码** 分模块实现——通常先用视频/4D/潜特征模型生成未来表征，再用逆动力学（IDM）或轻量动作头输出控制。优势是 **可复用大规模视频预训练**；代价是 **误差在级联间传递**，未来预测偏差会直接污染动作。

**代表机制（策展）：** 4D 场景重建（TesserAct）→ 视频扩散表征 + IDM（VPP、mimic-video）→ 无标签视频潜在动作（LaPA、villa-X）→ 模块化视频+动作扩散（Video Generators are Robot Policies）

**路线外延（非 15 项目策展）：** [Masked Visual Actions](../entities/paper-masked-visual-actions.md)（arXiv:2607.19343）在 **逆向** 设定下先合成机器人视频、再用 **IDM** 抽低维动作——级联形态清晰，但 **同一** 视频骨干亦可切换为前向沙盒（见 [route-03](./world-models-route-03-virtual-sandbox.md)），说明「级联 vs 沙盒」在工程上可共享条件接口。

## 本组论文（6 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | TesserAct | [paper-shenlan-wm-01-tesseract.md](../entities/paper-shenlan-wm-01-tesseract.md) | [source](../../sources/papers/shenlan_wm_survey_01_tesseract.md) |
| 02 | VPP | [paper-shenlan-wm-02-vpp.md](../entities/paper-shenlan-wm-02-vpp.md) | [source](../../sources/papers/shenlan_wm_survey_02_vpp.md) |
| 03 | LaPA | [paper-shenlan-wm-03-lapa.md](../entities/paper-shenlan-wm-03-lapa.md) | [source](../../sources/papers/shenlan_wm_survey_03_lapa.md) |
| 04 | mimic-video | [mimic-video 方法页](../methods/mimic-video.md) | [source](../../sources/papers/shenlan_wm_survey_04_mimic-video.md) |
| 05 | villa-X | [paper-shenlan-wm-05-villa-x.md](../entities/paper-shenlan-wm-05-villa-x.md) | [source](../../sources/papers/shenlan_wm_survey_05_villa-x.md) |
| 06 | Video Generators are Robot Policies | [paper-shenlan-wm-06-video-gen-robot-policies.md](../entities/paper-shenlan-wm-06-video-gen-robot-policies.md) | [source](../../sources/papers/shenlan_wm_survey_06_video-gen-robot-policies.md) |

## 在 15 项目地图中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 01 级联架构 |
| 篇数 | 6/15 |
| 学术对照 | [robot-world-models-training-loop-taxonomy](./robot-world-models-training-loop-taxonomy.md) 中「策略内预测」与「可控视频生成」的部分实例落在此组 |
| 姊妹路线 | [02 联合架构](./world-models-route-02-joint.md)、[03 虚拟沙盒](./world-models-route-03-virtual-sandbox.md) |

## 关联页面

- [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)
- [mimic-video（VAM）](../methods/mimic-video.md)
- [Generative World Models](../methods/generative-world-models.md)
- [操作 VLA 架构选型 Query](../queries/manipulation-vla-architecture-selection.md)
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md) — 逆设定 + IDM 抽动作（兼前向沙盒）

## 参考来源

- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md) — <https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)

## 推荐继续阅读

- [VPP（arXiv:2412.14803）](https://arxiv.org/abs/2412.14803) — 视频预测表征 + 隐式 IDM 基线
- [mimic-video 方法页](../methods/mimic-video.md) — 冻结视频骨干 + 流匹配动作头
