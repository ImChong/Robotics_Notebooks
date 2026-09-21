---
type: entity
tags:
  - vlm
  - embodied-reasoning
  - ai2
  - open-source
status: complete
updated: 2026-09-21
related:
  - ./paper-molmoact2.md
  - ./molmo2-vlm.md
  - ./paper-lightnav-0.md
  - ./lightnav-er.md
  - ../overview/spatial-reasoning-benchmarks-technology-map.md
sources:
  - ../../sources/papers/molmoact2_arxiv_2605_02881.md
  - ../../sources/sites/allenai-molmoact2.md
  - ../../sources/papers/molmo_er_molmoact2_2026.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "Molmo2-ER：Ai2 具身推理 VLM 骨干；MolmoAct2 的动作 reasoning 底座；ER 数据集已开源。"
---

# Molmo2-ER（MolmoER）

**Molmo2-ER**（**MolmoER**）是 **Ai2** 为 [MolmoAct2](./paper-molmoact2.md) 提供的 **具身推理（Embodied Reasoning）** 视觉-语言骨干：在通用 VLM 能力之上 mid-training 空间/指代/ER benchmark，再接入 flow-matching 动作专家。

## 一句话定义

**Molmo2-ER 是 MolmoAct2 的「先理解再动手」大脑——LightNav-ER 路线的开源对照。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ER | Embodied Reasoning | 具身推理 mid-training |
| VLM | Vision-Language Model | 视觉-语言骨干 |
| VLA | Vision-Language-Action | MolmoAct2 完整栈含动作头 |

## 为什么重要

- **LightNav 对照：** [LightNav-0](./paper-lightnav-0.md) 博客将 MolmoER / Gemini Robotics-ER 列为「先 ER 再 SFT」行业平行实现。
- **数据开源：** HF **Molmo2-ER datasets** 与 MolmoAct2 同步发布。
- **与 Molmo2 VLM 分工：** [Molmo2](./molmo2-vlm.md) 偏通用 VLM/grounding；Molmo2-ER 面向机器人 ER mid-training。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Allen Institute for AI（Ai2） |
| **下游** | [MolmoAct2](./paper-molmoact2.md) action expert |
| **开源** | **已开源** ER 数据集；权重随 MolmoAct2 HF collections |

## 结论

**读 MolmoAct2 应把 Molmo2-ER 当作独立 ER 层——不是普通 Molmo2 微调别名。**

- ER mid-training 与动作 SFT/post-training 分阶段
- 与 LightNav-ER 八项 benchmark 可对照读空间能力来源
- 开源 ER 数据支持复现 ablation

## 关联页面

- [MolmoAct2](./paper-molmoact2.md)
- [Molmo2 VLM](./molmo2-vlm.md)
- [LightNav-ER](./lightnav-er.md)
- [Gemini Robotics ER](./gemini-robotics.md)

## 参考来源

- [molmoact2_arxiv_2605_02881.md](../../sources/papers/molmoact2_arxiv_2605_02881.md)
- [allenai-molmoact2.md](../../sources/sites/allenai-molmoact2.md)
- [molmo_er_molmoact2_2026.md](../../sources/papers/molmo_er_molmoact2_2026.md)

## 推荐继续阅读

- [MolmoAct2 博客](https://allenai.org/blog/molmoact2)
- [HF Molmo2-ER datasets](https://huggingface.co/collections/allenai/molmo2-er-datasets-69f8d605d92d46a5fc24ced2)
