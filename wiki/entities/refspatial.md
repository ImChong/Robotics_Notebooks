---
type: entity
tags:
  - benchmark
  - spatial-reasoning
  - referring-expression
  - open-source
status: complete
updated: 2026-09-23
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./paper-roborefer.md
  - ./robospatial.md
  - ./lightnav-er.md
  - ../overview/lightorigins-3blogs-technology-map.md
sources:
  - ../../sources/papers/roborefer_arxiv_2506_04308.md
  - ../../sources/papers/refspatial_lightnav_2026.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "RefSpatial-Bench：RoboRefer 空间指代 benchmark；LightNav-ER 八项之一；HF BAAI/RefSpatial-Bench 已发布。"
---

# RefSpatial-Bench

**RefSpatial-Bench** 是 [RoboRefer](./paper-roborefer.md) 发布的 **空间指代** 评测（HF [`BAAI/RefSpatial-Bench`](https://huggingface.co/datasets/BAAI/RefSpatial-Bench)），也被 [LightNav-ER](./lightnav-er.md) 八项具身推理套件引用（[LightNav-0 博客](https://www.lightorigins.com/blog/lightnav-0) 脚注 [3]）。

## 一句话定义

**RefSpatial-Bench 测 VLM 能否在复杂场景里「指对你说的那个 3D 位置」——带推理而不只是单次 pointing。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| REF | Referring Expression | 指代表达 |
| VLM | Vision-Language Model | 视觉-语言模型 |
| ER | Embodied Reasoning | 具身推理 mid-training 评测轴 |

## 为什么重要

- **产业横评：** Qwen3-VL、Gemini Robotics 1.5 技术报告采用 RefSpatial-Bench。
- **Expand-Bench：** 扩展室内工厂/商店与 **室外** 街景/停车场（2025-10 发布）。
- **与 RoboRefer 绑定：** 训练数据 RefSpatial + 模型权重见 [RoboRefer 项目页](https://zhoues.github.io/RoboRefer/)。

## 核心信息

| 项 | 内容 |
|----|------|
| **canonical 论文** | [RoboRefer arXiv:2506.04308](./paper-roborefer.md) |
| **开源** | **已发布** Bench + Expand-Bench on HuggingFace |

## 结论

**读 LightNav-ER 的 RefSpatial 子项时，应指向 RoboRefer 官方 RefSpatial-Bench 协议与 Expand 版本。**

- 空间指代 + 推理；非纯 2D Point-Bench
- Expand-Bench 补室外——旧分数不可与新 split 直接比
- 与 [RoboSpatial-Home](./robospatial.md) POI/VQA 互补

## 关联页面

- [RoboRefer](./paper-roborefer.md)
- [RoboSpatial](./robospatial.md)
- [Point-Bench](./er-point-bench.md)
- [LightNav-ER](./lightnav-er.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页归其 ① 认知评测层：空间指代 benchmark，认知分仍需往 ③ 策略成功率压实

## 参考来源

- [roborefer_arxiv_2506_04308.md](../../sources/papers/roborefer_arxiv_2506_04308.md)
- [refspatial_lightnav_2026.md](../../sources/papers/refspatial_lightnav_2026.md)

## 推荐继续阅读

- [RefSpatial-Bench（HF）](https://huggingface.co/datasets/BAAI/RefSpatial-Bench)
- [RoboRefer 项目页](https://zhoues.github.io/RoboRefer/)
