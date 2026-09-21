---
type: entity
tags:
  - benchmark
  - spatial-reasoning
  - embodied-reasoning
  - open-source
status: complete
updated: 2026-09-21
arxiv: "2406.05756"
code: https://github.com/mengfeidu/EmbSpatial-Bench
related:
  - ./robospatial.md
  - ./paper-cambrian-1.md
  - ./lightnav-er.md
  - ../overview/spatial-reasoning-benchmarks-technology-map.md
sources:
  - ../../sources/papers/embspatial_bench_arxiv_2406_05756.md
  - ../../sources/papers/embspatial_lightnav_2026.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "EmbSpatial-Bench（arXiv:2406.05756）：egocentric 六类空间关系 benchmark；LightNav-ER 套件成员；已开源 mengfeidu/EmbSpatial-Bench。"
---

# EmbSpatial-Bench

**EmbSpatial-Bench**（[arXiv:2406.05756](https://arxiv.org/abs/2406.05756)，[代码](https://github.com/mengfeidu/EmbSpatial-Bench)）评测 **LVLM 在具身任务中的空间理解**：从 embodied 场景 **自动派生** egocentric 视角下的 **6 种空间关系** QA。

## 一句话定义

**EmbSpatial 问：LVLM 在「第一人称具身视角」下懂不懂 left-of / inside / closer 这类关系。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LVLM | Large Vision-Language Model | 大视觉-语言模型 |
| ER | Embodied Reasoning | 具身推理；LightNav-ER 八项之一 |
| VQA | Visual Question Answering | 视觉问答 |

## 为什么重要

- **LightNav-ER 套件：** [LightNav-0](./paper-lightnav-0.md) 博客脚注 [3] 列入八项 ER 评测（见 [lightnav-er](./lightnav-er.md)）。
- **自动构造：** 无需人工逐条标注，可扩展 embodied 场景库。
- **与 Cambrian CV-Bench 区分：** Cambrian 的 CV-Bench 偏 2D 空间；EmbSpatial 强调 **egocentric embodied** 设定。

## 核心信息

| 项 | 内容 |
|----|------|
| **关系类型** | 6 类 egocentric 空间关系 |
| **开源** | **已开源** `mengfeidu/EmbSpatial-Bench` |

## 结论

**EmbSpatial 是「具身 egocentric 空间 VQA」早期公共基准——读 LightNav-ER 分数时应对齐其六关系设定。**

- 自动派生保证与场景库同步扩展
- 与 RoboSpatial-Home（scan QA）、RefSpatial（指代）互补
- 开源仓库可复现 leaderboard 数字

## 关联页面

- [RoboSpatial](./robospatial.md)
- [LightNav-ER](./lightnav-er.md)
- [Cambrian-1 / CV-Bench](./paper-cambrian-1.md)

## 参考来源

- [embspatial_bench_arxiv_2406_05756.md](../../sources/papers/embspatial_bench_arxiv_2406_05756.md)
- [embspatial_lightnav_2026.md](../../sources/papers/embspatial_lightnav_2026.md)

## 推荐继续阅读

- [arXiv:2406.05756](https://arxiv.org/abs/2406.05756)
- [GitHub EmbSpatial-Bench](https://github.com/mengfeidu/EmbSpatial-Bench)
