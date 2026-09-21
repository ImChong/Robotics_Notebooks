---
type: entity
tags: [paper, dataset, egocentric, human-motion, project-aria, meta, motion-language]
status: complete
updated: 2026-09-21
arxiv: "2406.09905"
venue: "ECCV 2024"
code: https://github.com/facebookresearch/nymeria_dataset
related:
  - ./nymeria-dataset.md
  - ./light-o1.md
  - ./paper-egoexomocap.md
  - ../methods/egoscale.md
sources:
  - ../../sources/papers/nymeria_arxiv_2406_09905.md
  - ../../sources/sites/nymeria-dataset-projectaria.md
  - ../../sources/repos/nymeria_dataset.md
summary: "Nymeria（ECCV 2024，arXiv:2406.09905）：Aria+miniAria+XSens+observer 同步野外 egocentric 人类 motion 与层级 motion-language；论文演示 tracking/synthesis/recognition SOTA 评测。"
---

# Nymeria（论文）

**Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild**（Ma et al.，[arXiv:2406.09905](https://arxiv.org/abs/2406.09905)，ECCV 2024）介绍同名数据集的设计、采集协议与基准实验。数据产品细节见独立节点 [Nymeria Dataset](./nymeria-dataset.md)。

## 一句话定义

**第一篇系统描述「多 egocentric 设备 + 全身 GT + observer + 层级语言」野外人类 motion 超数据集的 ECCV 论文，并给出 egocentric 理解任务上的 SOTA 对照实验。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴 sensing |
| GT | Ground Truth | XSens 惯导 + Momentum 人体运动真值 |
| MPS | Machine Perception Services | Aria 6DoF/点云/眼动深度管线 |
| SMPL | Skinned Multi-Person Linear Model | 线性 blend skin 人体参数化 |
| MPJPE | Mean Per-Joint Position Error | 关节位置误差（tracking 常用） |

## 为什么重要

- **数据范式：** 把 egocentric **video + eye + IMU + wrist + exo observer + GT body** 绑进 **同一 3D 世界**——后续 ego-exo fusion（如 [EgoExoMoCap](./paper-egoexomocap.md)）常以此为训练/评测锚。
- **Motion-language 规模：** 当时 **最大** in-the-wild motion-language 之一（310.5K 句）。
- **具身 scaling：** [Light-O1](./light-o1.md) 将其作为 **人类 egocentric 适配** 曲线，验证 human action prior 跨视角迁移。

## 核心信息

| 字段 | 内容 |
|------|------|
| **机构** | Meta（Project Aria / Reality Labs） |
| **规模** | 300 h · 1200 seq · 264 participants · 50 locations |
| **语言** | 310.5K sentences · 8.64M words |
| **开源** | **已开源** 数据 + [nymeria_dataset](https://github.com/facebookresearch/nymeria_dataset) 工具 |

## 方法贡献（相对「只有视频的数据集」）

1. **硬件同步栈：** Aria + **miniAria** 腕带 + **XSens** + **observer** → 统一坐标 + 时间对齐。
2. **Momentum 重定向：** 将 skeleton motion 映射到 **参数化人体模型** 便于 learning-friendly 表示。
3. **层级语言协议：** motion narration → atomic action → activity summary，**in-context** 观看 ego+exo+motion 渲染后口述。
4. **Benchmark 演示：** 对 **egocentric body tracking / motion synthesis / action recognition** 跑 SOTA，证明数据增益。

## 实验与评测（论文级）

- 论文在三大任务上 **对比 contemporary SOTA**（具体数值以 PDF Table 为准）；核心信息是 **多模态 egocentric + GT** 相对单模态的增益。
- 后续工作（EgoExoMoCap 等）常在 **Nymeria 子集** 上报告 MPJPE / recognition 指标——本页不重复搬运全部表格。

## 结论

**Nymeria 的价值是把「野外人类 motion」从单一 MoCap 或单一 ego 视频，升级为可学习的「多设备 + 语言 + metric 3D」统一资产。**

1. **同步多设备** 是数据层创新，而非仅堆小时数。
2. **Motion-language 层级** 使 long-horizon activity 与 fine-grained motion 可同一资产检索。
3. **开放工具链**（Downloader + API）降低 egocentric 研究门槛。
4. **机器人读法：** 作 **human prior / scaling 探针**（Light-O1），非直接 robot demonstration。
5. **NymeriaPlus** 升级 bbox/ShapeR/音频等——实验需 **固定版本**。
6. **许可 NC** 限制商用 pipeline 直接复用。

## 源码运行时序图

**不适用（数据集论文）** — 运行时路径见 [nymeria_dataset](../../sources/repos/nymeria_dataset.md)：`Explorer JSON → aria_dataset_downloader → 本地序列 → 可视化/训练脚本`。

## 关联页面

- [Nymeria Dataset（产品页实体）](./nymeria-dataset.md)
- [Light-O1](./light-o1.md) — Transfer Scaling 适配轴
- [EgoExoMoCap](./paper-egoexomocap.md) — 下游 ego-exo 动捕

## 参考来源

- [nymeria_arxiv_2406_09905.md](../../sources/papers/nymeria_arxiv_2406_09905.md)
- [nymeria-dataset-projectaria.md](../../sources/sites/nymeria-dataset-projectaria.md)
- 论文：<https://arxiv.org/abs/2406.09905>

## 推荐继续阅读

- [ECCV 2024 Paper PDF](https://arxiv.org/pdf/2406.09905)
- [Dataset Explorer](https://www.projectaria.com/datasets/nymeria/)
