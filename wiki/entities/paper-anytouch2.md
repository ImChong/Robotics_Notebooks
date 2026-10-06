---
type: entity
tags: [paper, tactile-sensing, optical-tactile, representation-learning, force]
status: complete
updated: 2026-10-06
arxiv: "2602.09617"
venue: "ICLR 2026"
summary: "AnyTouch 2（ICLR 2026）：基于 242 万级 ToucHD 动态触觉样本，以视频掩码、语义/跨传感器匹配及力变化监督学习通用光学触觉表示。"
related:
  - ./project-anytouch2.md
  - ./paper-anytouch.md
  - ./project-anytouch.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ./paper-sparsh.md
sources:
  - ../../sources/papers/anytouch2_arxiv_2602_09617.md
---

# AnyTouch 2：动态光学触觉通用表征（ICLR 2026）

**AnyTouch 2**（*General Optical Tactile Representation Learning For Dynamic Tactile Perception*）将视觉触觉预训练重心从静态属性扩展至动态接触与力学相关表示。它提出 ToucHD 数据金字塔，并通过多类自监督与力监督目标训练表征。

| 项目 | 内容 |
|---|---|
| 作者 | Ruoxuan Feng, Yuxuan Zhou, Siyu Mei, Dongzhan Zhou, Pengwei Wang, Shaowei Cui, Bin Fang, Guocai Yao, Di Hu |
| 发表 | ICLR 2026 |
| 论文 | [arXiv:2602.09617](https://arxiv.org/abs/2602.09617) · [ICLR 论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/073c8584ef86bee26fe9d639ec648e28-Abstract-Conference.html) |
| 项目与代码 | 独立的[项目详情页](./project-anytouch2.md) |

## 英文缩写速查

| 缩写 | 全称 | 含义 |
|---|---|---|
| MAE | Masked Autoencoder | 掩码自编码器 |
| ToucHD | Tactile Understanding through Contact Hierarchy Dataset | AnyTouch 2 的动态触觉数据集 |
| RMSE | Root Mean Squared Error | 均方根误差 |

## ToucHD：由受控接触到真实操作

ToucHD 汇总 2,426,174 个触觉样本，划分为 Sim（1,118,896）、Mani（584,842）和 Force（722,436）。Sim 包含五类传感器、六种原子动作和 1,043 个物体；Mani 汇总 46 项真实操作任务；Force 配对触觉观测与接触力。数据金字塔涵盖不同动态层级：受控按压、指定滑动/旋转、预设接触动作、真实操作以及力监督。不同子集的采集协议并不相同。

## 方法：表征学习目标

1. **时序视频掩码建模**：重建触觉视频帧，并学习帧间变化。
2. **语义和匹配监督**：对齐语义描述、同物体数据及跨传感器样本。
3. **显式力目标**：预测接触力和力变化，补充仅靠视觉外观难以识别的动力学线索。

## 评测覆盖

论文覆盖静态属性、动态物理属性、传感器泛化及真实操作场景，任务包含触觉抓取、白板擦拭、USB 插入和芯片移动。指标必须连同输入帧、传感器与数据划分一起比较。AnyTouch 2 的贡献是更丰富的动态数据和目标；它不等同于可直接部署的通用操作策略。

## 方法流程

```mermaid
flowchart LR
  T["ToucHD 动态数据金字塔"] --> V["多传感器触觉片段"]
  V --> M["视频帧与帧差掩码目标"]
  V --> S["语义、物体与跨传感器匹配"]
  V --> F["接触力与力变化预测"]
  M --> R["动态触觉共享表示"]
  S --> R
  F --> R
  R --> E["静态/动态基准与操作评估"]
```

## 对比：AnyTouch 前作与 Sparsh

[AnyTouch（ICLR 2025）](./paper-anytouch.md)聚焦静态–动态统一表示、TacQuad 和文本锚定多模态对齐。AnyTouch 2 把监督范围拓展到更大规模动态接触层级和显式力变化目标。对应的软件实现、数据集和权重入口另列在[AnyTouch 2 项目页](./project-anytouch2.md)。

## 结论

AnyTouch 2 以更丰富的动态接触数据和力变化监督推进通用光学触觉表征，评测覆盖静态、动态、跨传感器与真实操作情境。仓库资产仍不等于完整真机部署栈，读取分数时应保留每个任务、传感器和输入设定。

## 关联页面

- [AnyTouch 2 项目详情](./project-anytouch2.md)
- [AnyTouch 论文与项目](./paper-anytouch.md) · [项目实现](./project-anytouch.md)
- [触觉感知](../concepts/tactile-sensing.md) · [视触觉融合](../concepts/visuo-tactile-fusion.md)
- [Sparsh](./paper-sparsh.md)

## 参考来源

- [论文来源摘录：AnyTouch 2](../../sources/papers/anytouch2_arxiv_2602_09617.md)
- [arXiv](https://arxiv.org/abs/2602.09617) · [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/073c8584ef86bee26fe9d639ec648e28-Abstract-Conference.html)
