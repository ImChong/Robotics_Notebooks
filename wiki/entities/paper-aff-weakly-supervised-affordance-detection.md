---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-weakly-supervised-affordance-detection
url: "https://openaccess.thecvf.com/content_cvpr_2017/papers/Sawatzky_Weakly_Supervised_Affordance_CVPR_2017_paper.pdf"
venue: "CVPR 2017"
summary: "CVPR 2017 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_weakly-supervised-affordance-detection.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Weakly Supervised Affordance Detection

**Weakly Supervised Affordance Detection**（CVPR 2017）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Weakly Supervised Affordance Detection——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual）**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2017 |
| 方法名（清单标注） | - |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文 | <https://openaccess.thecvf.com/content_cvpr_2017/papers/Sawatzky_Weakly_Supervised_Affordance_CVPR_2017_paper.pdf> |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读开放获取原文全文核对补充。

### CAD120 真实交互场景可供性数据集

针对早期 UMD 数据集在转盘单物体无干扰背景下的局限，本文基于 CAD120（215 段视频、8 名演员、14 种高层活动）对活跃人–物交互序列每 10 帧进行像素级标注，构建并开源了包含 **3,090 张图像、9,916 个物体实例**的厨房交互可供性数据集（沿标注边界框外扩 30 像素裁剪），涵盖 `openable`、`cuttable`、`containable`、`pourable`、`supportable`、`holdable` 6 类非互斥可供性及背景类。

### 多标签分割网络与关键点弱监督两阶段训练

1. **多标签 Sigmoid 分割头**：同一物体部件可同时具备多种可供性（如同时可握持与可支撑），故在 **VGG-16**（DeepLab 风格，无需额外 CRF）与 **ResNet-101** 上去掉互斥的 Softmax 层，改为对每个像素 $x_i$ 与每个可供性类 $l \in L$ 独立计算 Sigmoid 概率 $P(y_{i,l}|I;\theta) = \frac{1}{1+\exp(-f_{i,l}(y_{i,l}|I;\theta))}$，最大化对数似然 $J(\theta) = \sum_{i=1}^n \sum_{l\in L} \log P(y_{i,l}|I;\theta)$。
2. **关键点弱监督学习（EM 式迭代 + GrabCut 精炼）**：
   - **初始半径扩张训练**：仅给定稀疏关键点标注 $Z_x = \{(z_1, x_1), \dots\}$ 时，在距标注点 $\sigma$ 像素邻域内初始化伪标签 $\hat{y}_{i,l}=1$（UMD 取 $\sigma = 40$ px，CAD120 取 $\sigma = 50$ px），利用 ImageNet 预训练权重初始化并训练多标签 CNN。
   - **图像级门控潜在标签估计**：用第一阶段 CNN 对训练集重新预测 $P(Y_l | I; \theta)$，并结合关键点指示的图像级类别存在先验 $P(l | Z_x) \in \{0, 1\}$ 过滤不存在的类，按阈值 `0.5` 二值化得到粗略前景/背景划分。
   - **按类独立 GrabCut 掩码精炼与重训**：对每个可供性类 $l$ 独立以 6 分量高斯混合模型（GMM）建模前景（$\hat{y}_{i,l}=1$）与背景（$\hat{y}_{i,l}=0$）颜色分布，运行 **GrabCut** 生成边缘贴合的精炼伪掩码，再用精炼后的训练标签重新训练多标签 CNN（推理阶段可选再次使用 GrabCut）。

### 方法流程图

```mermaid
flowchart LR
    INP["训练图像 I 与稀疏关键点标注 Z_x"] --> INIT["邻域扩张伪标签初始化 (UMD sigma=40px / CAD120 sigma=50px)"]
    INIT --> CNN1["阶段一多标签 CNN 训练 (VGG-16 / ResNet-101 + Sigmoid 损失)"]
    CNN1 --> PROB["重算训练集概率 P(Y_l | I; theta)"]
    PROB --> GATE["关键点存在门控 P(l | Z_x) 与 0.5 二值化"]
    GATE --> GC["按类独立 GrabCut 精炼 (6 分量 GMM 前景/背景建模)"]
    GC --> CNN2["阶段二精炼掩码重训多标签 CNN"]
    CNN2 --> OUT["像素级多标签可供性预测 (可选测试期 GrabCut)"]
```

## 评测与指标

- **基准数据集与划分**：
  1. **UMD Part Affordance Dataset**（28,074 张 RGB-D，17 类物体，7 类工具部件可供性）：`category split`（类内划分）与 `novel split`（未见物体类别泛化），指标为 Ranked Weighted F-measure 与 per-class **IoU**。
  2. **CAD120 Affordance Dataset**（本文新建，3,090 图 / 9,916 实例，6 类可供性 + 背景）：`actor split`（演员 5、9 为测试集）与 `object split`（未见物体类泛化），指标为 **IoU**。
- **UMD 结果（Tables 1 & 2，Mean IoU）**：
  - 全监督（`category split`）：`Proposed (VGG)` **0.76**、`Proposed (ResNet)` **0.76**，大幅领先手工特征基线 `HMP+SVM`（0.59）与 `DEP+SRF`（0.40）。
  - 全监督（`novel split`）：`Proposed (VGG)` **0.50**、`Proposed (ResNet)` **0.50** vs `HMP+SVM`（0.38）、`DEP+SRF`（0.31）。
  - 弱监督（`category split`）：`Proposed (VGG)` **0.58**（去训练 GrabCut 跌至 `0.36`，加测试 GrabCut 升至 **0.67**）；`Proposed (ResNet)` **0.54**（去训练 GrabCut 为 `0.37`，加测试 GrabCut 升至 **0.63**）；远超对比弱监督基线 `SEC`（0.22）、`Area constraints`（0.14）、`WTP`（0.13）、`Image label`（0.10）。
  - 弱监督（`novel split`）：`Proposed (VGG)` **0.37**（+测试 GrabCut **0.48**），`Proposed (ResNet)` **0.40**（+测试 GrabCut **0.56**，甚至超过全监督 ResNet 的 0.50），大幅优于 `SEC`（0.13）与同为关键点监督的 `WTP`（0.10）。
- **CAD120 结果（Tables 3 & 4，Mean IoU）**：
  - 全监督：`actor split` 下 `VGG` **0.54**、`ResNet` **0.59**；`object split` 下 `VGG` **0.49**、`ResNet` **0.57**（跨物体泛化掉点显著小于 UMD）。
  - 弱监督（`actor split`）：`VGG` **0.32**（去训练 GrabCut `0.29`，+测试 GrabCut `0.32`）、`ResNet` **0.30**（去训练 GrabCut `0.29`，+测试 GrabCut `0.26`），优于 `SEC`（0.22）、`WTP`（0.17）、`Image label`（0.15）与 `Area constraints`（0.15）。
  - 弱监督（`object split`）：`ResNet` **0.33**（去训练 GrabCut `0.24`，+测试 GrabCut `0.30`）、`VGG` **0.27**，优于 `SEC`（0.16）、`Area constraints`（0.14）、`WTP`（0.13）与 `Image label`（0.12）。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本文用“Sigmoid 多标签分割 + 关键点初始化与训练期按类 GrabCut 精炼”，在显著降低像素标注成本的同时在 UMD 与 CAD120 上实现了优于图像级/点级弱监督基线的可供性检测精度。**

- **多标签独立 Sigmoid 不可或缺**：相比传统语义分割 Softmax，Sigmoid 对同一区域重叠多种可供性的建模是可供性分割准确率提升的基础。
- **训练期 GrabCut 的场景依赖性**：训练期使用 GrabCut 精炼伪标签在两类数据集上均稳定大幅提分；但推理期叠加 GrabCut 仅在背景单一的 UMD 转盘图上有效（ResNet `novel split` 0.40→0.56），在杂乱真实的 CAD120 厨房场景中反而下降（ResNet `actor split` 0.30→0.26）。
- **数据与代码已开源**：CAD120 可供性标注与实现代码已发布于 <https://github.com/ykztawas/Weakly-Supervised-Affordance-Detection>。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_weakly-supervised-affordance-detection.md`](../../sources/papers/affordance_awesome_weakly-supervised-affordance-detection.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://openaccess.thecvf.com/content_cvpr_2017/papers/Sawatzky_Weakly_Supervised_Affordance_CVPR_2017_paper.pdf>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://openaccess.thecvf.com/content_cvpr_2017/papers/Sawatzky_Weakly_Supervised_Affordance_CVPR_2017_paper.pdf)
