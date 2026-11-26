---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-precise-affordances-from-egocentric-videos-for
url: "https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learning_Precise_Affordances_from_Egocentric_Videos_for_Robotic_Manipulation_ICCV_2025_paper.html"
venue: "ICCV 2025"
summary: "ICCV 2025 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_learning-precise-affordances-from-egocentric-videos-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# GAT

**Learning Precise Affordances from Egocentric Videos for Robotic Manipulation**（ICCV 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning Precise Affordances from Egocentric Videos for Robotic Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
| 出处 | ICCV 2025 |
| 方法名（清单标注） | GAT |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learning_Precise_Affordances_from_Egocentric_Videos_for_Robotic_Manipulation_ICCV_2025_paper.html> |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读 CVF 开放获取原文全文核对补充。

论文提出覆盖**自动数据采集 → 几何引导分割模型（GAT） → 实机操作框架（Aff-Grasp）**的完整系统，同时建模物体的**可抓取 affordance（graspable，如刀柄）**与**功能性 affordance（functional，如刀刃）**：

1. **第一人称视频自动标注管线（Sec. 3.1）**：
   - **输入与预处理**：从 EPIC-Kitchens 与 Ego4D 的动作旁白检索手–物交互（HOI）与工具–物交互片段，使用手–物检测器与 EfficientSAM 生成接触帧手部掩码与边界框。
   - **前接触帧投影与点采样**：取接触帧前最后一帧无遮挡的 **pre-contact frame**；抓取点在手掩码与物体框交集采样后经单应性变换（homography）投影回 pre-contact frame；功能点在工具与目标物体掩码边界框 IoU 低于阈值的前接触帧中取工具腐蚀掩码内距目标物体最近的点（缺省工具交互片段时改用距抓取点最远点采样），再经特征点对应关系映射回手–物交互 pre-contact frame。
   - **掩码生成**：将抓取点与功能点互为正/负点提示输入 **SAM**，生成像素级可抓取/功能分割掩码并裁剪物体图像作为训练集。
2. **Geometry-guided Affordance Transformer（GAT，Sec. 3.2）**：
   - **编码与微调**：以冻结的 **DINOv2** 图像编码器（分为 4 个 block）提取视觉特征，并在其上注入可训练 **LoRA** 低秩适配层（$W_0 + BA$）以缓解低分辨率裁剪与跨域差异。
   - **Depth Feature Injector（DFI）**：训练阶段先用 Depth-Anything 生成伪深度图，经 Stem（3 个 3×3 卷积 + 1×1 卷积）得到深度特征 $F_d$；在每个 block 起始处以图像特征 $F_i$ 为 Query、$F_d$ 为 Key/Value 做交叉注意力，并通过初始化为 0 的可学习缩放向量 $\beta \in \mathbb{R}^C$ 残差融合（$\hat{F}_i = \beta \cdot \text{softmax}(QK^T/\sqrt{d_k})V + F_i$）。DFI 仅在训练时起到几何正则化作用，推理时可移除以加速。
   - **分类头与训练目标**：输出特征经 MLP embedder 与 ×4 上采样后，与可学习类别嵌入 $M \in \mathbb{R}^{L\times C}$（开放词汇推理时可替换为 CLIP 文本嵌入）计算余弦相似度并双线性插值回原图分辨率，同时用阈值 $\tau$ 隐式判别背景（避免显式背景类过拟合）。针对前景–背景极度不平衡，采用混合损失 $L = \alpha \cdot L_{\text{focal}} + L_{\text{dice}}$。
3. **Aff-Grasp 实机操作框架（Sec. 3.3）**：
   - 给定「动词 + 目标」指令（如 *cut cake*），先用开放词汇检测器 Grounding DINO 定位目标物体及类无关候选物体（`"objects"`），裁剪送入 GAT 预测各物体对应动作的置信度与抓取/功能区域掩码；选定目标工具后由 **Contact-GraspNet** 结合深度图在可抓取掩码（人机递交任务则在功能掩码）内生成 6-DoF 抓取候选并择优执行，随后按动作原语完成工具–物体交互或递交。

### 方法流程图

```mermaid
flowchart LR
    EgoVid["第一人称视频 EPIC-Kitchens / Ego4D"] --> HOIDet["手-物检测与 EfficientSAM 分割"]
    HOIDet --> PreFrame["单应性投影与点对应映射至 pre-contact 帧"]
    PreFrame --> SAMGen["SAM 正负点提示生成抓取与功能掩码"]
    SAMGen --> TrainImg["裁剪物体图像与伪深度图 Depth-Anything"]
    TrainImg --> DINO["DINOv2 图像编码器 + LoRA 微调"]
    TrainImg --> DFIStem["Depth Stem 编码与 DFI 跨注意力注入"]
    DFIStem --> DINO
    DINO --> CosineHead["MLP Embedder 上采样与余弦相似度分类"]
    CosineHead --> Loss["Focal + Dice 损失训练"]
    SceneRGBD["实机 RGB-D 场景与动词-目标指令"] --> GDINO["Grounding DINO 开放词汇物体检测"]
    GDINO --> GATInf["GAT 推理抓取与功能掩码"]
    CosineHead -.-> GATInf
    GATInf --> GraspNet["Contact-GraspNet 6-DoF 抓取生成与动作原语执行"]
```

## 评测与指标

- **视觉跨域分割基准（AED，论文 Sec. 4.1 / Table 1）**：作者人工标注包含 **721 张图像、13 类物体、8 类 affordance** 的 Affordance Evaluation Dataset（AED），全程采用零样本/跨域设定评估；GAT 取得 **68.62 mIoU / 81.09 F1 / 83.51 Acc**，相比此前最优 OOAL（54.82 mIoU / 70.58 F1 / 68.00 Acc）提升 +13.80 mIoU，并显著优于 ViT-Adapter（50.86 mIoU）、DINOv2（46.16 mIoU）、SegFormer（23.72 mIoU）与 ZegCLIP（18.33 mIoU）。
- **模块消融与推理效率（Table 4 & Table 5）**：在基础管线上加入仅训练期使用的 DFI 可将 mIoU 从 56.70 提至 60.15，全程启用 DFI 达 64.66，叠加 LoRA 达 68.62；推理阶段关闭 DFI 可使参数量从 96.9M 降至 91.7M、单帧延迟从 10.1 ms 降至 6.3 ms（-37.6%）。
- **真机操作实验（Sec. 4.2 / Table 2 & Table 3）**：在 **7 类任务、34 件物体** 上开展测试（总计 179 次抓取试验平均成功率 77.1%）：
  - **已知类别精度（72 次试验，Table 2）**：Aff-Grasp 在 Affordance 预测准确率、抓取成功率、工具交互成功率分别达 **97.2% (70/72) / 80.6% (57/72) / 65.3% (47/72)**，优于 Robo-ABC（86.1% / 61.1% / n/a）与 LOCATE（58.3% / 45.8% / n/a）。
  - **未见物体泛化（35 次试验，Table 3）**：Affordance 准确率 **91.4% (32/35)**、抓取成功率 **80.0% (28/35)**，单次 affordance 推理耗时 **0.0063 s**（Robo-ABC 需 12.92 s、68.6% / 60.0%）；杂乱多干扰物场景下 affordance 预测准确率达 95%。
- 评测口径与完整图表以 [CVF 原文](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learning_Precise_Affordances_from_Egocentric_Videos_for_Robotic_Manipulation_ICCV_2025_paper.html) 为准；综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页已完成 CVF 原文核对：GAT / Aff-Grasp 打通了第一人称视频自动掩码采集、几何引导分割与真机 6-DoF 工具操作全链路。**

- 以 pre-contact 帧点投影 + SAM 替代粗糙高斯热力图，同时监督可抓取区域与功能区域。
- 用冻结 DINOv2 + LoRA 搭配仅训练期可用的 DFI 深度注入，在低分辨率裁剪与跨域场景下实现 68.62 mIoU（AED）且推理不增加深度分支开销（6.3 ms/帧）。
- 论文提供项目页（<https://reagan1311.github.io/affgrasp>）；复现真机部分仍需配合 Grounding DINO、Contact-GraspNet 与预设动作原语。
- 结合综述实体页对照 perception → reasoning → action 链路，定位其在「affordance 引导动作」环节的设计取舍。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_learning-precise-affordances-from-egocentric-videos-for.md`](../../sources/papers/affordance_awesome_learning-precise-affordances-from-egocentric-videos-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learning_Precise_Affordances_from_Egocentric_Videos_for_Robotic_Manipulation_ICCV_2025_paper.html>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learning_Precise_Affordances_from_Egocentric_Videos_for_Robotic_Manipulation_ICCV_2025_paper.html)
