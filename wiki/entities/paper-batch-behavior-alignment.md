---
type: entity
project_id: batch-behavior-alignment
tags:
  - paper
  - cross-embodiment
  - human-video
  - imitation-learning
  - video-action-model
  - manipulation
  - stanford
status: complete
updated: 2026-11-10
arxiv: "2611.10148"
project: https://batch-embodiment.github.io/
original:
  - ../../sources/papers/batch-behavior-alignment_arxiv_2611_10148.md
related:
  - ../methods/imitation-learning.md
  - ../methods/behavior-cloning.md
  - ../methods/diffusion-policy.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../concepts/world-action-models.md
  - ../tasks/manipulation.md
  - ./paper-egoverse.md
  - ./paper-ego-03-egomimic.md
  - ./paper-sa-2509-19626-egobridge-domain-adaptation-for-generalizable-im.md
  - ./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md
  - ./paper-notebook-humanoid-policy-human-policy.md
sources:
  - ../../sources/papers/batch-behavior-alignment_arxiv_2611_10148.md
summary: "BATCH（Stanford，arXiv:2611.10148）：用 GTCC 时间循环一致性嵌入加子序列 DTW 自动对齐人机示范阶段，再在视频-动作扩散模型中做对齐联合训练与人到机器人风格迁移；两项真机任务 OOD 优于 EgoVerse 式协同训练，但仍需机器人示范且代码未放出。"
---

# BATCH：跨本体人类示范的行为对齐 / Behavior Alignment for Cross-Embodiment Learning from Humans

**论文：** [BATCH（arXiv:2611.10148）](https://arxiv.org/abs/2611.10148)，Zeyi Liu、Shuran Song，Stanford University（论文首页），v1 于 2026-11-06｜[项目页](https://batch-embodiment.github.io/)｜小站 2026-11-10 更新收录。

## 一句话定义

**BATCH 先用自监督的时间循环一致性嵌入加子序列 DTW 在无标注下找出人类与机器人示范里「做到同一阶段」的帧，再把对齐后的人机配对喂给视频-动作扩散模型，做联合训练与「人类视频到机器人视频」的风格迁移。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BATCH | Behavior Alignment for Cross-Embodiment Learning from Humans | 本文方法名 |
| TCC | Temporal Cycle-Consistency | 用循环最近邻一致性学习视频帧对齐嵌入（Dwibedi 等，2019） |
| GTCC | Gaussian-mixture TCC（论文称 Donahue & Elhamifar 的多邻居多循环版本） | 用高斯混合保留多个候选对应，缓解重复状态 |
| DTW | Dynamic Time Warping | 动态时间规整，允许非均匀时间伸缩的序列对齐 |
| OT | Optimal Transport | 最优传输，论文用作对照的表征对齐损失 |
| OOD / ID | Out-of-Domain / In-Domain | 域外 / 域内场景 |
| VAE | Variational Autoencoder | 视频压缩编码器 |
| CLIP | Contrastive Language-Image Pre-training | 论文用 CLIP ViT-H/14 编码首帧 |

## 为什么重要

- 人类示范便宜且场景多样，但与机器人在速度、时间结构、视角、动作空间上都不同；论文引言（§1）指出盲目增加错配的人类数据可能增益递减甚至负迁移，并以此提出「先显式建立行为对应」的假设（该假设是作者的主张，不是定论）。
- 对齐发生在时间轴上，且不要求人工阶段标签或逐帧匹配，可以接入已有的协同训练流程（§2）。
- 与本库的 [EgoVerse](./paper-egoverse.md)（数据集与协同训练基线）、[EgoBridge](./paper-sa-2509-19626-egobridge-domain-adaptation-for-generalizable-im.md)（OT 式对齐）互补：BATCH 的对照基线直接取自这两条线（§4.2）。

## 核心机制

### 流程总览

```mermaid
flowchart LR
  H[人类示范 Aria 头戴视角] --> F[冻结 ResNet-50 特征]
  R[机器人遥操作示范] --> F
  F --> G[GTCC 编码器 128 维嵌入]
  G --> D[子序列 DTW 检索 top-k 人类片段]
  D --> P[对齐的人机配对]
  P --> C[联合协同训练 L_joint]
  P --> S[风格迁移 L_ST]
  C --> M[视频-动作扩散模型]
  S --> M
  M --> A[部署 只用机器人流]
```

### 1. 行为对齐（论文 §3.1）

- 特征：冻结 ImageNet ResNet-50 逐帧取 1024×14×14，因果时间窗堆叠后经两层 3D 卷积、时空最大池化与三层 MLP 得 128 维嵌入。
- GTCC：每个机器人查询帧对人类序列做软最近邻分布，再用 K 分量高斯混合近似，保留多个对应假设；损失为 `Σ_k π_ik (i-μ̂_ik)² + λ log σ̂_ik`。辅助头预测相关度 `b_i`，无对应的帧被降权。
- 检索：子序列 DTW 在嵌入上找代价最小的人类子段，按归一化代价取 top-k（Object in Container 为 5，Cup on Saucer 为 1，附录 B.1）；训练时沿 warping path 把人类帧、状态、动作「时间规整」到机器人窗口上。

### 2. 视频-动作模型（§3.2）

- 骨架：Wan2.1-I2V-14B 的前 8 个 Transformer 块、视频 VAE、CLIP ViT-H/14；单任务故去掉文本条件。样本含 9 帧头部相机、16 步动作块、本体状态与各腕部相机的一帧条件。
- 人类流与机器人流共享权重但注意力隔离，人类流只有头部相机，腕部区域补零；RoPE 索引与 CLIP 交叉注意力按本体分开，所以推理时去掉人类流不改变机器人流计算。

### 3. 两个微调目标（§3.2.1）

- **联合协同训练：** `L_joint = L_r_vid + λa L_r_act + λhv L_h_vid + λha L_h_act`，人类通过共享参数影响机器人通路。
- **风格迁移：** 把完整人类片段作干净上下文，预测对应机器人的未来视频；遮蔽机器人条件潜变量并把动作 / 状态寄存器置零，防止模型绕过人类流；只监督机器人视频。
- 训练：机器人预训练 20,000 步，随后微调 1,000 步，每步各以 0.5 概率选两目标之一（附录 B.2.3）。

## 实验与结果（作者报告，均为论文数字）

| 项 | 数值 | 出处 |
|----|------|------|
| 对齐 Object in Container，GTCC+DTW 阶段重叠 ID / OOD | 69.0% / 57.8% | Table 1 |
| 同上，去掉 DTW 的 OOD 重叠 | 24.7%（进度误差 .098 升至 .231） | Table 1 |
| 对齐 Cup on Saucer，GTCC+DTW OOD 重叠 vs DINOv3+DTW | 71.7% vs 22.1% | Table 1 |
| Cup on Saucer 最佳对齐 | GTCC+绝对位姿：ID 79.2%、OOD 76.5% | Table 1 |
| Object in Container 归一化分数：OURS / OT / 无对齐 / EgoVerse / 仅机器人 | 0.63 / 0.50 / 0.42 / 0.40 / 0.28 | Figure 4a |
| Cup on Saucer 成功率：同上 | 83.3% / 63.3% / 50.0% / 63.3% / 33.3% | Figure 4b |
| 作者汇总 | 较协同训练平均 +45%，较仅机器人预训练平均 +138% | §1 |

读法说明：Figure 4 的柱子与方法对应来自文本提取的位置与正文描述，源归档已注明；两任务的评测规模为 20 次与 30 次滚动（§4.2），未报告方差或置信区间（我的观察）。

## 工程实践

- **硬件与数据（附录 A、Table 2）：** 人类用 Aria Gen 1 眼镜（MPS 手位姿，缺夹爪宽度，补零并不计入人类动作损失）；机器人是两个 ARX5 臂，头部装 Aria Gen 1、腕部装罗技摄像头；机器人预训练演示 90 条（单臂）与 270 条（双臂），人类微调每场景仅 5 条与 8 条。
- **算力（附录 B）：** 预训练用 4 张 RTX PRO 6000 Blackwell 96GB；GTCC 在 N=M=30、T≈200 时约 100 s / epoch，200 epoch 约 5.5 GPU 小时；检索在单 CPU 核上数分钟。
- **推理（附录 B.2.4）：** 16 步动作块执行前 8 步，RTX 5090 延迟 0.35 s（单臂）/ 0.42 s（双臂）。
- **复现：** 项目页 Code 标 coming soon，论文称接受后释出代码、数据与检查点；当前不可运行（2026-11-10 核查）。

## 局限

**论文自述（§4.2 Limitation）：**
- 假设人机共享行为结构，策略或动作顺序差异大时可能失败。
- 对齐误差会把错配监督带入风格迁移目标。
- 后训练阶段仍需机器人示范。

**我的观察（未经论文验证）：**
- 仅两项任务、单一机器人平台，且人类数据只有每场景 5 至 8 条，结论是否推广到更多任务与更大人类数据规模，论文未验证。
- 仅在文内评测小规模滚动，无统计置信度；Object in Container 中「无对齐」（0.42）仅略高于 EgoVerse（0.40），说明对齐带来的增益与风格迁移带来的增益在该任务上难以分离（推测）。
- 对齐评测的 GTCC 与基线对比有位姿特征消融，但没有在策略层面单独消融「GTCC 对齐 vs 位姿对齐」，因此「GTCC 对齐更好」只在 Table 1 的对齐指标层面成立。
- 系统依赖大型视频骨干（14B 的前 8 块起步）与 DreamZero 代码库，不是轻量方案；这是对架构的描述性判断。

## 与小站说法的核对

- 小站「用时间循环一致性和动态时间规整对齐…联合训练与风格迁移」与 §3 一致；「两项真机任务显示收益」与 §4.2 一致；「依赖机器人示范及相近的行为结构」与 Limitation 一致。机构 Stanford University 与论文首页一致。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [行为克隆](../methods/behavior-cloning.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)
- [World Action Models](../concepts/world-action-models.md)
- [操作任务](../tasks/manipulation.md)
- [EgoVerse](./paper-egoverse.md)
- [EgoMimic](./paper-ego-03-egomimic.md)
- [EgoBridge](./paper-sa-2509-19626-egobridge-domain-adaptation-for-generalizable-im.md)
- [DreamZero](./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md)
- [Humanoid Policy ~ Human Policy](./paper-notebook-humanoid-policy-human-policy.md)

## 参考来源

- [batch-behavior-alignment_arxiv_2611_10148.md](../../sources/papers/batch-behavior-alignment_arxiv_2611_10148.md)
- [arXiv:2611.10148](https://arxiv.org/abs/2611.10148)
- [项目页](https://batch-embodiment.github.io/)

## 推荐继续阅读

- [arXiv PDF v1](https://arxiv.org/pdf/2611.10148v1)
- [EgoVerse 论文页](./paper-egoverse.md)
- [World Action Models](../concepts/world-action-models.md)
