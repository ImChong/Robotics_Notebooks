---
type: method
tags: [skeleton, action-recognition, open-vocabulary, language-alignment, heterogeneous, contrastive-learning, transformer]
status: complete
updated: 2026-05-01
related:
  - ./imitation-learning.md
  - ./claw.md
  - ./vla.md
  - ./motion-retargeting-gmr.md
  - ./diffusion-motion-generation.md
  - ../concepts/foundation-policy.md
  - ../concepts/data-flywheel.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/universal_skeleton.md
summary: "骨架动作识别通过对关节序列建模来理解人体/机器人动作；HOVL 进一步解决了异构骨架和开放词汇两大难题，是跨形态动作理解的关键方法。"
---

# 骨架动作识别（Skeleton-Based Action Recognition）

**骨架动作识别**：以关节坐标序列（3D/2D 骨架）为输入，识别或描述人体 / 机器人正在执行的动作类别，是模仿学习数据质量和跨形态泛化的重要支撑技术。

## 一句话定义

输入一段骨架关节序列，输出动作类别标签或自然语言描述——在机器人场景中尤指能跨不同机器人形态和未见动作类别泛化的统一识别能力。

## 为什么重要

- **数据标注流水线**：自动识别演示数据中的动作类别，直接服务于模仿学习（IL）数据引擎
- **跨形态迁移**：人形机器人种类繁多（宇树 G1、ALOHA、Atom01 等），骨架拓扑各异；统一表示是跨机器人知识迁移的基础
- **语言对齐**：将动作与自然语言绑定，使 VLA / CLAW 这类数据生成管线能生产语义准确的配对数据
- **开放词汇泛化**：真实部署场景动作类别无限，模型需识别训练集外的新动作

## 主要技术路线

骨架动作识别方法演化经历三个阶段：

**路线 1：基于图卷积（GCN）**
- 将骨架建模为图（节点=关节，边=骨骼连接），用时空图卷积提取动作特征
- 代表：ST-GCN（AAAI 2018）、CTR-GCN、MS-G3D
- 局限：骨架拓扑固定、仅支持封闭词汇

**路线 2：基于 Transformer**
- 用自注意力机制全局建模关节间关系，取代局部图卷积
- 代表：PoseConv3D、Skeleton-MAE
- 局限：仍受固定词汇限制，跨数据集泛化差

**路线 3：语言对齐 + 异构统一（当前前沿）**
- 引入 CLIP 等视觉-语言预训练模型，将动作与文本联合嵌入
- 同时处理不同骨架拓扑（异构统一表示）
- 代表：HOVL（2026）
- 面向 [基础策略](../concepts/foundation-policy.md) 和 [数据飞轮](../concepts/data-flywheel.md) 场景设计

## 传统方法 vs 现代方法

| 维度 | 传统方法 | 现代方法（HOVL 等） |
|------|---------|----------------|
| 骨架格式 | 单一固定拓扑（如 NTU 25 关节） | 异构多拓扑统一表示 |
| 词汇范围 | 封闭集（固定 N 类） | 开放词汇（零样本新类别） |
| 文本利用 | 无 | CLIP 对比学习对齐 |
| 跨数据集 | 几乎不支持 | HOV Dataset 多源整合 |
| 代表方法 | ST-GCN、CTR-GCN、PoseConv3D | HOVL（2026） |

## HOVL：异构开放词汇学习

**HOVL（Heterogeneous Open-Vocabulary Learning）** 是目前最系统解决上述两大限制的方法，出自 Kuang et al.（2026）：

### 架构三组件

```
骨架序列（异构）
     ↓
[统一骨架表示模块]      ← 消除拓扑异构性
     ↓
[多流时空动作编码器]    ← Transformer，提取多模态骨架嵌入
     ↓
[多粒度动作-文本对齐]   ← 基于 CLIP 的三层对比学习
     ↓
动作类别 / 自然语言描述
```

### 多粒度对齐：三层对比学习

**层 1 — 全局实例对齐**
- 整段动作序列嵌入 ↔ 动作类别文本嵌入
- 对应传统 CLIP 图像-文本全局匹配

**层 2 — 流级对齐**
- 各骨架流（3D 位置流、2D 姿态流等）分别与文本模态对齐
- 保留各流细节，避免信息在全局融合时损失

**层 3 — 细粒度对齐**
- 帧级 / token 级精细对应
- 捕捉"抬腿"对应"leg lift"、"挥手"对应"wave"等局部语义

### HOV 数据集

整合 NTU RGB+D 120（3D + 2D 双格式）与 HumanML3D，构建大规模异构基准：
- 覆盖 120+ 动作类别
- 长尾多标签评测设置，贴近真实分布
- 同时包含人体骨架与运动描述文本

## 与机器人技术栈的连接

### 1. 模仿学习数据质量
演示数据中的动作分割 + 自动标注依赖骨架动作识别精度。HOVL 的开放词汇能力使得数据引擎能处理任意新动作类型，直接提升 [模仿学习](./imitation-learning.md) 数据集质量。

### 2. CLAW 数据引擎
[CLAW](./claw.md) 通过语言描述标注全身动作数据；骨架动作识别提供反向验证：确认生成动作与语言标签的语义一致性。两者互补。

### 3. VLA 语言对齐
[VLA](./vla.md) 需要精准的动作-语言对应数据。HOVL 的多粒度对齐产生了帧级动作-文本配对，可作为 VLA 训练的高质量输入。

### 4. 跨形态迁移
骨架统一表示思路与 [运动重定向](./motion-retargeting-gmr.md) 高度互补：重定向解决关节运动的几何映射，统一骨架表示解决特征空间的语义对齐。

## 局限性

- **计算开销**：三层对比学习的多流 Transformer 比单流 GCN 计算量高
- **3D 骨架质量依赖**：真实机器人上需要可靠的关节估计（IMU / 编码器），精度直接影响识别效果
- **机器人场景验证**：当前工作主要在人体骨架数据集上验证，直接应用于机器人骨架需要进一步适配

## 参考来源

- [sources/papers/universal_skeleton.md](../../sources/papers/universal_skeleton.md) — Kuang et al. (2026) HOVL 原论文 ingest 归档
- Kuang et al., *Universal Skeleton-Based Action Recognition: Heterogeneous Open-Vocabulary Learning via Multi-Grained Motion-Text Alignment*, arXiv:2604.17013
- Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (ST-GCN, AAAI 2018) — 传统 GCN 路线代表
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, ICML 2021) — 语言对齐基础

## 关联页面

- [Imitation Learning](./imitation-learning.md) — 骨架动作识别是 IL 数据标注的上游技术
- [CLAW](./claw.md) — 语言标注动作数据生成管线，与 HOVL 互补
- [VLA](./vla.md) — 语言-动作对齐的下游应用场景
- [Motion Retargeting GMR](./motion-retargeting-gmr.md) — 跨骨架形态运动映射
- [Diffusion Motion Generation](./diffusion-motion-generation.md) — 动作生成的下游应用

## 推荐继续阅读

- [Universal-Skeleton GitHub](https://github.com/jidongkuang/Universal-Skeleton)
- [arXiv:2604.17013](https://arxiv.org/abs/2604.17013)
- [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) — 骨架 GCN 经典基线
