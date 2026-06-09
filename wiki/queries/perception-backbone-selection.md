---
type: query
tags: [perception, computer-vision, backbone, representation-learning, policy-input, robotics, deployment, vla]
status: complete
updated: 2026-06-09
summary: "机器人感知骨干/表征选型 Query：从「分类骨干 / 检测头 / 通用预训练表征」三类出发，给出从延迟预算、数据量与下游接法切入的选型决策树、推荐组合 pipeline 与典型失败模式。"
related:
  - ../concepts/vision-backbones.md
  - ../concepts/visual-representation-for-policy.md
  - ../comparisons/cnn-vs-vit-backbones.md
  - ../methods/object-detection.md
  - ../methods/vla.md
  - ../entities/paper-resnet-deep-residual-learning.md
  - ../entities/paper-yolo-unified-realtime-detection.md
sources:
  - ../../sources/papers/resnet_arxiv_1512_03385.md
  - ../../sources/papers/yolo_arxiv_1506_02640.md
  - ../../sources/papers/vision_backbone_detection_classics.md
---

> **Query 产物**：本页由以下问题触发：「机器人感知栈里到底该怎么选视觉骨干和表征？什么时候用一个普通分类骨干，什么时候要接检测/分割头，什么时候直接拿通用预训练表征（DINOv2 / R3M / VC-1）喂策略？选型的第一刀应该砍在哪？」
> 综合来源：[视觉骨干（概念）](../concepts/vision-backbones.md)、[视觉表征作为策略输入（概念）](../concepts/visual-representation-for-policy.md)、[CNN vs ViT 视觉骨干对比](../comparisons/cnn-vs-vit-backbones.md)、[目标检测（方法）](../methods/object-detection.md)、[VLA（方法）](../methods/vla.md)

# Query：机器人感知骨干/表征选型（分类骨干 vs 检测头 vs 通用预训练表征）

## TL;DR 决策树

```text
策略到底需要从图像里拿到「什么」？
├── 要结构化的「物体 + 位置」（避障 / 抓取候选 / 计数）
│   └→ 骨干 + 检测/分割头：ResNet-FPN / 轻量 CNN + YOLO 系
│      （延迟优先 → 单阶段量化；精度优先 → 两阶段，详见检测选型）
├── 要「整张图的语义特征」直接喂策略（端到端操作 / VLA）
│   ├── 真机数据稀缺、要快速起步
│   │   └→ 冻结通用预训练表征：DINOv2 / VC-1（+ 轻量适配头）
│   ├── 操作类、强调可迁移
│   │   └→ 机器人专用预训练表征：R3M / VC-1
│   └── 数据充足、任务单一、追求上限
│       └→ 端到端联合训练（随策略从头学骨干）
└── 只要「这是哪一类」（状态识别 / 触发判断）
    └→ 普通分类骨干（ResNet / 轻量 ViT）+ 线性头
```

## 快速结论

- **第一刀砍在「策略需要的输出形式」，不是「哪个骨干 mAP 高」**：要结构化物体就接检测/分割头，要整图语义就走表征直喂，要类别判断就普通分类头。
- **第二刀砍在「真机数据量与延迟预算」**：数据稀缺先冻结通用表征起步，数据充足且任务单一才考虑端到端；机载实时永远先锁低延迟 CNN + 量化。
- **冻结预训练表征是务实默认，不是偷懒**：在样本效率上常比从头学省一到两个量级数据，代价是要校准 **预训练域与机器人第一视角的差距**（见 [视觉表征作为策略输入](../concepts/visual-representation-for-policy.md)）。
- **骨干族（CNN vs ViT）的选择与「接什么头」正交**：先定输出形式与表征路径，再按数据量/吞吐在 [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md) 里挑骨干族。
- **2D 特征不提供几何**：无论分类骨干还是通用表征，6DoF 抓取仍要在后面级联深度/位姿头，别让骨干背锅。

## 三类选型对比

### 类 1：分类骨干（只要类别 / 全局特征）

| 角色 | 代表 | 优势 | 风险 | 何时优先 |
|------|------|------|------|---------|
| 轻量 CNN 分类 | ResNet-18/34、MobileNet | 低延迟、易量化、小数据可微调 | 全局语义弱、无定位 | 边缘状态识别、触发判断 |
| 通用 ViT 分类 | ViT / Swin | 大数据规模化、强语义 | 小数据欠拟合、算子偏重 | 服务器侧、有大数据/SSL 预训练 |

### 类 2：骨干 + 检测/分割头（要结构化物体）

| 输出诉求 | 骨干 + 头 | 下游级联 | 失败模式 |
|------|---------|---------|---------|
| 有什么 + 大致在哪 | 轻量 CNN + 单阶段（YOLO 系） | 跟踪滤波 | 量化掉点、小目标漏检 |
| 高定位精度小目标 | ResNet-FPN + 两阶段 | 显式置信度筛选 | 延迟大、链路长 |
| 逐像素掩码 | 骨干 + 实例分割头 | Mask / SAM 接力 | 边界粘连、域差距 |
| 6DoF 抓取 | 检测做 ROI | 深度/点云 → 6D 位姿 | 透明/反光件深度缺失 |

> 检测头内部「单阶段 vs 两阶段 / 机载 vs 服务器侧」的细分选型，见 [目标检测模型选型 Query](object-detection-model-selection.md)。

### 类 3：通用预训练表征（整图特征直喂策略）

| 表征来源 | 代表 | 样本效率 | 泛化 | 典型风险 |
|------|------|----------|------|----------|
| 通用 SSL 骨干（冻结） | DINOv2 / MAE-ViT | 高 | 取决于预训练域 | 网络图片域 ≠ 机器人第一视角 |
| 机器人专用预训练 | R3M / VC-1 | 高 | 对操作场景更对口 | 覆盖任务面有限、可得性弱 |
| 端到端联合训练 | 随策略从头学 | 低 | 易过拟合任务 | 视觉与控制耦合、难复用 |

## 推荐组合 pipeline

按工程化路线给出四种常见真机出镜组合：

1. **机载固定类别检测（球 / 障碍 / 人）**
   - 轻量 CNN 骨干 → 单阶段检测头 → TensorRT/INT8 → 跟踪 → 控制层
   - 见 [Humanoid Soccer](../tasks/humanoid-soccer.md)：帧率是硬约束，宁可降分辨率也别上重骨干
2. **数据稀缺的端到端操作起步**
   - 冻结 DINOv2 / VC-1 → 轻量可训练 neck/适配头 → 策略网络 → 动作
   - 兼顾稳定性与任务适配，是数据稀缺时的工程默认（见 [视觉表征作为策略输入](../concepts/visual-representation-for-policy.md)）
3. **操作类强迁移表征**
   - R3M / VC-1（具身数据预训练）→ 策略头；在抓取/操作上常优于纯 ImageNet 骨干
4. **检测 → 抓取候选**
   - 骨干 + 检测头出 ROI → 深度/点云裁剪 → [Grasp Pose Estimation](../methods/grasp-pose-estimation.md) → 碰撞检查 → IK
   - 最后几厘米切 [Visual Servoing](../methods/visual-servoing.md) / 触觉对齐

## 关键工程经验

### 1. 先定输出形式，再挑骨干

把「策略要从图像拿到什么」想清楚，骨干与头的组合就基本确定了。颠倒顺序——先盯榜单挑最强骨干，再硬接下游——是最常见的过度工程来源。

### 2. 冻结表征的域差距要在选型阶段预判

冻结一个网络图片预训练的骨干，遇到机器人第一视角（鱼眼、近距离、运动模糊）可能直接拖累。选型时就该评估预训练域与目标域差距，必要时领域内微调或换专用表征（R3M / VC-1）。

### 3. 样本效率 vs 上限的取舍是数据量函数

数据稀缺时冻结表征几乎总赢；数据极充足且任务单一时端到端能逼近上限但牺牲复用。中间地带（冻结骨干 + 可训练 neck）覆盖大多数真机项目。

### 4. 2D 特征不给几何，6DoF 必须级联

无论分类骨干还是通用表征，输出多为 2D 语义特征，缺空间几何。抓取失败常被误归因到「骨干不行」，实际是缺了深度/位姿级联。

## 什么时候应该上端到端 / 从头训骨干

满足下列任意条件，再放弃冻结表征、转向端到端联合训练：

- 真机/仿真数据充足，且任务高度单一、追求性能上限
- 通用与专用预训练表征均与目标域差距过大、微调收益有限
- 视觉与控制需深度耦合（如隐式时序/触觉融合），冻结特征丢信息明显

## 常见误区

- **「骨干越强，策略越好」** —— 相机标定、时序对齐与控制策略本身常比换骨干更关键。
- **「冻结表征一定省事」** —— 预训练域与机器人第一视角差距大时，冻结反而拖累。
- **「通用表征能替代检测头」** —— 要结构化物体/位置时仍需检测/分割头，整图特征给不了显式边界框。
- **「先挑最强骨干再说」** —— 不先定输出形式，骨干选型就是无的放矢。
- **「2D 表征够抓取用」** —— 6DoF 抓取必须级联深度/位姿，纯 2D 特征给不了夹爪接近方向。

## 一句话记忆

> 先按 **策略要的输出形式** 分三类——要物体接 **检测/分割头**、要整图语义走 **预训练表征直喂**、要类别用 **分类骨干**；再按 **数据量与延迟** 在冻结/端到端、CNN/ViT 之间取舍；**2D 特征永远不给几何**，抓取在后面级联位姿。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNN | Convolutional Neural Network | 卷积神经网络，自带局部归纳偏置 |
| ViT | Vision Transformer | 图像分块 + Transformer 全局注意力骨干 |
| SSL | Self-Supervised Learning | 自监督预训练，无需人工标注 |
| FPN | Feature Pyramid Network | 多尺度特征金字塔 neck |
| R3M | Reusable Representations for Robotic Manipulation | 人类视频预训练的操作表征 |
| VC-1 | Visual Cortex 1 | CortexBench 通用具身视觉骨干 |
| DINOv2 | self-DIstillation with NO labels v2 | 大规模自监督 ViT 通用视觉特征 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| ROI | Region of Interest | 感兴趣区域，检测与位姿级联的中间产物 |
| 6DoF | 6 Degrees of Freedom | 6 自由度位姿（平移 + 旋转） |
| IK | Inverse Kinematics | 逆运动学，抓取执行前置 |
| INT8 | 8-bit Integer Quantization | 8 位整型量化，边缘部署常用 |

## 参考来源

- [ResNet 论文摘录（arXiv:1512.03385）](../../sources/papers/resnet_arxiv_1512_03385.md) — CNN 分类/检测骨干地基
- [YOLO v1 论文摘录（arXiv:1506.02640）](../../sources/papers/yolo_arxiv_1506_02640.md) — 单阶段检测头范式
- [经典视觉骨干与检测文献簇](../../sources/papers/vision_backbone_detection_classics.md) — 骨干谱系与对比

## 关联页面

- [视觉骨干（概念）](../concepts/vision-backbones.md) — 三类骨干的概念基础页
- [视觉表征作为策略输入（概念）](../concepts/visual-representation-for-policy.md) — 通用预训练表征三条路径
- [CNN vs ViT 视觉骨干对比](../comparisons/cnn-vs-vit-backbones.md) — 骨干族取舍（与接什么头正交）
- [目标检测模型选型 Query](object-detection-model-selection.md) — 检测头内部单阶段/两阶段细分选型
- [目标检测（方法）](../methods/object-detection.md) — 检测头方法谱系
- [VLA（方法）](../methods/vla.md) — 整图特征直喂的端到端多模态策略
- [Grasp Pose Estimation（方法）](../methods/grasp-pose-estimation.md) — 检测 ROI 之后的 6D 位姿级联
- [Visual Servoing（方法）](../methods/visual-servoing.md) — 抓取最后几厘米的对齐
