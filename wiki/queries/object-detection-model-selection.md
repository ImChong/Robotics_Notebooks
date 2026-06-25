---
type: query
tags: [object-detection, perception, computer-vision, real-time, yolo, faster-rcnn, robotics, deployment]
status: complete
updated: 2026-06-25
summary: "目标检测模型选型 Query：从「机载实时 vs 服务器侧高精度」「单阶段 vs 两阶段 / 实时 DETR」「2D 框够不够 vs 要不要级联位姿」三轴出发，给出机器人感知栈里检测器的选型逻辑、部署陷阱与组合 pipeline。"
related:
  - ../methods/object-detection.md
  - ../concepts/vision-backbones.md
  - perception-backbone-selection.md
  - ../entities/paper-yolo-unified-realtime-detection.md
  - ../entities/paper-resnet-deep-residual-learning.md
  - ../entities/rf-detr.md
  - ../tasks/manipulation.md
  - ../tasks/humanoid-soccer.md
  - ../methods/visual-servoing.md
  - ../methods/grasp-pose-estimation.md
sources:
  - ../../sources/papers/yolo_arxiv_1506_02640.md
  - ../../sources/papers/resnet_arxiv_1512_03385.md
  - ../../sources/papers/vision_backbone_detection_classics.md
  - ../../sources/papers/rf_detr_arxiv_2511_09554.md
---

> **Query 产物**：本页由以下问题触发：「机器人感知栈里到底该选单阶段（YOLO 系）还是两阶段（Faster R-CNN 系）检测器？机载实时和服务器侧高精度的选型逻辑有什么不同？2D 框够用吗？」
> 综合来源：[Object Detection（方法）](../methods/object-detection.md)、[视觉骨干（概念）](../concepts/vision-backbones.md)、[YOLO v1（论文实体）](../entities/paper-yolo-unified-realtime-detection.md)、[ResNet（论文实体）](../entities/paper-resnet-deep-residual-learning.md)、[Manipulation（任务）](../tasks/manipulation.md)、[Humanoid Soccer（任务）](../tasks/humanoid-soccer.md)

# Query：目标检测模型选型（机载实时 vs 服务器侧 / 单阶段 vs 两阶段 / 2D 框 vs 级联位姿）

## TL;DR 决策树

```text
检测要在哪里跑、延迟预算多少？
├── 机载 / 边缘（Jetson 级算力，要 >10–30 FPS 闭环）
│   ├── 目标类别固定、场景受控（球 / 障碍 / 人）
│   │   └→ 单阶段（YOLO 系）或 无 NMS 实时 DETR（[RF-DETR](../entities/rf-detr.md)）+ TensorRT/FP16
│   └── 类别开放 / 需语言指令
│       └→ 轻量开放词汇检测（OWL-ViT / Grounding-DINO 蒸馏版）兜底
└── 服务器侧 / 离线（算力充足，精度优先）
    ├── 小目标密集 / 高定位精度要求（抓取候选框）
    │   └→ 两阶段（Faster R-CNN + ResNet-FPN）或 RetinaNet
    └── 已知物体、要完整 6DoF
        └→ 检测器只做 ROI → 级联 6D 位姿网络（见 Grasp Pose Estimation）
```

## 快速结论

- **第一刀永远是「延迟预算」而不是「mAP 榜单」**：机载实时闭环先锁 **单阶段 YOLO 或 RF-DETR 等无 NMS 实时检测器** + TensorRT/量化，把架构版本之争放到后面。
- **两阶段不是「过时」而是「换赛道」**：Faster R-CNN/RetinaNet 在小目标、密集、定位精度敏感的服务器侧场景仍是更稳的起点。
- **2D 框只是感知的起点，不是终点**：抓取/操作几乎都要在检测器后面级联 **深度/点云/6D 位姿**（见 [Grasp Pose Estimation](../methods/grasp-pose-estimation.md)）。
- **输入分辨率与 NMS 阈值常比换最新版本更关键**：仿真纹理 vs 真机的数据分布差异，往往比 YOLOv5→v8 的架构差距更影响真机成功率。
- **误差结构是可以组合利用的**：单阶段背景误报少、两阶段定位准，必要时可用一个给另一个重打分（YOLO v1 对 Fast R-CNN 重打分 +3.2 mAP）。

## 三轴选型对比表

### 轴 1：部署位置（机载实时 vs 服务器侧高精度）

| 场景 | 主线方案 | 关键约束 | 失败模式 | 兜底 |
|------|---------|---------|---------|------|
| 机载边缘（Jetson / NPU） | 单阶段 YOLO 或 [RF-DETR](../entities/rf-detr.md) + TensorRT/FP16 | 延迟、显存、功耗 | 量化掉点、小目标漏检 | 降分辨率 + 提高输入 FPS；ROI 跟踪减少全图推理 |
| 移动机器人导航 | 单阶段中等输入分辨率 | 运动模糊、视角变化 | 远距离小目标、动态遮挡 | 多帧时序聚合 + 跟踪滤波 |
| 服务器侧抓取感知 | 两阶段 / RetinaNet | 精度优先、延迟可放宽 | 透明/反光件深度缺失 | 级联深度补全 + 6D 位姿网络 |
| 离线标注 / 数据闭环 | 两阶段高分辨率集成 | 召回与定位精度 | 标注分布偏置 | 半自动标注 + 人工复核 |

### 轴 2：架构范式（单阶段 vs 两阶段）

| 范式 | 代表 | 优势 | 风险 | 何时优先 |
|------|------|------|------|---------|
| 单阶段密集回归 | [YOLO v1](../entities/paper-yolo-unified-realtime-detection.md)、SSD、RetinaNet | 端到端、全图上下文、快 | 小目标/密集场景定位错误偏高 | 机载实时、类别固定、闭环感知 |
| 端到端 DETR（无 NMS） | [RF-DETR](../entities/rf-detr.md)、RT-DETR | **ViT 域迁移**、确定性延迟、检测/分割统一 API | closed-vocab 需微调；XL 权重许可受限 | 垂直域 fine-tune、要与 YOLO 比 RF100-VL 类 benchmark |
| 两阶段提议+分类 | Faster R-CNN（RPN + RoI） | 定位精度高、小目标更稳 | 延迟大、工程链路长 | 服务器侧、高精度、小目标 |
| 单阶段 + Focal loss | RetinaNet | 单阶段逼近两阶段精度 | 仍需调难易样本平衡 | 想兼顾速度与精度时的折中 |
| 骨干升级（与范式正交） | ResNet-FPN（[ResNet](../entities/paper-resnet-deep-residual-learning.md)） | 多尺度特征、精度地基 | 算力随深度上升 | 任何范式都先确认骨干与 FPN 配置 |

### 轴 3：输出形式（2D 框够用 vs 需级联位姿）

| 任务输出诉求 | 检测器角色 | 下游级联 | 典型场景 |
|------|---------|---------|---------|
| 只要「有什么 + 大致在哪」 | 终端输出 2D 框 | 无 | 计数、避障触发、人/球检测 |
| 要稳定实例 + 跨帧 id | 检测 + 跟踪 | SORT/ByteTrack 类跟踪 | 移动机器人、足球人形 |
| 要完整 6DoF 抓取 | 仅做 ROI / 候选 | 深度/点云 → 6D 位姿或检测式 grasp | 桌面抓取、bin picking |
| 要逐像素掩码 | 检测 → 实例分割头 | Mask 头 / SAM 接力 | 顺应操作、可供性分析 |

## 推荐组合 pipeline

按工程化难度递增的四种真机出镜组合：

1. **机载固定类别（球 / 障碍 / 人）**
   - 单阶段（YOLO 系）→ TensorRT/INT8 → NMS → 跟踪滤波 → 控制层
   - 见 [Humanoid Soccer](../tasks/humanoid-soccer.md)：>30 FPS 是硬约束，宁可降分辨率也别上两阶段
2. **服务器侧高精度检测**
   - 两阶段（Faster R-CNN + ResNet-FPN）→ 高分辨率多尺度 → 显式置信度筛选
   - 失败兜底：小目标专用高分辨率切图（tiling）推理
3. **检测 → 抓取候选**
   - 单/两阶段检测出 ROI → 深度/点云裁剪 → [Grasp Pose Estimation](../methods/grasp-pose-estimation.md) → 显式碰撞检查 → IK
   - 最后几厘米切 [Visual Servoing](../methods/visual-servoing.md) / 触觉对齐
4. **开放词汇 / 语言指令检测**
   - 开放词汇检测（OWL-ViT / Grounding-DINO）→ 区域抓取 / 任务分解
   - 边缘部署时用蒸馏版或先检索缩小词表，再交给轻量检测头

## 关键工程经验

### 1. mAP 高 ≠ 机器人能用

离线 mAP 是数据分布内的匹配指标，真机部署还受 **相机标定、推理延迟、类别开放集、遮挡、运动模糊** 共同影响。真机调优优先盯：闭环端到端延迟、目标距离-召回曲线、误报触发的下游代价（误避障 / 误抓）。

### 2. 量化掉点要在选型阶段就预判

机载部署几乎一定要 INT8。**量化后小目标与低对比度目标最先掉**。选型时就该在量化后的精度上比较，而不是 FP32 榜单；必要时对关键层保留 FP16 混合精度。

### 3. 数据分布比架构版本更敏感

仿真纹理训练、真机部署是最常见的翻车点。换最新 YOLO 往往不如：补真机域数据、调输入分辨率、调 NMS/置信度阈值、做针对性数据增强（运动模糊、光照、纹理随机化）。

### 4. 2D 框不提供 6DoF，别让检测器背锅

抓取失败常被归因到「检测不准」，实际是缺了位姿级联。纯 2D 框无法给出夹爪接近方向与深度；操作任务必须把检测器当 ROI 生成器，后面接深度/点云/位姿网络。

## 什么时候应该上两阶段 / 服务器侧

满足下列任意条件，再放弃机载单阶段、转向两阶段或服务器侧高精度：

- 小目标密集且定位精度直接决定下游（抓取框、装配对齐）
- 延迟预算宽松（离线标注、非闭环感知、可上云）
- 需要高召回的安全关键检测（漏检代价远大于延迟代价）
- 类别细粒度区分难，单阶段分类头容量不够

## 常见误区

- **「YOLO 最新版一定更好」** —— 数据分布、输入分辨率、NMS 阈值通常比版本号更决定真机表现。
- **「两阶段过时了」** —— 在小目标 / 高定位精度 / 服务器侧场景，两阶段仍是更稳的起点。
- **「检测器选好就完事」** —— 抓取/操作的瓶颈往往在检测之后的位姿与碰撞检查环节。
- **「只看 mAP 不做量化与真机试验」** —— 必须在量化后精度 + 真机延迟上复核，离线榜单会误导选型。
- **「开放词汇检测可以直接上边缘」** —— 大模型检测器延迟高，边缘部署需蒸馏 / 缩词表，否则闭环帧率撑不住。

## 一句话记忆

> 先按 **延迟预算** 分机载/服务器：机载走 **单阶段 + 量化**，服务器侧走 **两阶段 / RetinaNet**；但 **2D 框只是起点**，抓取永远要在后面级联位姿与碰撞检查。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| mAP | mean Average Precision | 检测精度标准指标 |
| YOLO | You Only Look Once | 单阶段一次回归实时检测范式 |
| R-CNN | Regions with CNN features | 区域 CNN 两阶段检测族 |
| RPN | Region Proposal Network | Faster R-CNN 的可学习提议子网 |
| FPN | Feature Pyramid Network | 多尺度特征金字塔 |
| NMS | Non-Maximum Suppression | 检测框去重后处理 |
| ROI | Region of Interest | 感兴趣区域，两阶段与级联位姿的中间产物 |
| FPS | Frames Per Second | 帧率，机载实时闭环硬约束 |
| INT8 | 8-bit Integer Quantization | 8 位整型量化，边缘部署常用 |
| 6DoF | 6 Degrees of Freedom | 6 自由度位姿（平移 + 旋转） |
| IK | Inverse Kinematics | 逆运动学，抓取执行前置 |

## 参考来源

- [YOLO v1 论文摘录（arXiv:1506.02640）](../../sources/papers/yolo_arxiv_1506_02640.md) — 单阶段实时检测范式与误差画像
- [ResNet 论文摘录（arXiv:1512.03385）](../../sources/papers/resnet_arxiv_1512_03385.md) — 检测骨干与 FPN 的精度地基
- [经典视觉骨干与检测文献簇](../../sources/papers/vision_backbone_detection_classics.md) — 两阶段/单阶段谱系与对比
- [RF-DETR 论文摘录（arXiv:2511.09554）](../../sources/papers/rf_detr_arxiv_2511_09554.md) — 实时 DETR 与域迁移 benchmark

## 关联页面

- [感知骨干/表征选型 Query](perception-backbone-selection.md) — 上一层的「分类骨干 / 检测头 / 通用预训练表征」三类总选型
- [目标检测（方法）](../methods/object-detection.md) — 本 Query 的方法谱系基础页（两阶段 vs 单阶段技术路线）
- [视觉骨干（概念）](../concepts/vision-backbones.md) — 检测器骨干与多尺度特征的上游
- [YOLO v1（论文实体）](../entities/paper-yolo-unified-realtime-detection.md) — 单阶段回归检测开山工作
- [RF-DETR（实体）](../entities/rf-detr.md) — 无 NMS 实时 DETR 与 vertical-domain fine-tune
- [ResNet（论文实体）](../entities/paper-resnet-deep-residual-learning.md) — ResNet-FPN 骨干代表
- [Manipulation（任务）](../tasks/manipulation.md) — 检测 → 抓取候选的下游任务
- [Humanoid Soccer（任务）](../tasks/humanoid-soccer.md) — 机载实时检测的典型约束场景
- [Grasp Pose Estimation（方法）](../methods/grasp-pose-estimation.md) — 检测 ROI 之后的 6D 抓取位姿级联
- [Visual Servoing（方法）](../methods/visual-servoing.md) — 抓取最后几厘米的亚毫米级对齐
