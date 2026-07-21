# 自动驾驶感知算法盘点｜目标检测篇（一）

> 来源归档（blog / 微信公众号）

- **标题：** 自动驾驶感知算法盘点｜目标检测篇（一）
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接（短链）：** https://mp.weixin.qq.com/s/7Mm5OwVKgoyT4Zpr45E34A
- **原始链接（长链）：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247601965&idx=1&sn=bd482fb32d1e40b17bcdc8dfedf7ea04
- **专栏专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（第 1 篇）
- **发表日期：** 2026-06-14
- **入库日期：** 2026-07-21
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；**短链直连成功**（长链此前 CAPTCHA / 搜狗未收录）
- **一句话说明：** 车载 RGB **2D 检测**四大族：两阶段 Anchor → 单阶段 CNN Anchor（SSD/RetinaNet/EfficientDet/YOLO v3–v9）→ Anchor-Free（CenterNet/CornerNet）→ Transformer（ViT 族骨干 + DETR/Deformable DETR）。

## 核心摘录（归纳，非全文）

### 四族划分（车载落地视角）

| 族 | 共性 | 车载取舍 |
|----|------|----------|
| **两阶段 Anchor-Based** | 候选区 → 精修分类回归；误检低、遮挡稳 | 链路长、边缘帧率吃紧 |
| **单阶段 CNN Anchor** | 整图一次前向；帧率高、易部署 | 小目标与正负样本失衡是通用短板 |
| **单阶段 Anchor-Free** | 去锚框超参；中心点 / 角点两派 | 形变目标更友好；角点匹配易错 |
| **Transformer 骨干/检测** | 全局注意力；ViT 骨干 vs DETR 端到端 | 遮挡/远距强；算力与收敛成本高 |

### 两阶段：R-CNN → Fast → Faster

| 算法 | 核心思想 | 代表意义 |
|------|----------|----------|
| **R-CNN** | Selective Search ~2000 框 → 逐框 CNN → SVM | 卷积检测开山；不可实时 |
| **Fast R-CNN** | 整图一次卷积 + RoI 池化；联合训练 | 共享特征；仍外挂提议 |
| **Faster R-CNN** | **RPN** 自生成 Anchor 候选 | 全深度两阶段基线；高精度感知 |

### 单阶段 CNN Anchor：SSD → RetinaNet → EfficientDet → YOLO

| 算法 | 核心思想 | 代表意义 |
|------|----------|----------|
| **SSD** | 多尺度特征层 + 多尺寸 Anchor | 单阶段多尺度奠基 |
| **RetinaNet** | FPN + **Focal Loss** | 单阶段精度追平两阶段 |
| **EfficientDet** | 复合缩放 + **BiFPN** | 轻量多尺度；恶劣天气融合叙事 |
| **YOLOv3** | Darknet-53 + 三尺度 FPN | 速度–精度均衡工程基线 |
| **YOLOv4** | Mosaic / CIoU 等工程 trick | 逆光夜间鲁棒性叙事 |
| **YOLOv5** | C3 / 自适应锚与缩放 | 低算力量产首选叙事 |
| **YOLOv7** | ELAN + 辅助头 | 高速实时标杆叙事 |
| **YOLOv8** | 解耦头；检测+分割 | 拥堵轮廓/后处理规划 |
| **YOLOv9** | GELAN / 信息蒸馏 | 文内作「2024 卷积 YOLO 收官」；v10+ 未评 |

### Anchor-Free 与 Transformer

| 算法 | 核心思想 | 代表意义 |
|------|----------|----------|
| **CenterNet** | 中心点 + 宽高；可去 NMS | 中等尺寸车/人；低延迟 |
| **CornerNet** | 对角点配对成框 | 异形障碍更强；匹配易错 |
| **ViT / BEiT / DeiT** | 纯 ViT / 掩码预训练 / 蒸馏轻量 | 骨干侧全局语义 |
| **Swin / PVT** | 窗口分层 / 金字塔 Transformer | 工业骨干首选 vs 更轻量折中 |
| **DETR** | Encoder–Decoder 集合预测；无 Anchor/NMS | 重叠车流好；收敛慢、小目标弱 |
| **Deformable DETR** | 稀疏可变形注意力 | 文内「车规 Transformer 检测基线」叙事 |

### 收束判断

- 量产叙事轴：**YOLO / Faster R-CNN / DETR** 三系覆盖近十年辅助驾驶 2D 感知。
- L2 低成本仍重 2D；挑战是 **精度–实时、天气/隧道域偏移、长尾障碍、跨场景标注成本**。
- 下接专辑第 2 篇 3D 检测。

## 对 wiki 的映射

- [autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
- [object-detection](../../wiki/methods/object-detection.md)
- [object-detection-model-selection](../../wiki/queries/object-detection-model-selection.md)
- [vision-backbones](../../wiki/concepts/vision-backbones.md)

## 可信度与使用边界

- 策展导读；FPS/SOTA/「量产首选」表述随时间变化，以论文与车规实测为准。
- 原始抓取正文见 [wechat_shenlan_ai_ad_2d_detection_2026-06-14.md](../raw/wechat_shenlan_ai_ad_2d_detection_2026-06-14.md)。
