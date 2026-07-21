# 3D目标检测经典算法全盘点：单目、双目、激光雷达

> 来源归档（blog / 微信公众号）

- **标题：** 3D目标检测经典算法全盘点：单目、双目、激光雷达
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接（短链）：** https://mp.weixin.qq.com/s/1d7P4HDXmmZUZiVNx1HfXw
- **原始链接（长链）：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602190&idx=1&sn=a9e9a29449a395f8c08f54f4c78fed06
- **专栏专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（第 2 篇）
- **发表日期：** 2026-06-22
- **入库日期：** 2026-07-21
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；**短链直连成功**（长链此前 CAPTCHA / 搜狗未收录）
- **一句话说明：** 3D 检测按传感器分 **单目 / 双目 / LiDAR（含多模态）**；点云侧体素化→稀疏卷积→Pillars→点级两阶段→融合；**CenterPoint** 衔接下篇跟踪。

## 核心摘录（归纳，非全文）

### 传感器路线选型（文内立场）

| 路线 | 深度来源 | 成本/精度叙事 | 典型定位 |
|------|----------|---------------|----------|
| **单目** | 几何/先验/学习「猜」深度 | 最低成本；远距易漂 | 经济型智驾主视觉或辅感知 |
| **双目** | 视差 + 标定几何 | 深度优于单目；低于 LiDAR | 港口/园区等低速密集场景叙事 |
| **LiDAR / 融合** | 主动测距点云 | 最高精度与鲁棒；硬件贵 | 量产高端与基准榜单主力 |

> **收束：** 三条路线 **不互相取代**；选型看车型定位、传感器配置与场景，而非「谁更先进」。

### 单目 3D

| 算法 | 核心思想 | 代表意义 / 局限叙事 |
|------|----------|---------------------|
| **FCOS3D** | Anchor-Free；语义与深度解耦 + 深度不确定性 | 量产均衡主流叙事；>15 m / 逆光夜间易跌 |
| **SMOKE** | 几何中心点推 3D；极简分支 | 后视/环视泊车/低速配送；远距遮挡弱 |
| **MonoGRNet** | 透视与接地点等几何推理 | 少依赖海量标注；长距强遮挡仍不稳 |

### 双目 3D

| 算法 | 核心思想 | 代表意义 |
|------|----------|----------|
| **3DOP** | 在 3D 空间直接生成候选（非 2D→3D 抬升） | 双目提案奠基 |
| **Disp R-CNN** | **实例级**视差，避开全局纹理缺失区 | 小目标友好；港口/园区落地叙事 |

### 点云与多模态

点云核心问题：无序点 → **可计算结构化表示**，同时保留几何。文内四条线：**体素化 / 点直接处理 / 点–体素混合 / 多模态融合**。

| 算法 | 核心思想 | 代表意义 |
|------|----------|----------|
| **VoxelNet** | Voxel + VFE（微型 PointNet）→ RPN | 端到端体素奠基；稠密 3D 卷积贵 |
| **SECOND** | **稀疏 3D 卷积**只算非空体素 | VoxelNet 实用化；后续体素族底座 |
| **PointPillars** | 竖直 Pillars → 伪图 + 2D 卷积 | 文内「量产最广实时 3D」叙事（~62–105 Hz） |
| **PointRCNN** | 点级两阶段：前景分割提案 → 框精修 | 高精度；算力重、偏慢 |
| **AVOD** | LiDAR BEV/FV + RGB 多视图融合 | 早期融合标杆 |
| **Frustum PointNets** | 2D 框抬升视锥 → 锥内 PointNet | 「2D 驱动 3D」范式 |
| **LiDAR-RCNN** | 可插拔二阶段点级精修；显式用框尺寸 | 挂任意一阶段检测器提点 |
| **CenterPoint** | CenterNet 思想迁 3D；中心点+速度；贪心匹配跟踪 | 检测–跟踪一体；衔接下篇 MOT |

## 对 wiki 的映射

- [autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
- [object-detection](../../wiki/methods/object-detection.md)
- [object-detection-model-selection](../../wiki/queries/object-detection-model-selection.md)
- [wechat_shenlan_ai_ad_tracking_prediction](wechat_shenlan_ai_ad_tracking_prediction.md)（CenterPoint 跟踪侧）
- [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

## 可信度与使用边界

- 策展导读；KITTI/Waymo/nuScenes 数字与「量产最广」表述会过时，以论文与车规实测为准。
- 原始抓取正文见 [wechat_shenlan_ai_ad_3d_detection_2026-06-22.md](../raw/wechat_shenlan_ai_ad_3d_detection_2026-06-22.md)。
