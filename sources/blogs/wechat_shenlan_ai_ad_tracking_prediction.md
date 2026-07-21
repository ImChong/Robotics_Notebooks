# 自动驾驶核心算法盘点｜目标跟踪与轨迹预测篇

> 来源归档（blog / 微信公众号）

- **标题：** 自动驾驶核心算法盘点｜目标跟踪与轨迹预测篇
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247603090&idx=1&sn=50660dccdee1fe7f77438eb839203156
- **专栏专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（第 5 篇）
- **发表日期：** 2026-07-19
- **入库日期：** 2026-07-21
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai；直连 CAPTCHA，经搜狗微信中转取正文
- **一句话说明：** 跟踪侧 AB3DMOT → CenterPoint → EagerMOT；预测侧 Social LSTM/GAN → VectorNet → TNT；主题是从「当前帧检测」走向连续时空理解与意图驱动多模态未来。

## 核心摘录（归纳，非全文）

### 目标跟踪（Tracking-by-Detection）

| 算法 | 年份/出处 | 核心思想 | 代表意义 |
|------|-----------|----------|----------|
| **AB3DMOT** | IROS 2020 | 3D KF + 匈牙利 + 3D IoU；检测够好则跟踪可极简 | 在线 3D MOT 工程基线（KITTI ~207 FPS） |
| **CenterPoint** | CVPR 2021 | BEV 中心点热力图 + 速度；贪心最近点匹配，弃 3D IoU | 检测–跟踪一体；nuScenes/Waymo 标杆 |
| **EagerMOT** | ICRA 2021 | 先 3D LiDAR 匹配，再把未匹配轨迹投到 2D 相机匹配 | 远距稀疏点云时相机保 ID |

### 轨迹预测（多模态未来）

| 算法 | 年份/出处 | 核心思想 | 代表意义 |
|------|-----------|----------|----------|
| **Social LSTM** | CVPR 2016 | 每主体 LSTM + Social Pooling | 交互建模开端 |
| **Social GAN** | CVPR 2018 | GAN + Variety Loss 生成多条轨迹 | 单一回归 → 多模态生成 |
| **VectorNet** | CVPR 2020 | 地图/轨迹折线向量化 + 分层 GNN | 弃 BEV 栅格渲染；预测骨干标配 |
| **TNT** | CoRL 2020 | 终点分类 → 轨迹补全 → 打分 | 意图驱动；长时域可解释 |

### 收束判断

- 跟踪 = 短期记忆（离散框 → 连续实体）；预测 = 未来想象力（拓扑 + 交互 → 意图）。
- 端到端模糊模块边界，但 3D 极简滤波、点特征、多模态融合、向量地图、意图驱动仍是时空理解基石。

## 一手论文索引（文内）

1. Weng et al., AB3DMOT, IROS 2020
2. Yin et al., CenterPoint, CVPR 2021
3. Kim et al., EagerMOT, ICRA 2021
4. Alahi et al., Social LSTM, CVPR 2016
5. Gupta et al., Social GAN, CVPR 2018
6. Gao et al., VectorNet, CVPR 2020
7. Zhao et al., TNT, CoRL 2020

## 对 wiki 的映射

- [autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
- [object-detection](../../wiki/methods/object-detection.md)（跟踪上游检测）
- [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [kalman-filter](../../wiki/formalizations/kalman-filter.md)（AB3DMOT 滤波底座）

## 可信度与使用边界

- 策展导读；基准数字与 SOTA 声明随时间变化，以论文与排行榜为准。
- 原始抓取正文见 [wechat_shenlan_ai_ad_tracking_prediction_2026-07-19.md](../raw/wechat_shenlan_ai_ad_tracking_prediction_2026-07-19.md)。
