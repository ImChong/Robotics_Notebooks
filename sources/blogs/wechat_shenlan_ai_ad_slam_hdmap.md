# 自动驾驶核心算法盘点｜SLAM与高精地图篇

> 来源归档（blog / 微信公众号）

- **标题：** 自动驾驶核心算法盘点｜SLAM与高精地图篇
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602862&idx=1&sn=9918db21b11a1d5fcec96482798bbff7
- **专栏专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（第 4 篇）
- **发表日期：** 2026-07-13
- **入库日期：** 2026-07-21
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai；直连 CAPTCHA，经搜狗微信中转取正文
- **一句话说明：** 定位/建图从 ORB-SLAM2 → LOAM → LIO-SAM 的传感器融合主线，叠加高精地图多层结构与 MapTR 在线向量化建图，收束为「重感知、轻地图 / 地图与感知联合建模」。

## 核心摘录（归纳，非全文）

### 为何需要 SLAM + HD Map

- GPS 失效（地下车库）与车道线被遮挡（雨雪）时，依赖「记忆与定位」。
- SLAM = 边走边定位 + 建图；HD Map = 几何 + 语义（车道/灯/限速）的「超视距传感器」。

### SLAM 三件套

| 算法 | 模态 | 核心机制 | 局限/演进 |
|------|------|----------|-----------|
| **ORB-SLAM2**（2017） | 视觉 | Tracking / Local Mapping / Loop Closing；ORB 特征；地图复用与重定位 | 光照/无纹理脆弱，量产需融合 |
| **LOAM**（CMU 2014） | 激光 | 高频里程计（边/面特征）+ 低频精细建图解耦 | 启发 LeGO-LOAM / A-LOAM |
| **LIO-SAM**（2020） | 激光–IMU（+GPS） | 因子图紧耦合；IMU 去畸变与撑场，激光纠漂，GPS 消全局误差 | 退化场景（笔直同质高速）主力解 |

### 高精地图

- **多层结构：** SD 导航 → 几何 → 语义 → 先验 → 实时层；支撑厘米级定位与「虚拟铁轨」。
- **MapTR（2022）：** 多相机 → Transformer → BEV 向量车道/边界/人行道；推动「重感知、轻地图」与众包更新。

### 文末参考文献注意

- 推送文末「参考资料」误粘贴了规控篇论文列表（Hybrid A* / Frenet / EM Planner）；**以正文论述的 ORB-SLAM2、LOAM、LIO-SAM、MapTR 为准**，勿按文末列表溯源。

## 对 wiki 的映射

- [autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
- [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [orb-slam3](../../wiki/entities/orb-slam3.md)（ORB-SLAM 族）、[lio-sam](../../wiki/entities/lio-sam.md)、[lego-loam](../../wiki/entities/lego-loam.md)、[fast-lio](../../wiki/entities/fast-lio.md)
- [lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

## 可信度与使用边界

- 策展导读；开源实现与评测以官方仓库 / 论文为准。
- 原始抓取正文见 [wechat_shenlan_ai_ad_slam_hdmap_2026-07-13.md](../raw/wechat_shenlan_ai_ad_slam_hdmap_2026-07-13.md)。
