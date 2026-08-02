# “同样是具身感知”，这六种空间表征到底差在哪里？

> 来源归档（blog / 微信公众号）

- **标题：** “同样是具身感知”，这六种空间表征到底差在哪里？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/lWvdz9cjuurS7ikBkZk0vQ
- **发表日期：** 2026-08-02
- **入库日期：** 2026-08-02
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [`sources/raw/wechat_shenlan_six_spatial_representations_2026-08-02/article.md`](../raw/wechat_shenlan_six_spatial_representations_2026-08-02/article.md)
- **一句话说明：** 把具身感知中常被并列混谈的 **2D 视觉 / 深度 / 点云 / 占据栅格（含 TSDF·ESDF）/ 语义地图 / 隐式地图** 拆成感知栈上的不同层级：各自回答不同问题；强调「不是 2D→3D 单向升级」，语义地图（存什么）与隐式地图（怎么表示）正交。

## 核心摘录（归纳，非全文）

### 总判断：层级混淆，而非术语偏差

- 同一房间可同时维护多种「地图」；真实系统在识别、定位、规划、执行节点做表征转换与融合。
- 常见混谈：深度≈点云、TSDF≈占据栅格、所有神经隐式≈NeRF——实质是**感知栈层级混淆**。
- 文内口诀：**在正确层级，选择正确表征**。

### 六种表征各自回答什么

| 表征 | 核心问题 | 主要服务 | 根本边界 |
|------|----------|----------|----------|
| **2D 视觉** | 画面里有什么 | 分类/检测/分割/关键点/可供性；视觉伺服；VLM 入口 | 单图通常无唯一三维尺度；「检测到」≠ 6D 位姿 |
| **深度** | 沿视线表面多远 | 尺度估计、RGB-D SLAM、度量对齐 | 视角绑定的第一层表面；相对深度≠米制度量深度 |
| **点云** | 观测到的表面在哪里 | ICP、法向、抓取位姿（PointNet++ / GraspNet 等） | 「有点」≠ 自由空间已确认；缺射线穿过语义 |
| **占据栅格 / 距离场** | 哪里空闲、离障碍多远 | 碰撞检查、导航、轨迹优化 | 同为体素结构但物理量不同：占据概率 / TSDF / ESDF |
| **语义地图** | 这里是什么、与任务何关 | 目标导航、语言指令落地（含 VLMaps 开放词汇） | 依赖底层几何与长期一致性；≠「分割图上色」 |
| **隐式地图** | 任意坐标处几何/外观是什么 | 稠密重建、连续空间查询（iMAP / NICE-SLAM 等） | ≠ NeRF  alone；在线更新、遗忘、碰撞边界提取仍难 |

### 体素三兄弟（文内硬区分）

| 表示 | 体素存什么 | 回答 | 典型用途 |
|------|------------|------|----------|
| 占据概率栅格（如 OctoMap） | 占据概率 + 空闲/占据/未知 | 这里能不能走 | 碰撞检查、探索 |
| **TSDF** | 到观测表面的截断符号距离 | 表面零交叉在哪 | 多帧深度融合、网格提取 |
| **ESDF** | 到最近障碍的欧氏符号距离 | 离障碍还有多远 | 轨迹优化、安全余量 |

### 两个最易误解点

1. **点云之后不必强制进占据栅格**——可直连抓取/三维检测，也可分别进距离场或语义地图。
2. **语义地图与隐式地图不是前后替代**——前者描述地图保存「什么内容」，后者描述地图「如何表示」。

### 工程收束

前端侧重实时反应与丰富语义；后端侧重空间一致性与任务安全。典型链路示例：深度→点云→占据栅格做规划，并行语义地图关联指令，隐式场做高保真重建。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md) | **主沉淀页**：六层问题边界、体素三兄弟、选型口诀 |
| [2D→3D 语义提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md) | 2D/深度→语义几何的信息损失根因 |
| [机器人视觉感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md) | 传感→2D→3D→策略消费的工程选型链 |
| [导航·SLAM·自动驾驶开源栈](../../wiki/overview/navigation-slam-autonomy-stack.md) | 占据/ESDF/Nav2 工程落点 |
| [Isaac ROS nvblox](../../wiki/entities/isaac-ros-nvblox.md) | GPU TSDF/ESDF 重建代表 |
| [GO2 三维语义建图 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) | 语义地图落地案例 |
| [三维坐标变换](../../wiki/formalizations/3d-coordinate-transforms-vision-robotics.md) | 像素↔相机↔世界几何底座 |
