# Ego、UMI、遥操作、动捕：具身数据采集黑话，按四层读懂

> 来源归档（blog / 微信公众号）

- **标题：** Ego、UMI、遥操作、动捕：具身数据采集黑话，按四层读懂
- **类型：** blog（术语科普 / 采集方案阅读地图）
- **作者：** 具身智能前沿（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Eh2EWm9YSf0EgkksHIjVRQ
- **发表日期：** 2026-09-05
- **入库日期：** 2026-09-19
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_jushen_qianyan_embodied_data_collection_taxonomy_2026-09-05.md`](../raw/wechat_jushen_qianyan_embodied_data_collection_taxonomy_2026-09-05.md)
- **一句话说明：** 用视角 / 设备 / 教法 / 产物四层拆解 Ego、GoPro、UMI、遥操作、MoCap、episode 等采集黑话，并贯穿「动作标签从哪来」这一判读主线。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 总览概念页 | [embodied-data-collection-four-layers-taxonomy](../../wiki/concepts/embodied-data-collection-four-layers-taxonomy.md) |
| 交叉：遥操作 | [teleoperation](../../wiki/tasks/teleoperation.md) |
| 交叉：模仿学习 | [imitation-learning](../../wiki/methods/imitation-learning.md) |
| 交叉：IMU / SLAM | [imu-principles-algorithms-camera-sync](../../wiki/concepts/imu-principles-algorithms-camera-sync.md) |
| 交叉：重定向 | [motion-retargeting](../../wiki/concepts/motion-retargeting.md) |
| 交叉：HuMI | [paper-notebook-humanoid-manipulation-interface](../../wiki/entities/paper-notebook-humanoid-manipulation-interface.md) |
| 交叉：DROID | [droid-policy-learning](../../wiki/entities/droid-policy-learning.md) |
| 交叉：OXE | [open-x-embodiment](../../wiki/concepts/open-x-embodiment.md) |

## 核心摘录（MVP）

### 1) 四层阅读地图（非互斥分类）

- **视角**：Ego / Exo — 从谁的位置观察；不决定是否有机器人动作标签。
- **设备**：RGB / RGB-D / IMU / SLAM — 采到什么物理信号；GoPro 是品牌不是数据类型。
- **教法**：被动视频、UMI、遥操作、拖动示教、MoCap、数据手套 — 人怎样提供演示；决定动作标签形成路径。
- **产物**：图像、位姿、关节角、力触觉、episode / trajectory — 最终记录与组织单位。

### 2) 动作标签是贯穿问题

- 模仿学习常见设定：给定观测 → 预测示范动作；动作标签是训练目标，不是唯一正确解。
- 被动视频有「人在抓杯子」≠ 有「夹爪应移动多少」；后者需估计、对应或机器人数据。
- 遥操作指令 ≠ 实际执行位姿；UMI 需 SLAM + 轨迹处理 + 可行性筛选后才可训练。

### 3) 代表系统（原文重点案例）

- **UMI**（arXiv:2402.10329，RSS 2024）：手持 GoPro 夹爪 + 视觉惯性 SLAM → 6-DoF 位姿 + 视觉估开合；**已开源**（`real-stanford/universal_manipulation_interface`）。
- **FastUMI**（arXiv:2409.19499）：T265 跟踪 + GoPro 观测 + 多机器人适配件；>1 万条 / 22 任务。
- **HuMI**（arXiv:2602.06643）：UMI 夹爪 + 五 Vive Ultimate Tracker 全身 MoCap → 人形全身无机器人示范（本库已有实体页）。
- **遥操作谱系**：ALOHA / Mobile ALOHA、GELLO、DROID（Quest 2 + Franka）、SpaceMouse、外骨骼。
- **跨本体**：DROID（7.6 万条 / 350 h）、Open X-Embodiment（22 本体 / 100 万+ 轨迹）。

### 4) 读方案时的四个追问

1. 视角是什么？是否与部署相机一致？
2. 设备输出哪些模态？IMU/深度是否参与动作估计？
3. 教法如何形成动作标签？是否需要重定向 / IK / 延迟匹配？
4. 产物如何组织为 episode？小时数 vs 条数 vs 成功率各代表什么？
