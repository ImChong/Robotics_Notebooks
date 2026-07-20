# MediaPipe（Google 官方仓库）

> 来源归档

- **标题：** MediaPipe
- **类型：** repo
- **机构：** Google
- **链接：** <https://github.com/google/mediapipe>
- **开发者文档：** <https://developers.google.com/mediapipe>
- **许可证：** Apache License 2.0（见仓内 `LICENSE`）
- **入库日期：** 2026-07-20
- **一句话说明：** Google 开源的跨平台端侧 ML 框架与 **Solutions / Tasks** 套件：在移动、Web、桌面与边缘设备上提供可定制的视觉、文本与音频流水线；机器人栈中常用于 **手部 21 关键点、全身姿态、人脸** 等低成本感知输入。
- **沉淀到 wiki：** [MediaPipe](../../wiki/entities/mediapipe.md)
- **交叉归档：** [MediaPipe 开发者文档站](../sites/mediapipe-developers-google.md)

---

## 仓库要点（README ingest 快照，2026-07-20）

| 项 | 说明 |
|----|------|
| **定位** | On-device machine learning for everyone — 可定制并部署到 Android、iOS、Web、桌面、边缘与 IoT |
| **MediaPipe Solutions** | 预置任务库：**Vision**（物体检测、手势、姿态、分割等）、**Text**、**Audio**；含预训练模型与跨平台 **Tasks API** |
| **MediaPipe Framework** | 底层图计算框架（Packets / Graphs / Calculators），用于自建高效端侧流水线 |
| **工具链** | **Model Maker**（用自有数据微调）、**Studio**（浏览器可视化与评测） |
| **文档迁移** | 自 2023-04-03 起，主文档迁至 `developers.google.com/mediapipe`；GitHub README 为转发页 |
| **Legacy Solutions** | 部分旧版 Solution 已于 2023-03-01 停止支持；代码与预编译包仍 **as-is** 保留在仓库 |

## 机器人相关常用能力（Tasks / Legacy）

| 能力 | 典型输出 | 机器人用途 |
|------|----------|------------|
| **Hand Landmarker** | 21 个 3D/2D 手部关键点 | 灵巧手遥操作、重定向 warm-start（如 [TopoRetarget](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)） |
| **Pose Landmarker** | 33 点全身骨架 | 全身遥操作、动作模仿上游 |
| **Face Landmarker** | 面部网格 / 关键点 | 表情驱动、人脸相关数据采集 |
| **Holistic（Legacy）** | 手 + 姿态 + 脸联合 | 旧项目仍常见；新项目优先 Tasks 分拆 API |

## 开源状态

- **已开源**：完整框架源码、Solutions 实现、预训练模型与多平台示例均在公开仓库与开发者站可获取。
- **注意**：2023 年后新功能以 **MediaPipe Tasks** 与 `developers.google.com` 文档为主；集成旧版 Graph / Solution 时需核对是否已列入 Legacy 列表。

## 对 wiki 的映射

- 主实体页：**`wiki/entities/mediapipe.md`**
- 遥操作实践：**`wiki/queries/dexterous-data-collection-guide.md`**、**`wiki/entities/midas-hand.md`**
- 重定向方法：**`wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md`**
