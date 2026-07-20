# Neuromorphic Computing Chip with Spatiotemporal Elasticity for Multi-Intelligent-Tasking Robots（Science Robotics, 2022）

> 来源归档（ingest）

- **标题：** Neuromorphic computing chip with spatiotemporal elasticity for multi-intelligent-tasking robots
- **类型：** paper / neuromorphic computing / hybrid ANN-SNN / robot chip / multi-task
- **期刊：** Science Robotics, 2022
- **DOI：** <https://doi.org/10.1126/scirobotics.abk2948>
- **arXiv：** 暂无公开 arXiv 预印本
- **项目页：** 无独立开源项目页；配套演示视频随 Science Robotics 发布
- **通讯作者：** Shi Luping（施路平），清华大学类脑计算研究中心（Center for Brain-Inspired Computing Research，CBICR）
- **机构：** 清华大学类脑计算研究中心（Tsinghua CBICR）
- **平台：** TianjicX 神经形态芯片；轮式机器人平台；多任务演示（跟踪 + 避障 + 语音识别）
- **代码与数据：** 无公开代码仓库；截至入库日（2026-07-20）无 GitHub；属于确认未开源
- **入库日期：** 2026-07-20
- **一句话说明：** 清华施路平团队发布 **TianjicX** 神经形态芯片，在单芯片上融合 **ANN 机器学习算子**与 **SNN 脉冲神经网络**，以时空弹性计算架构使轮式机器人在**低功耗**条件下同时完成**目标跟踪、障碍物避让、语音识别**三类任务。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.abk2948](https://doi.org/10.1126/scirobotics.abk2948) | Science Robotics 2022 论文主页 |
| 团队前作 | Tianjic（Nature 2019）— 首代混合 ANN/SNN 芯片 | TianjicX 是其升级版；增加时空弹性 |
| NeuroGPR（后续工作） | [`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](../../wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md) | 同团队将 ANN+SNN 扩展至多模态位置识别 |
| 主 wiki 页 | [`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](../../wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md) | 知识沉淀目标页 |
| Locomotion 任务 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 轮式机器人导航属 locomotion 子领域 |

## 摘要级要点

- **问题：** 机器人执行多任务时，ANN（深度学习）效果好但功耗高；SNN（脉冲神经网络）低功耗但表达力受限；单一架构无法同时覆盖实时感知、时序推理与语音理解。
- **TianjicX 时空弹性架构：** 芯片内可动态重构计算核的**空间弹性**（灵活划分 ANN 核与 SNN 核）与**时间弹性**（调整各任务计算时隙比例），在单芯片内混合部署多种神经网络模型。
- **三任务机器人演示：**
  1. **目标跟踪**（Object Tracking）：基于 ANN CNN 的视觉目标跟踪，低延迟响应。
  2. **障碍物避让**（Obstacle Avoidance）：SNN 处理距离传感器的时序脉冲，低功耗实时决策。
  3. **语音识别**（Speech Recognition）：ANN 声学模型在同芯片上并行执行指令理解。
- **低功耗优势：** 相比全 ANN 方案，TianjicX 混合架构显著降低总体功耗，适合边缘移动机器人。
- **Tianjic 系谱：** 2019 年 Nature 上发表首代 Tianjic 芯片（ANN+SNN 首次同芯片），TianjicX 进一步增加时空弹性调度，扩展到多任务机器人应用。
- **局限（论文 Discussion）：** 演示平台为轮式；SNN 编程工具链成熟度低于主流 GPU/TPU；ANN 精度在部分任务上仍低于专用加速器。

## 核心摘录（面向 wiki 编译）

### 1) 时空弹性架构核心设计

| 维度 | ANN 计算核 | SNN 计算核 | 弹性调度 |
|------|-----------|-----------|----------|
| 适用模型 | 卷积 / 全连接深度网络 | 脉冲时序网络（LIF 神经元） | 运行时重配比例 |
| 优势 | 表征精度高 | 时序编码低功耗 | 任务动态分配 |
| 代表任务 | 目标跟踪、语音识别 | 障碍物时序避让 | 三任务并发 |

- **对 wiki 的映射：** [`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](../../wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md)

### 2) 机器人集成演示

- **平台：** 小型轮式移动机器人，搭载 TianjicX 芯片 + 摄像头 + 距离传感器 + 麦克风
- **三任务并发：** 视觉跟踪 + 避障 + 语音识别在同一芯片实时运行
- **功耗对比：** 混合 ANN+SNN 方案相比纯 ANN GPU 方案功耗大幅降低（具体数值见论文 Table 2）
- **对 wiki 的映射：** [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)（轮式机器人自主导航）

### 3) 与 Tianjic 系谱关系

- **Tianjic（Nature 2019）：** 首代混合 ANN+SNN 芯片；无人自行车平衡 + 识别任务演示
- **TianjicX（本文，Science Robotics 2022）：** 增加时空弹性，多任务轮式机器人
- **NeuroGPR（Science Robotics 2023）：** 将混合框架扩展至多模态位置识别（LiDAR + 视觉 + ANN+SNN）
- **对 wiki 的映射：** [`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](../../wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md)

### 4) 开源状态

- 无 GitHub；截至入库日确认未开源；wiki「局限与风险」一句标注。
- **对 wiki 的映射：** [`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](../../wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md) § 局限

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](../../wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**
- 谱系：**[`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](../../wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md)**
