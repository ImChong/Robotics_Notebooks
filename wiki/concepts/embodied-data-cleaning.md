---
type: concept
tags: [data, machine-learning, manipulation, teleoperation, simulation]
status: complete
updated: 2026-05-29
related:
  - ../tasks/manipulation.md
  - ../queries/demo-data-collection-guide.md
  - ../methods/behavior-cloning.md
  - ../entities/humannet.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/humannet.md
summary: "具身数据清洗（Embodied Data Cleaning）是指对采集到的原始机器人操作演示数据进行质量过滤、时序对齐和误差补偿的过程，是保证模仿学习策略不学偏的关键步骤。"
---

# Embodied Data Cleaning (具身数据清洗)

**具身数据清洗**：在具身智能（Embodied AI）中，将人类示教或自动采集的原始“脏数据”转化为高质量、可用于训练的专家演示轨迹（Expert Trajectories）的过程。

## 为什么需要数据清洗？

具身数据与传统视觉数据集（如 ImageNet）不同，它包含了高频的时序信息和物理反馈。原始数据中常见的缺陷包括：
1. **网络延迟波动**：遥操作时，控制信号与图像帧之间可能存在 10-100ms 的非恒定延迟，导致动作与视觉状态不匹配（Misalignment）。
2. **人类无效动作**：操作者在开始前的迟疑、中途的犹豫或多余的抖动，如果不剔除，会增加策略学习的分布熵。
3. **传感器噪声与漂移**：低成本动捕设备产生的指尖位置漂移会导致“虚空抓取”现象。
4. **重定向误差**：人类手部结构与机器人灵巧手之间的拓扑差异，在映射（Retargeting）过程中会产生物理不可达的动作。

## 核心清洗流程

### 1. 时序对齐 (Temporal Alignment)
利用 NTP 同步或硬件触发信号，确保 RGB 图像、深度图、关节编码器、触觉信号在同一时间轴上。对于变延迟信号，常使用线性插值（Linear Interpolation）将其重新采样到固定的控制频率（如 50Hz）。

### 2. 异常轨迹过滤 (Outlier Detection)
- **物理校验**：剔除违反关节限位（Joint Limits）、最大角速度或产生瞬间巨大接触力的轨迹。
- **任务成败判定**：仅保留最终完成任务（如成功抓取并放置）的样本。
- **平滑度检查**：使用二阶导数检测加速度异常突变的点。

### 3. 重定向误差修复
在 retargeting 后运行一个小型的逆运动学（IK）微调，确保机器人的夹爪或指尖确实与物体表面发生了接触，修正“隔空移物”的错误。

### 4. 动作分段与剪辑
自动剔除轨迹开头和结尾的静止期（Idling），仅保留对任务有实质性贡献的交互片段，以提高数据的信息密度。

## 在大规模数据集中的应用

在 **Open X-Embodiment** 等超大规模数据集中，自动化的清洗脚本通常会结合基础模型（如用 VLM 判定任务是否成功）来实现。

### HumanNet 的大规模管线对照

[HumanNet（arXiv:2605.06747）](../entities/humannet.md) 虽非机器人示教数据，其 **采集 → 处理 → 标注** 三阶段管线与本页概念层高度同构：**去重归一化、内容/质量过滤、场景切分、clip 化** 对应时序对齐与异常过滤；**3D 手体姿态 + SLAM + 运动重定向**（论文给出 robot-ready 子集：重定向误差 <15 mm、有效帧覆盖率 >60%）对应重定向误差修复；**LLM 辅助字幕/运动描述** 对应任务成败与语义质量判定。HumanNet 证明 **人中心互联网视频** 在固定 VLA 设定下可与小规模真机数据竞争——前提是清洗管线把异构网页视频变成 **可审计、可预训练** 的资产，而非 raw dump。

## 关联页面
- [Manipulation 任务](../tasks/manipulation.md)
- [演示数据采集指南](../queries/demo-data-collection-guide.md)
- [Behavior Cloning (行为克隆)](../methods/behavior-cloning.md)

## 参考来源
- Padalkar, A., et al. (2023). *Open X-Embodiment: Robotic Learning at Scale*.
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md)
- [sources/papers/humannet.md](../../sources/papers/humannet.md) — HumanNet 百万小时清洗管线对照
