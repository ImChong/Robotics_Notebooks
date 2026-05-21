# NVIDIA：SO-101 机械臂 Sim-to-Real 动手课（Isaac + GR00T + LeRobot）

> 来源归档

- **标题：** Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac
- **类型：** course（厂商官方动手教程）
- **来源：** NVIDIA Physical AI Learning
- **链接：** https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- **上级门户：** https://docs.nvidia.com/learning/physical-ai/
- **入库日期：** 2026-05-21
- **一句话说明：** 以 SO-101 操作臂完成离心管瓶非结构化 pick-and-place：仿真遥操作采集 → GR00T N1.6 后训练 VLA → Isaac Lab 评测 → 真机部署，并系统演练 DR、Co-training、Cosmos 与 SAGE+GapONet 四类 sim2real 策略。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-so101-sim2real-lab-workflow.md`](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md)

---

## 课程结构（章节索引）

| 章节 | 主题 |
|------|------|
| 01 Overview | Physical AI 定义、离心管瓶任务动机、仿真价值 |
| 02 How to take this course | 环境与容器说明 |
| 03 Sim-to-real | 理论：四类 gap（感知/执行/物理/建模）、分布偏移与复合误差 |
| 04 LeRobot | Hugging Face 机器人数据与社区生态 |
| 05 Building workspace | 物理工作台搭建 |
| 06 Get the code | 代码与 Docker 环境 |
| 07–08 Calibrating / Operating SO-101 | 真机标定与操作 |
| 09 Strategy 1 | **域随机化（DR）** + 仿真遥操作采集演示 |
| 10 GR00T | **VLA**：Isaac GR00T / N1.6、预训练→后训练阶段 |
| 11–12 Sim / Real evaluation | 仿真与真机策略评测 |
| 13 Strategy 2 | **Co-training**：少量真机演示（约 5 ep）+ 大量仿真演示（约 70–100 ep）混合训练 |
| 14 Strategy 3 | **Cosmos** 世界模型增广：超越显式 DR 的视觉/场景多样性 |
| 15 Strategy 4 | **SAGE + GapONet**：执行器层 gap 度量与补偿建模 |
| 16 Conclusion | 总结 |

官方标注时长约 **6–10 小时**；GR00T 后训练需数小时 GPU，课程提供预训练策略以便聚焦 workflow 与策略对比。

---

## 核心摘录

### 1) 任务与学习目标

- **任务：** 桌面上散落的离心管瓶（centrifuge vials）非结构化抓取并放入 rack；策略主要依赖 **2D 相机**；抓取后腕部相机会遮挡，策略需能在缺失视觉下继续完成放置。
- **平台：** SO-101 为教学向低成本操作臂（非产线机器人）；重点在 **可复用 workflow**。
- **学习目标（官方）：** 标定 SO-101；遥操作 + DR 采集；用 GR00T 训练 VLA；仿真评测；真机部署并观察 gap；应用四种 sim2real 策略。

### 2) Sim-to-real 理论模块（03）

- **定义：** 在仿真中训练策略并部署到真机；高仿真成功率 ≠ 真机可用。
- **Gap 四类：** Sensing（噪声/光照/深度）、Actuation（摩擦/回差/热/控制周期）、Physics（接触/可变形/流体）、Modeling（CAD/质量惯量估计）。
- **难点：** 分布偏移、感知误差累积、未建模动力学、实时约束。

### 3) Strategy 1 — Domain Randomization + 仿真遥操作

- **DR 思想：** 不追求仿真与真机完全一致，而在训练时随机化可能变化的参数，使策略对区间内任意真值鲁棒。
- **随机化示例：** 视觉（颜色/纹理/光照/相机位姿/背景）；物理（质量摩擦恢复系数、关节阻尼摩擦限位、执行器延迟噪声、传感器噪声）。
- **局限：** 调参偏经验、可能保守变慢、对高动态任务效果有限。
- **数据：** 通过 LeRobot 风格流程在 **Isaac Lab 仿真** 中用真实遥操作臂驱动仿真 SO-101 采集模仿学习演示。

### 4) GR00T / VLA（10）

- **GR00T：** NVIDIA 通用人形/机器人基础模型研究与数据管线平台；本课使用 **GR00T N1.6** 对 SO-101 做 **post-training**。
- **VLA 输入输出：** 相机图像 + 自然语言指令 → 关节位置/速度序列。
- **训练阶段：** Pre-training（互联网规模视觉/语言/视频/机器人演示，学习通用表征）→ Post-training（任务与机器人特化）。

### 5) Strategy 2 — Co-training

- **思想：** 训练时混合 **仿真演示（量大、分布近似）** 与 **真机遥操作演示（量少、分布精确）**。
- **课程示例比例：** 约 5 集真机 + 70–100 集仿真；提供 Hugging Face 数据集与已 post-train 模型，亦可自采 `lerobot-record`（`so101_follower` / `so101_leader`）。

### 6) Strategy 3 — Cosmos 增广

- **动机：** DR 只能改变显式参数；渲染仍偏「合成感」；难以生成全新场景布局。
- **Cosmos：** NVIDIA Physical AI **世界基础模型**；输入机器人演示视频 + 文本 prompt，生成光照/物体位置/外观等一致物理下的多样视频，用于扩充训练集（可结合 depth/edge/seg 等控制权重）。

### 7) Strategy 4 — SAGE + GapONet

- **焦点：** 前述策略主要未系统处理 **执行器层 gap**（ hobby servo 回差沿运动链累积等）。
- **SAGE：** 采集同一运动在仿真与真机的成对数据，按关节对比位置/速度/力矩，量化 gap 并可视化；仓库 [isaac-sim2real/sage](https://github.com/isaac-sim2real/sage)。
- **GapONet：** 建模简单参数调参难以捕获的执行器动力学，用于 targeted bridging。

---

## 对 wiki 的映射

| 知识点 | wiki 页 |
|--------|---------|
| 端到端动手 workflow（本课主线） | [`wiki/entities/nvidia-so101-sim2real-lab-workflow.md`](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md) |
| Sim2Real 概念与 gap 分类 | [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/queries/sim2real-gap-reduction.md`](../../wiki/queries/sim2real-gap-reduction.md) |
| 域随机化 | [`wiki/concepts/domain-randomization.md`](../../wiki/concepts/domain-randomization.md) |
| VLA / GR00T（研究向 Visual Sim2Real 另页） | [`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/entities/gr00t-visual-sim2real.md`](../../wiki/entities/gr00t-visual-sim2real.md) |
| LeRobot 采集 | [`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md) |
| Isaac Lab 仿真训练 | [`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md) |
| SAGE 执行器 gap | [`wiki/entities/sage-sim2real-actuator-gap-estimator.md`](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md) |
| 门户与其他路径 | [`wiki/entities/nvidia-physical-ai-learning.md`](../../wiki/entities/nvidia-physical-ai-learning.md) |

---

## 推荐继续阅读（外部）

- 课程首页：https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- Physical AI 门户：https://docs.nvidia.com/learning/physical-ai/
- SAGE 仓库：https://github.com/isaac-sim2real/sage
