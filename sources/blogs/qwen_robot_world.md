# Qwen-RobotWorld（官方深度博客）

> 来源归档（ingest）

- **标题：** Qwen-RobotWorld: Boundless Worlds for Embodied Agents
- **类型：** blog + technical report
- **组织：** Qwen Team
- **原始链接：** <https://qwen.ai/blog?id=qwen-robotworld>
- **技术报告：** <https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotWorld.pdf>
- **入库日期：** 2026-06-16
- **父节点：** [Qwen-Robot Suite 总览](qwen_robot_suite.md)
- **一句话说明：** **语言条件视频生成** 式具身世界模型：以 **自然语言为统一动作接口**，在 **Embodied World Knowledge（EWK，8.6M video-text）** 上联合训练 **操作 / 驾驶 / 室内导航 / 人→机迁移**，**60 层双流 MMDiT** 以 **Qwen2.5-VL** 作动作编码器，支持 **2–4 视角几何一致** 预测与 **Scene2Robot** 跨本体编辑。

## 核心摘录

### 设计动机

- 通用视频模型 **缺 embodied 物理**；专用 embodied WM **难跨场景**。
- **语言统一动作空间：** 关节角、方向盘、航向等 **投影到同一 NL 条件视频生成任务**。

### 架构

- **Understanding stream：** 冻结 **Qwen2.5-VL** 编码语言动作 \(a_t\)。
- **Generation stream：** 视频 VAE latent 表示 \(s_t\)。
- **Joint attention 每层双向融合**；用 **MLLM 而非 T5/CLIP** 作动作编码以带入 **刚性/流体/落体** 等世界知识先验。
- **Scene2Robot：** 人演示 → **14 种机器人形态** 重定向（训练数据引擎 + 推理编辑）。
- **Multi-view：** 训练时多相机帧 **空间拼接**，**asymmetric 3D RoPE**；生成 **2–4 路同步视角** 且 **3D 一致**。

### EWK 数据四轴

| 轴 | 覆盖 |
|----|------|
| Multi-Embodiment | 人手、7 种臂配置、ego 车、移动体；**20+** 机器人模型 |
| Multi-Task | **500+** 动作类；原子技能、长时组合、可变形交互 |
| Multi-Scenario | 真机优先 + 仿真增强（厨房、车间、户外等） |
| Multi-View | ~**1.6M / 6M** embodied 样本含 2–4 视角拼接 |

- **五层标注管线**；训练时 **详/略 caption 等概率采样**。

### 训练课程

- **General → expert 渐进：** 先在 Ego4D / EPIC-Kitchen 等建视觉先验（含 T2I 锚定几何），再 **四阶段 SFT** 加深 embodied，每 batch **保留通用数据** 防遗忘。

### 性能（博客摘要）

- 对比 Sora2 / Veo3 / Wan2.6 / Cosmos / LVP 等：**HSD 0.566**、场景一致 **0.914**、逻辑约束满足；**GR1-Object IF 0.878**；域理解 **0.857**、运动平滑 **0.990**。
- Demo 覆盖 **对比指令**（改一词即改未来）、**多步长指令**、**跨本体 / 跨任务 / 多视角 / zero-shot 鲁棒**；另展示 **Bench2Drive / Waymo 驾驶** 与 **VLNVerse 室内导航** 生成。

## 对 wiki 的映射

- [Qwen-RobotWorld](../../wiki/entities/qwen-robot-world.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)
- [Qwen-Robot Suite](../../wiki/entities/qwen-robot-suite.md)
