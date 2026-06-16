# Qwen-RobotManip（官方深度博客）

> 来源归档（ingest）

- **标题：** Qwen-RobotManip: Alignment Unlocks Scale for Robotic Manipulation Foundation Models
- **类型：** blog + technical report
- **组织：** Qwen Team
- **原始链接：** <https://qwen.ai/blog?id=qwen-robotmanip>
- **GitHub：** <https://github.com/QwenLM/Qwen-RobotManip>
- **技术报告：** <https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotManip.pdf>
- **入库日期：** 2026-06-16
- **父节点：** [Qwen-Robot Suite 总览](qwen_robot_suite.md)
- **一句话说明：** 面向 **操作泛化** 的 VLA 基础模型：**Qwen3.5-4B VL + flow-matching DiT**，以 **80 维 canonical state-action、相机系 EEF delta、in-context policy adaptation** 实现跨本体对齐，在 **仅开源数据** 下构建 **>38,100h** 预训练语料（含 **Human-to-Robot 合成 24,808h**），并以 **OOD 基准** 为 north star。

## 核心摘录

### 数据规模（三源）

| 来源 | 规模 | 角色 |
|------|------|------|
| 开源机器人数据 | ~11,420h | 单/双臂、移动操作 |
| 开源 egocentric 人视频 | ~1,933h | 物体交互与场景先验 |
| Human-to-Robot 合成 | ~24,808h / 15 本体 | **主要 scaling 引擎**（重定向、去手/inpaint、深度合成） |

- **五阶段 state-action 过滤 + 三阶段跨模态校验**（语言–视频、视觉–状态、帧完整性）。

### 模型设计

1. **Canonical 80-d state-action** + per-dim mask（单臂/双臂/灵巧手/移动底座共用向量空间）。
2. **Camera-frame delta EEF** + **CaPE**（外参进 cross-attn）+ 内参进 visual token + **EEF type embedding**。
3. **In-context policy adaptation：** embodiment prompt（平台、速度、FPS）+ 历史 obs-action chunk；训练时 **随机 context 采样** 防 action-copy。
4. **训练：** 预训练 VLA:VLM = **9:1** 双流；后训练 generalist SFT + **VLA/VLM 共训** 提升 OOD 指令。

### OOD 指标（博客节选）

| 基准 | 亮点 |
|------|------|
| LIBERO-Plus | **91.4%**（Context 版；+7.0 vs π₀.₅） |
| RoboTwin-C2R Hard | **69.4%**（+21.5 vs π₀.₅） |
| RoboCasa365 Composite-Unseen | **14.9%**（约 **3×** 次优） |
| EBench | **45.6% SR**（+18.5 vs 次优） |
| RoboTwin-IF | **72.2%**（+22.6 vs π₀.₅） |
| RoboTwin-XE zero-shot 跨本体 | **23.9%**（3.2× π₀.₅ eef） |
| RoboChallenge Table30 v1 Generalist | **#1，45% SR**（领先第二名 **20%**） |

- **关键论断：** IID 榜（LIBERO / RoboTwin Clean）**无法区分** 是否真预训练；**OOD 轴**（场景/指令/跨本体）才体现 foundation 质量；无统一表示时 **加数据不 scaling**。

### 真机

- 域内 **88.6%** / OOD **87.5%**（7 任务；对比 π₀.₅ 42.9/37.5%，StarVLA 20/0）。
- **130 demo few-shot**、**6K+130 跨 CobotMagic/ARX 技能迁移**、双臂协调与 **反应式重抓**（scale 涌现，非硬编码）。

## 与 Qwen-VLA 的关系（阅读提示）

- **Qwen-VLA** 强调 **操作+导航+轨迹** 同一通才 checkpoint（见 [qwen-vla.md](../repos/qwen-vla.md)）。
- **Qwen-RobotManip** 是 Suite 内 **操作专精** 路线：更强 **跨本体对齐 + Human-to-Robot 合成 + OOD 评测叙事**；架构同为 **Qwen3.5-4B + DiT flow**，但训练目标与数据工程更聚焦 manipulation foundation。

## 对 wiki 的映射

- [Qwen-RobotManip](../../wiki/entities/qwen-robot-manip.md)
- [Qwen-VLA](../../wiki/entities/qwen-vla.md)
- [VLA 方法页](../../wiki/methods/vla.md)
