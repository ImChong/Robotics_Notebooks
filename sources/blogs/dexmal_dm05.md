# DM0.5 | 走出实验室，走向开放世界（Dexmal 官方博客）

> 来源归档（ingest）

- **标题：** DM0.5 | 走出实验室，走向开放世界
- **类型：** blog
- **组织：** Dexmal（大晓智能）
- **原始链接：** <https://www.dexmal.com/blog/dm0.5/index.html>
- **入库日期：** 2026-07-09
- **前代模型：** DM0（2026-02 发布，Dexmal 第一代原生具身基础模型）
- **一句话说明：** Dexmal **DM0.5** 是在 **Gemma3-4B VLM + 680M Action Expert** 上的 **开放世界 VLA 基础模型**：以 **最长约 60s 历史上下文**、**11 类具身 CoT 自回归任务** 与 **动态轨迹对齐（DP 动作匹配）** 强化长程记忆与指令遵循；多源混合预训练覆盖操作、导航与人视频，在 **zero-shot、Table30 v2、LIBERO、RoboTwin2.0、R2R/RxR** 等基准报告 SOTA 或显著领先 **DM0 / π0.5-Droid**。

## 核心摘录

### 相对 DM0 的五项能力主张

| 维度 | 要点 |
|------|------|
| Zero-Shot | 未知开放环境 + 自然语言指令完成任务 |
| Fine-Tuning | 更强基础 → 下游专家更省数据、更稳 |
| 长记忆 | 架构支持历史学习，最长约 **60s** |
| 动作鲁棒 | 光照、相机位姿、人为干扰下保持稳定 |
| 多机型 | 多机型多任务预训练 + 后训练迁移未见本体 |

### 模型架构

- **骨干：** **Gemma3 4B VLM** + **680M Action Expert**（连续动作，**Flow Matching**）。
- **Context Abstraction Layer：** 当前帧 + 多历史 slot（时间/空间抽样 → 固定视觉 token 数）；随机历史长度与增强，支持退化到纯当前帧。
- **Embodiment CoT Tasks（11 类）：** 任务规划（阶段/进度）、事件与环境预测、未来动作/动作语义摘要；把监督从单一动作扩展为「指令理解 + 时序推理 + 动作生成」联合学习。
- **Trajectory Alignment Layer：** 预测固定长度未来动作 chunk，在真实轨迹上用 **单调递增锚点的动态规划** 做 **轨迹进展对齐**（非固定时间点对齐），并约束相邻锚点轨迹连续性，降低遥操作节奏噪声。
- **训练：** 机器人操作 + VLM + 导航 + 视频理解 **混合训练**；VLM 主干 **小 LR**、Action Expert **大 LR**；混合精度分布式。
- **推理：** 默认 **50 步 action chunk**、**10 步 Flow Matching**；**4090 单卡 ~10Hz**、**H100 ~20Hz**。

### 数据

- **机器人操作：** ALOHA、Galaxea R1 Lite、AgiBot G1、Franka、UR5、ARX5、Dexmal 自研双臂移动操作平台等。
- **导航、第一人称人操作、通用 VLM 数据**（含自动生成空间定位/未来状态/反事实推理管线）。
- **清洗：** 异常值、静止帧、无价值动作、关节等价模式去重、子任务 **跨模态自动重标注**。

### 实验摘录

| 基准 | 结果（博客） |
|------|----------------|
| Zero-Shot（Dexmal-Mirror / Franka） | 全面优于 **π0.5-Droid** 与 **DM0** |
| RoboChallenge **Table30 v2** Generalist | **43% SR**，Score **54.42**（文称 SOTA） |
| **LIBERO** 平均 | **99.0%** |
| **RoboTwin2.0** 平均 | **93.5%** |
| **R2R Val-Unseen** SR / NE | **59.7%** / **4.8**（DM0.5-Nav） |
| **RxR Val-Unseen** | 四项指标文称第一 |
| 相机位姿扰动（Franka 九组第三视角） | 成功率 **80–100%**，粗定位 + 腕部精调两阶段 |
| 人为干扰（Dexmal-Mirror） | 目标移动/短时遮挡后仍能重规划继续任务 |

### 记忆案例

- **短程：**「拿起杯子擦桌再复位」——依赖历史恢复杯子初始位姿。
- **长程：** 人类早期示范电池摆放规则，机器人后续执行需 **>1min** 历史；文称 **Video Prompt** 可作新指令形式的 **in-context learning**。

## 对 wiki 的映射

- [DM0.5](../../wiki/entities/dexmal-dm05.md)
- [VLA 方法页](../../wiki/methods/vla.md)
- [Action Chunking](../../wiki/methods/action-chunking.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
