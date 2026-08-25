# 万字长文 ｜ GEN-1.5 火了，但机器人的「上下文学习」到底在学什么？

> 来源归档（blog / 微信公众号）

- **标题：** 万字长文 ｜ GEN-1.5 火了，但机器人的「上下文学习」到底在学什么？
- **类型：** blog
- **作者：** 具身智能之心（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/V_Dm8kHvB2YxtGY7qScjXA
- **发表日期：** 2026-08-25
- **入库日期：** 2026-08-25
- **抓取方式：** `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_heart_robot_icl_gen15_2026-08-25/`](../raw/wechat_embodied_heart_robot_icl_gen15_2026-08-25/)
- **一句话说明：** 以 GEN-1.5 **physical prompting** 为入口，系统拆解机器人「上下文」消解的三类不确定性（映射选择 / 状态估计 / 映射本身），并按示范来源（遥操作、人视频、任务无关探索）与规模涌现（GEN-1.5、Qwen-RobotManip）组织 26 篇相关工作；强调仅第三类才是真 ICL。

## 核心摘录（归纳，非全文）

### 触发案例：GEN-1.5 physical prompt

- **机制：** 3–12 秒 **sensorimotor 示范**（人/机/仿真 rollout）插入 **30 秒上下文**，**无梯度** 执行新短程任务；官方自报 10 任务 one-shot **~59%（±10%）**，10 步微调 **~83%（±9%）**。
- **与显式 ICL 训练的区别：** 未改架构、未加 ICL 辅助损失；能力归因于 **8+ 月持续预训练** 涌现（闭源自报）。
- **衍生：** 组合多段 prompt 链成长程行为；仿真 prompt 零样本真机；人用手示范→机器人手；1–10 步极少数据适应。

### 为什么机器人需要上下文

- 标准 VLA 形式 \(a_t = \pi(o_t, \ell)\) 隐含 **马尔可夫假设**；部署时相机/本体/任务阶段变化使 **当前帧不足以定动作**。
- 类比 n-gram 语言模型：长程依赖 + 数据稀疏 → 需要更长上下文或更好的归纳机制。

### 三类不确定性（taxonomy 核心）

| 类型 | 上下文装什么 | 读完之后什么变了 | 是否「学习」 |
|------|--------------|------------------|--------------|
| **映射选择** | 语言、目标图、episode metadata | 从权重已有映射中 **选一个** | 否 |
| **状态估计** | 执行历史、记忆 token | **代入映射的状态** 变了 | 否 |
| **映射本身** | 示范轨迹、人视频、系统辨识片段 | **观测→动作函数** 变了 | **是（真 ICL）** |

### 按示范来源分线

**遥操作轨迹：** One-Shot IL → ICRT（序列 next-token）→ Instant Policy（图 diffusion）→ KAT / LipVQ-VAE action tokenizer → BPP / StellaVLA；配对数据来自同任务多示范重组或仿真生成（SynthICL）；RICL 在 π0-FAST 上小规模 post-training。

**人类视频：** Vid2Robot（cross-attn + 对比对齐）、MimicDroid（无配对人视频 + retargeting）、Point Policy（语义关键点统一表示）。

**任务无关随机运动（系统辨识）：** ICWM — 任务前数秒随机运动的三元组作 prompt，归纳 **当前相机/本体下的动作–视觉映射**，非任务映射。

### 预训练规模与涌现

- **Qwen-RobotManip：** 近期 H 个 (o,s,a) chunk 作 in-context 行为风格适配；**stochastic context sampling** 防止退化为「复制最近 chunk」。
- **GEN-1.5：** 迄今唯一报告 **无显式 ICL 训练** 下 one-shot 涌现的产业案例（闭源）。

### 非学习型上下文与正交路径

- **π0.7：** metadata / subgoal image → **映射选择**（非 ICL）。
- **MemoryVLA / MemER / ContextVLA / MEM / HiMe：** **状态记忆**；形式与示范轨迹相似但机制不同。
- **RoboTTT / VANE：** **test-time training** — 证据写进权重而非上下文；与 ICL 目标重叠、代价不同。

### 开放问题（文内）

1. 机器人 ICL 如何涌现？数据分布 / 规模 / 与显式 ICL 的泛化差异。
2. demo 最终形态：token 序列 vs 图 vs 关键点 vs 结构化计划 vs 原始感觉运动序列。
3. long-context scaling：控制频率下每步推理成本 vs 语言模型延迟问题。

## 对 wiki 的映射

- **父概念页（新建）：** [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md)
- **交叉更新：** [GEN-1.5 实体](../../wiki/entities/generalist-gen15-one-shot.md)、[imitation-learning](../../wiki/methods/imitation-learning.md)、[foundation-policy](../../wiki/concepts/foundation-policy.md)、[manipulation 任务页](../../wiki/tasks/manipulation.md)

### 文内论文 → 既有 wiki 节点（复用，不新建）

| 论文 | wiki |
|------|------|
| GEN-1.5 | [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) |
| RoboTTT | [paper-robottt-test-time-training-vla-context](../../wiki/entities/paper-robottt-test-time-training-vla-context.md) |
| BPP | [paper-behavior-prompting-policy](../../wiki/entities/paper-behavior-prompting-policy.md) |
| MimicDroid | [paper-notebook-mimicdroid](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md) |
| Qwen-RobotManip | [qwen-robot-manip](../../wiki/entities/qwen-robot-manip.md) |
| π0.7 | [pi07-policy](../../wiki/methods/pi07-policy.md) |
| WAM-TTT（对照 ICL） | [paper-wam-ttt](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) |

### 文内其余论文（本 ingest 仅索引，待单篇 ingest）

ICRT、Instant Policy、KAT、SynthICL、RICL、Vid2Robot、Point Policy、ICWM、StellaVLA、MemoryVLA、MemER、ContextVLA、MEM、HiMe、Gated Memory Policy、VANE 等 — 见文内参考文献 3–26。
