# Qwen-Robot Suite（官方总览博客）

> 来源归档（ingest）

- **标题：** Qwen-Robot Suite — Bridging Language and Physical Action
- **类型：** blog（Qwen Team 官方研究博客）
- **组织：** Qwen Team（阿里巴巴通义）
- **原始链接：** <https://qwen.ai/blog?id=qwen-robotsuite>
- **镜像：** <https://qwenlm.github.io/blog/qwen-robotsuite/>
- **入库日期：** 2026-06-16
- **一句话说明：** 通义 **Qwen-Robot Suite** 以 **Qwen-RobotNav / Qwen-RobotManip / Qwen-RobotWorld** 三模型分别对齐 **移动、操作、世界预测** 三类物理动作，并可通过 **语言优先接口** 与通用 Qwen 规划器（如 Qwen3.5 / Qwen3.7-Plus / Qwen-Omni）组合成 agentic 闭环。

## 子节点（深度博客）

| 子模型 | 深度博客 | GitHub | 技术报告 PDF |
|--------|----------|--------|--------------|
| **Qwen-RobotNav** | [qwen-robotnav](https://qwen.ai/blog?id=qwen-robotnav) | [QwenLM/Qwen-RobotNav](https://github.com/QwenLM/Qwen-RobotNav) | [Qwen_RobotNav.pdf](https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf) |
| **Qwen-RobotManip** | [qwen-robotmanip](https://qwen.ai/blog?id=qwen-robotmanip) | [QwenLM/Qwen-RobotManip](https://github.com/QwenLM/Qwen-RobotManip) | [Qwen_RobotManip.pdf](https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotManip.pdf) |
| **Qwen-RobotWorld** | [qwen-robotworld](https://qwen.ai/blog?id=qwen-robotworld) | （博客未链公开仓库，以 PDF 为准） | [Qwen_RobotWorld.pdf](https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotWorld.pdf) |

本仓库 sources 子页：[`qwen_robot_nav.md`](qwen_robot_nav.md)、[`qwen_robot_manip.md`](qwen_robot_manip.md)、[`qwen_robot_world.md`](qwen_robot_world.md)。

## 核心摘录（归纳，非全文）

### 1) 问题：理解 ≠ 行动

- **VLM 已能规划语言步骤**（如「去厨房、找红杯、拿起、放到架子上」），但 **无法直接输出可执行电机指令**。
- **具身数据** 与互联网文本本质不同：导航轨迹、遥操作抓取、行车片段 **动作空间 / 观测格式 / 本体** 异构；简单池化易 **冲突而非协同**。

### 2) 三模型分工

| 模型 | 对齐域 | 骨干 / 头（博客摘要） | 要点 |
|------|--------|----------------------|------|
| **Qwen-RobotNav** | 移动 / 导航 | Qwen3-VL + 轻量 MLP 航点头 | 参数化 **任务模式 + 观测上下文**（token budget、时间衰减、相机权重、帧采样）；15.6M 样本；五类导航任务 **单权重** |
| **Qwen-RobotManip** | 操作 | Qwen3.5-4B VL + flow-matching DiT | **80 维统一 state-action**、**相机系 EEF delta**、**in-context policy adaptation**；>38,100h 预训练（含 Human-to-Robot 合成） |
| **Qwen-RobotWorld** | 世界模型 | 双流 MMDiT + Qwen2.5-VL 动作编码 | **自然语言统一动作接口**；EWK 8.6M video-text；20+ 本体、500+ 动作类 **联合训练** |

### 3) Agent 闭环与工具化

- 三模型均暴露 **language-first 接口**，通用 Qwen 模型可将其作为 **物理世界工具** 调用。
- **Qwen-RobotClaw**（文内 in-house harness）：管理长时程上下文与记忆，供 VLM agent 调用 Suite 模型。
- **Manip + Planner：** Qwen-3.5 高层分解原子子任务，RobotManip 低层执行；OOD 桌面清理等 **组合泛化** 优于单 VLA。
- **Nav + Agent：** 与 Qwen-RobotNav 组合在 **HM-EQA / MT-HM3D / EXPRESS-Bench** 等长时程 3D 探索任务上 **显著优于 prior SOTA**；真机开放世界寻物（如找可用洗手间、找回雨伞）有 demo。
- **Qwen-Omni × RobotManip：** 语音随机提议操作任务并实时评判，无预定义任务表。
- **Chat2Robot（实验功能）：** 浏览器内自然语言控真机；当前仅 **RobotManip**，策略仅 **RoboTwin-Clean 50 任务** 训练（非完美策略，演示 zero-shot 指令跟随）。

### 4) 真机亮点（总览文内）

- **Go2 四足：** RobotNav **zero-shot**，Jetson Thor **196ms**，内置低分辨率单相机；公寓多房间逐步 verbal 导航；展会厅 **21.78m 往返** 精确折返。
- **操作：** 跨场景 / 未见指令 / **跨本体 zero-shot & few-shot** 有图集叙事。

### 5) 路线判断（文内）

- 物理智能仍处早期；**接触丰富长时程、终身学习、规划–执行更紧耦合、 richer HRI** 仍开放。
- 路径：**强多模态理解 → 分域对齐 VLM 表示到物理动作 → 规模化训练 → 坚持泛化**。

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [Qwen-Robot Suite](../../wiki/entities/qwen-robot-suite.md) | 套件总览实体页（含 Mermaid 闭环） |
| [Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md) | 导航 / agentic navigation 子实体 |
| [Qwen-RobotManip](../../wiki/entities/qwen-robot-manip.md) | 操作 VLA 子实体 |
| [Qwen-RobotWorld](../../wiki/entities/qwen-robot-world.md) | 语言条件视频世界模型子实体 |
| [Qwen-VLA](../../wiki/entities/qwen-vla.md) | 交叉引用：通才 VLA 与 Suite 内 Manip 路线对照 |
| [VLN](../../wiki/tasks/vision-language-navigation.md) | RobotNav 基准与 agentic 导航语境 |

## 外部参考

- [Qwen-Robot Suite 官方博客](https://qwen.ai/blog?id=qwen-robotsuite)
- [Qwen-RobotNav GitHub](https://github.com/QwenLM/Qwen-RobotNav)
- [Qwen-RobotManip GitHub](https://github.com/QwenLM/Qwen-RobotManip)
