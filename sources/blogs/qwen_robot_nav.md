# Qwen-RobotNav（官方深度博客）

> 来源归档（ingest）

- **标题：** Qwen-RobotNav: A Scalable Navigation Model Designed for an Agentic Navigation System
- **类型：** blog + technical report
- **组织：** Qwen Team
- **原始链接：** <https://qwen.ai/blog?id=qwen-robotnav>
- **GitHub：** <https://github.com/QwenLM/Qwen-RobotNav>
- **技术报告：** <https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf>
- **入库日期：** 2026-06-16
- **父节点：** [Qwen-Robot Suite 总览](qwen_robot_suite.md)
- **一句话说明：** 基于 **Qwen3-VL** 的 **可配置上下文导航基座**：**任务模式**（VLN / PointNav / ObjNav / Tracking / 自动驾驶）与 **观测参数**（token budget、时间衰减、相机权重、帧采样）在训练时随机化，推理时可外部指定，**15.6M 样本** 五域 **单权重**，并作为 **Qwen3.7-Plus** 等上层 planner 的导航原语。

## 核心摘录

### 可控观测协议（类比 MCP）

- 指令跟随需要 **长时记忆**；目标跟踪依赖 **近帧**；物体检索在 episode 内 **探索→接近** 切换策略。
- **四控制轴**（训练时每样本随机，推理可配）：
  - Visual token budget
  - Temporal decay
  - Camera weights
  - Frame sample mode（random vs latest）
- **架构：** Qwen3-VL + **4 层 MLP** 输出 **8 个航点**（位置 + 朝向）；相机身份与时间顺序通过 **与视觉 token 交错的自然语言 tag** 注入。

### Agentic 两层系统

- **上层：** Qwen3.7-Plus 分解长时目标、切换 task mode 与 context。
- **下层：** RobotNav 执行子段航点预测。
- **双层记忆：** 每段轨迹摘要 + **evidence notebook**（已搜区域、候选位置、被拒绝假设）。

### 训练与数据

- **15.6M** 样本，五任务族 + VLM 推理数据保 grounding。
- **2B→8B** 稳定 scaling，长时推理增益最大。
- **T2V→轨迹管线：** prompt → 视频合成 → VLM 质检 → 单目深度 → 运动学滤波 → **+40K** 逼真样本（无 3D 重建）。

### 公开指标（博客）

| 基准 | 结果 |
|------|------|
| VLN-CE R2R（8B） | **72.1% SR** |
| VLN-CE RxR（8B） | **76.5% SR** |
| HM3Dv2 ObjNav RGB-only（4B） | **75.6% SR**，平均距目标 **1.72m** |
| EVT-Bench tracking | **90.0%** |
| NAVSIM PDMS（4B） | **91.4** |
| Agentic + 3× EQA | **新 SOTA**（含 EXPRESS-Bench **+15.4%**、步数 **-77%** vs prior best） |

### 真机

- **Unitree Go2 + Jetson Thor：** **196ms（5.1 Hz）**，仅内置低分相机，**zero-shot** 未见环境。
- Agent 模式示例：「检查 Cotti Coffee 是否遗留绿色雨伞」→ 分解、导航、取证、回答。

## 对 wiki 的映射

- [Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md)
- [VLN 任务页](../../wiki/tasks/vision-language-navigation.md)
- [Qwen-Robot Suite](../../wiki/entities/qwen-robot-suite.md)
