---
type: entity
tags: [paper, robotics, robot-learning, nvidia, ucsd]
status: complete
updated: 2026-10-20
project_id: asena-self-evolving-agents
project: https://research.nvidia.com/labs/gear/asena/
arxiv: "2609.39207"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
  - ./paper-embodied-evo-coding-agent.md
  - ./rax-scaling-online-exploration.md
sources:
  - ../../sources/papers/asena_arxiv_2609_39207.md
  - ../../sources/sites/asena-nvidia-gear.md
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "ASENA让 coding agent 通过受监督 robot harness 编程、检查并迭代人形机器人任务；固定模型权重下，经验沉淀在可执行工作区，ASENA-VLN 是可选的单目导航工具。"
---

# ASENA：让具身 Agent 编写、验证并复用机器人程序

ASENA 把 coding agent 接入人形机器人的观测、计算、运动执行和任务记录。Agent 可编写并调整程序，把导航、抓取、全身动作和用户交互组合为长程任务；安全校验与操作者批准约束真机执行。ASENA-VLN 是可选的导航策略，工作区可从执行经验中改进，模型权重保持固定。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| ASENA | Self-evolving Agents for Embodied Navigation | 以程序、执行反馈和持久经验组织人形机器人任务的系统。 |
| VLN | Vision-and-Language Navigation | 根据视觉观测与语言目标进行导航。 |
| EQA | Embodied Question Answering | 通过具身探索收集证据并回答问题。 |
| MCAP | Message Capture | 记录多模态机器人状态、动作、反馈与程序的容器格式。 |
| IK | Inverse Kinematics | 将末端位姿目标转为关节配置。 |
| G1 | Unitree G1 Humanoid | ASENA 真机演示所用人形平台。 |

## 为什么重要

ASENA 的核心不是再训练一个更大的机器人模型，而是把机器人系统变成 coding agent 可检查、可组合、可迭代的程序接口。它将行为能力划分为稳定的硬件接口与可调整的软件工作区：Agent 可以依据本轮记录修复程序，并将可复用技能留给后续任务；评测中模型权重保持不变。

这让导航从孤立的“到达目标”变成搜索、查看、空间推理、抓取、携物和交接等任务的一部分。它也保留了独立视觉导航策略，让 Agent 在需要时把局部移动交给策略执行。

## 方法栈：系统结构

1. **读取现场信息。** Agent 通过统一 robot harness 获取 360° 与 RGB-D 相机、LiDAR、机身状态等观测，并可执行几何计算。
2. **提出导航 / 操作步骤。** 导航可由 LiDAR 占据图规划，也可调用 ASENA-VLN；抓取从 SAM 2 掩膜与深度产生候选，必要时加入 GraspGen-X。
3. **检查计划并执行。** 运动规划与 IK 生成动作；仿真和在线检查筛查关节限制、自碰撞、动作连续性、场景净空与持物空间。物理运动需操作者批准。
4. **记录结果并修正。** MCAP 对齐记录状态、动作、程序、可视化和反馈，供 Agent 回放、修复程序并复用技能。

SONIC 负责全身控制，YAM 夹爪使用单独控制器。独立导航限制会在里程计过期或电量不足时拒绝动作；心跳停止和手动急停作为额外保护。

## 工作区如何演进

ASENA 将每次运行的传感器与机器人状态、动作、反馈、程序和可视化保存为 MCAP 记录。回放记录后，Agent 可以找出程序意图与实际结果的偏差，修正出错步骤，并把有效笔记和可执行技能写入持久工作区。

重复任务的仿真评测中，16 个 worker 分别在私有工作区执行与反思，随后由 improver 聚合轨迹和反馈、整理共享工作区，再开始下一轮。更新对象是程序与经验，模型和控制器权重保持固定。每轮结果仍会波动，因此工作区改进并非单调保证。

## ASENA-VLN：可选单目导航策略

ASENA-VLN 基于 Qwen3-VL-4B-Instruct，读取最多 8 帧前向 RGB 历史与结构化导航指令，预测最多 8 个机身坐标 waypoint（x、y、朝向）。指令可指定结束姿态、参照物锚点或运动限制。策略可由 coding agent 调用，也可以和几何规划组合；它不是 ASENA 系统的强制控制器。

作者用 MP3D 与 HM3D 的场景几何生成并验证 9 类原子导航任务，包括定距移动、转向、定向接近物体、对齐、进房间、穿门、走廊跟随和楼层切换。训练混合 R2R、RxR、FGR2R、ScaleVLN、SRDF、ObjectNav 与多视角 VQA；像素目标 token 是辅助监督，评测使用 waypoint 轨迹 token。

## 实验与评测

| 评测 | 结果 | 读法 |
|---|---|---|
| ASENA-VLN，R2R val-unseen | 成功率 68.7%，SPL 64.2% | 独立单目导航策略结果。 |
| ASENA-VLN，RxR val-unseen | 成功率 70.2%，SPL 59.7% | 指标口径同上。 |
| Agent + VLN，Claude Sonnet 5 | R2R、RxR 成功率各提高 11 个百分点 | 收益较明显；取决于 coding agent 后端。 |
| Agent + VLN，GPT-6 Astra | 工具调用降低 46.9% / 51.0% | R2R / RxR；RxR 成功率略降，强后端不一定因加策略而更准确。 |
| 100 项重复导航任务，10 轮工作区演进 | R2R 72% → 98%；RxR 65% → 89% | 使用模拟器反馈，模型权重固定；评测重复相同任务集。 |
| G1 Jetson AGX Thor 推理 | FP8 + 缩减词表：877 ms，3.20× PyTorch BF16 | 较早 checkpoint 的补充实验，不是主基准 checkpoint，也不等于整套任务端到端耗时。 |

具身问答的演进结果也展示准确率与探索步数之间的权衡：HM-EQA 冻结工作区第 17 轮达到 81.2%，MT-HM3D 冻结工作区第 4 轮达到 66.5%。这两个数字来自不同基准与轮次，不能合并成一个统一得分。

## Unitree G1 演示

论文报告四个无预建地图的真机任务：搜索卫生间位置并返回指路；搜索 pantry 并核实咖啡设备；比对零食与自动售货机库存并用全身手势回答；拾取罐装物品、携带行走并交给用户。系统在线建立地图，并将视觉和几何证据、空间记忆、导航、抓取与动作程序串联。

这些是能力组合的定性演示，不是不同布局与用户下的系统可靠性统计。论文指出真机任务约需 10–20 分钟；要走向稳定部署，还需更快的推理执行、更多重复任务可靠性数据，以及适应动态环境的安全机制。

## 与其他工作对比

| 工作 | 改进对象 | 经验如何保留 | 区别 |
|---|---|---|---|
| [EmbodiedEvo](./paper-embodied-evo-coding-agent.md) | coding agent 的 Skills / Lessons 与任务解决流程 | 将 Playground 中验证的经验提炼为技能，再供后续 agent 复用 | EmbodiedEvo 聚焦由 agent 自动演进可复用代码能力；ASENA 将这套思路落到具身 robot harness、受监督执行、机器人安全边界及可选 VLN 策略。 |

## 工程实践与局限

- **代码状态：** 截至 2026-10-16，核查到的 NVIDIA GEAR 官方项目页未列训练 / 推理代码、模型权重或数据集下载链接；当前没有可验证的官方复现入口。
- **不是在线更新模型权重：** 论文中的自我改进主要来自修改程序、笔记和技能；固定模型在更新后的工作区上运行。
- **验证范围有限：** 模拟器中的重复任务使用相同任务子集和反馈；这不能单独证明对新任务、场景或用户的持续泛化。
- **真实机器人证据为定性演示：** G1 展示任务组合能力，但没有多布局、多对象、多用户的重复可靠性统计；论文报告单次任务仍需 10–20 分钟。
- **推理延迟需读清边界：** Thor 的 877 ms 是较早 checkpoint、指定 batch / 帧历史 / 解码设置的策略测量；不代表整个 robot harness 或 100 Hz 低层控制闭环的周期。
- **安全结构是方法的一部分：** 真机执行仍需操作者批准、运动检查、独立导航限制和急停；不应将其简化为无监督自主执行。

## 源码运行时序图

**不适用：** 截至 2026-10-16，项目页没有链接官方可运行训练、推理或部署仓库；因此无法核实源码入口与调用关系。正文的系统结构说明论文方法，不代表源码运行链路。

## 结论

**ASENA 的主要贡献是把具身任务表达为可检查、可修复、可复用的机器人程序，并以执行记录驱动工作区演进；学习型 VLN 是可选能力模块。**

1. 判断“自我演进”时应区分工作区内容更新与模型权重更新；本文评测固定模型权重。
2. ASENA-VLN 输出短序列机身 waypoint；coding agent 决定是否调用并检查后续进展。
3. 导航策略对 Claude Sonnet 5 的成功率收益明显，对 GPT-6 Astra 的收益更多体现为减少工具调用，且结果依任务变化。
4. R2R / RxR 的 10 轮改进基于重复的 100 项任务和模拟器反馈，不能视作未见任务的真机泛化结论。
5. G1 展示覆盖搜索、检查、手势与物体交接，但可靠性、任务时长与安全仍是部署瓶颈。
6. Thor 量化性能数字来自早期 checkpoint 的补充测试；官方项目页目前未提供可复现代码或权重下载入口。

## 关联页面

- [Day 5：世界模型与决策导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [EmbodiedEvo](./paper-embodied-evo-coding-agent.md) — coding agent 将交互经验沉淀为技能。
- [生成式世界模型](../methods/generative-world-models.md) — 与 ASENA-VLN 所用视觉语言导航模型相关的训练与生成范式。

## 与 RAX 的关系

- [RAX](./rax-scaling-online-exploration.md) 同样让 coding agent 通过机器人接口将交互经验写入可复用程序；RAX 聚焦扩展在线探索并将成功轨迹编译为程序，ASENA 还覆盖人形任务 harness、MCAP 回放与导航策略组件。两者的评测协议不同，数字不可直接比较。

## 参考来源

- [ASENA 论文 v2 摘录](../../sources/papers/asena_arxiv_2609_39207.md)
- [NVIDIA GEAR ASENA 项目页核查](../../sources/sites/asena-nvidia-gear.md)
- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)

## 推荐继续阅读

- [arXiv:2609.39207 v2](https://arxiv.org/html/2609.39207v2)
- [NVIDIA GEAR ASENA 项目页](https://research.nvidia.com/labs/gear/asena/)
