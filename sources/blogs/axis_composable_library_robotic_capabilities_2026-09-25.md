# Beyond More Tasks: Axis Is Building a Composable Library of Robotic Capabilities

> 来源归档（blog / Axis Robotics）

- **标题：** Beyond More Tasks: Axis Is Building a Composable Library of Robotic Capabilities
- **类型：** blog
- **机构：** Axis Robotics（AXIS ROBOTICS）
- **URL：** <https://axisrobotics.ai/blogs/blog/beyond-more-tasks-axis-is-building-a-composable-library-of-robotic-capabilities>
- **发表日期：** 2026-09-25
- **入库日期：** 2026-09-27
- **抓取方式：** WebFetch
- **一句话说明：** 官方实验博客：Grounded RSI 真机共训（22%→52%）、轻量 proxy 轨迹筛选、单任务 Expert 约 $5–10/A100 近满成功率并可链式组合，指向「可组合能力库」而非堆轨迹。

## 三条发现（原文结构）

### Finding 1 — Grounded RSI（Sim2Real 自改进环）

- 大规模仿真可训出具 Sim2Real 能力的策略，**不依赖** 大规模人工真机数据。
- 初部署成功率可较低；部署策略在真机 **自主产生少量成功 rollout**，与原始仿真数据 **共训（co-training）** 可继续提升。
- 实验：**约 22% → 52%** 真机成功率。
- 仿真仍提供 broad coverage；每代 sim 策略部署后产生 **小规模真机校准数据**，修正 deployment shift；真机数据需求 **不必** 随任务数/仿真规模线性增长。
- 叙事：**Recursive self-improvement (RSI), grounded in real robotics** — 每代部署策略产生训练下一代的数据。

### Finding 2 — 轻量 Proxy 选轨迹

- Expert 可低成本扩仿真数据后，**全量加入训练未必更好**；过滤部分仿真轨迹后性能仍升。
- 轨迹价值两维：**对当前模型性能的直接贡献** + **加入全集后的行为覆盖/互补性**。
- 大模型上直接评估海量轨迹不现实 → **轻量 proxy model** 作 data probe，估计 utility 与 diversity，从大池选子集。
- 更小、精心筛选的数据集训练更大视觉模型 **可优于** 全池；scaling 重点转向 **选择、组合与利用** 数据池。

### Finding 3 — 低成本 Expert 与可组合性

- 单任务训到近满成功率：**约 $5–10 纯算力**、**单 A100 约 3–6 小时**（少量用户数据或 agent prior）。
- Expert 非单条固定轨迹：不同初态、扰动、控制扰动下仍可恢复并完成 → **操作域内鲁棒执行**。
- **模块化：** 不要求上一模块把机器人带到精确固定初态；状态落在 Expert 可处理范围内即可接管；已验证 **多 Expert 链式**：上一 Expert 末态 = 下一 Expert 初态 → 长 horizon。
- Expert 成为可重复生产、调用、组合的 **capability unit**。

### Next Step（文内）

- 已有 **数千任务**；scaling 单位从「更多原始轨迹」转向「更多 fully solved skills」。
- Expert 可组合成长任务，或抽象为 packing/sorting/handling/tool grasping/placement 等 **skill family**。
- 另一路径：Expert 持续为统一 **VLA / foundation model** 提供高质量 rollout 与监督，把数千专精能力 **蒸馏** 进单模型；关心 skill family 覆盖后对新物体/布局/实例的泛化。
- 高层 agent（文内类比 GPT-6）负责意图与规划，底层 Expert/skills 负责可靠执行。

## 对 wiki 的映射

- [axis-composable-capability-library](../../wiki/entities/axis-composable-capability-library.md)
- [axis-robotics](../../wiki/entities/axis-robotics.md)

## 开源核查摘要

- **平台/数据管线：** [AxisAIOrg](https://github.com/AxisAIOrg) **部分开源**（见 [axisaiorg.md](../repos/axisaiorg.md)）
- **本篇博客实验栈（Expert / proxy / RSI 共训）：** 截至 2026-09-27 **未列** 独立 GitHub 发布
