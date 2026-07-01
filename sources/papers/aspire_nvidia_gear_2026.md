# ASPIRE：Agentic /Skills Discovery for Robotics

> 来源归档（ingest）

- **标题：** ASPIRE: Agentic /Skills Discovery for Robotics
- **全称：** Agentic Skill Programming through Iterative Robot Exploration
- **类型：** paper（arXiv preprint；公开 PDF 2026-06-30）
- **机构：** NVIDIA GEAR Lab；UMich；UIUC；UC Berkeley；CMU
- **原始链接：**
  - 项目页：<https://research.nvidia.com/labs/gear/aspire/>
  - PDF：<https://research.nvidia.com/labs/gear/aspire/assets/Aspire.pdf>
  - 站点归档：[`sources/sites/nvidia-research-aspire.md`](../sites/nvidia-research-aspire.md)
- **入库日期：** 2026-07-01
- **一句话说明：** **持续学习** 的 code-as-policy 系统：闭环执行引擎提供 **逐原语多模态 trace** 供 coding agent 诊断与修补，**进化搜索** 探索多样控制程序，并把验证过的修复 **写入可扩展技能库**——让后续任务以 in-context 技能指导加速适应，并在仿真→真机、跨具身上降低编程 token 成本。

## 核心论文摘录（MVP）

### 1) 问题：coding agent 调试机器人程序缺细粒度证据，且经验不累积

- **链接：** Aspire.pdf §1 Introduction
- **摘录要点：** 机器人失败可能来自感知、规划、抓取、接触或长时程协调；仅 **任务级成败** 无法归因。现有系统完成后 **丢弃修复策略**，第 100 个任务与第一个任务 **同等无经验**。人类工程师则通过回放执行、检查 overlay/轨迹、内化 **可迁移恢复启发式** 而越调越快。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — 「经验复利」作为与 ENPIRE 策略搜索并列的 GEAR 叙事。

### 2) 闭环执行引擎：逐原语 multimodal trace 取代粗反馈

- **链接：** Aspire.pdf §2.1；Fig. 2（BEHAVIOR-1K radio 任务）
- **摘录要点：**
  - 每次 primitive 调用记录 API、输入输出、状态码及 **RGB 关键帧、overlay、抓取候选、规划结果** 等；**不给完整视频**，只保留调用前后帧与相关证据。
  - 示例：感知成功但 `navigate_to_pose` 反复 `PLANNING_ERROR` → trace 定位目标落在桌子碰撞缓冲区内 → agent 写入 **multi-angle approach** 绕行补丁 → 验证后入库为 **Multi-Angle Approach** 技能。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — 执行引擎与 trace-guided debugging 机制。

### 3) 技能库：验证修复 → 可检索 in-context 知识

- **链接：** Aspire.pdf §2.2；Fig. 3
- **摘录要点：** 技能涵盖 **定位消歧、运动原语构造、导航恢复、抓取约束、场景理解、调试工作流** 等异质类别；**非**固定人工原语集合。Coordinator 将 actor 验证过的修复写入共享库；未来 actor 检索技能作为 **in-context 指导**，上下文聚焦任务规格、当前程序与失败 trace。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — 技能库结构与跨任务迁移。

### 4) 进化搜索：超越单轨迹自改进

- **链接：** Aspire.pdf §2.3；Fig. 6(c)
- **摘录要点：** 在单轮 trace 调试之后，采样多组修复假设并行 rollout；下一代条件于 **存活程序 + 残余失败 trace**。前几轮迭代收益最大，后续递减但仍有助于难例。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — 与执行引擎的消融分工（引擎 +48 pp，进化搜索再抬难例）。

### 5) 主评测：LIBERO-Pro / Robosuite / BEHAVIOR-1K

- **链接：** Aspire.pdf §3.2–3.4；Fig. 4–5
- **摘录要点：**
  - 基线：**CaP-Agent0**（每 seed 重生成程序 + test-time reasoning）、OpenVLA、π₀.₅、人类专家程序。
  - ASPIRE 在 debug seed 上学习并 **每任务只生成一个程序**，在更大 held-out seed 上评测。
  - 宏平均：LIBERO-Pro 三套件 Pos/Task 轴上较最强基线 **+77 / +41.5 / +42.5 pp**；Robosuite 双手递送 **20% → 92%**；BEHAVIOR-1K 导航与任务成功率均超人类与 CaP-Agent0。
  - **LIBERO-Pro Long 零样本：** 在 LIBERO-90 积累技能库后 **31%** vs 基线 **4%**；技能库越大零样本越高（Fig. 5b）。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 操作基准交叉引用。

### 6) 真机跨具身：仿真技能降低真机编程 token

- **链接：** Aspire.pdf §3.6；Table 1
- **摘录要点：** Franka 仿真中发现的三个技能（soda-can、bowl-on-plate、drawer）作为 in-context 指导用于 **YAM 双臂真机**（不同 embodiment 与 API）。**非权重部署**；agent 仍需真机 trace 调试。检索技能后 **总 token 可降一个数量级**；drawer 任务无技能基线耗尽预算仍 0/20，有技能 **11/20**。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — sim-to-real **know-how 迁移**（程序/技能层，非策略权重）。

### 7) 局限与未来工作

- **链接：** 项目页 Limitations；Aspire.pdf §4
- **摘录要点：** frontier LLM 依赖、原语 API 边界、技能库陈旧/冗余管理、真机 autonomy 与安全、搜索 **token/rollout 成本**。
- **对 wiki 的映射：**
  - [ASPIRE](../../wiki/methods/aspire.md) — 常见误区与局限节。

## 当前提炼状态

- [x] 三组件系统、执行引擎 trace、技能库、进化搜索、主基准结果、零样本迁移、真机跨具身与局限已摘录到可维护粒度
- [ ] arXiv 号确认后补充 HTML/PDF 锚点
- [ ] 官方代码发布后应单独 `sources/repos/` 索引并回链

## 对 wiki 的映射（汇总）

- [ASPIRE](../../wiki/methods/aspire.md)
- [ENPIRE](../../wiki/methods/enpire.md)
- [NVIDIA GEAR Lab](../../wiki/entities/nvidia-gear-lab.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
