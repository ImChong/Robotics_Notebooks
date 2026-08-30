# Physical Agentic AI（多机器人编排）

> 来源归档（ingest）

- **标题：** Physical Agentic AI: An Architecture for Orchestrating a Robot Crew with LLMs
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22657>
- **机构：** 亚利桑那州立大学（Arizona State University）
- **作者：** Xinyuan Liu、Eren Sadikoglu、Riana Chatterjee、Ransalu Senanayake
- **代码：** <https://github.com/Liuuuxy/physical-agentic-ai>
- **入库日期：** 2026-08-30
- **一句话说明：** 语义规划与物理执行分离：无执行权的 Mission Planner 出 JSON 计划，确定性 Robot Orchestrator 按工作流契约逐项验证技能调用。

## 核心摘录（MVP）

### 1) 接地不等于安全

- **摘录要点：** 给 LLM 本体能力、物理前提和跨机协作信息后，技能落地可从 51% 升到 96%，但知情规划器仍会派发 23–29% 的故障步。检索改善接地，不消除不可行/错时/不安全动作。
- **对 wiki 的映射：**
  - [Physical Agentic AI](../../wiki/entities/paper-physical-agentic-ai.md) — 问题设定。

### 2) 契约驱动的编排器

- **摘录要点：** 每个机器人暴露带类型的可执行技能库。Planner 分解任务并分配机器人–技能对，但不执行。Orchestrator 根据状态、命名位置和工作流契约一次授权一个技能。契约同一份 spec 既渲染 prompt 又驱动门控，避免漂移。
- **对 wiki 的映射：**
  - [Physical Agentic AI](../../wiki/entities/paper-physical-agentic-ai.md) — 架构。

### 3) 评测分层

- **摘录要点：** 无人机–UGV 搜索派遣在 Gazebo 全量 live 执行；人形–四足搬运用硬件等价技能接口 + Unitree G1 / Go2 两次真机试验。逐项 enforcement 把错误派发降到 **0%** 且无误拦；held-plan 消融证明是门控而非计划变化负责。注入 8 个故障：无门控则全部越过编排边界、6 个产生运动；有门控则全部在运动前拒绝。
- **对 wiki 的映射：**
  - [physical-agentic-ai 仓库](../repos/physical-agentic-ai.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **已开源** MIT。`crew_g1_go2/`（真机 ROS 2）+ `sar_ws/`（Gazebo + PX4 SITL）。三层：`mock`（`CREW_SIM=1` / `SAR_SIM=1` 打桩）、`live-gazebo`、`hardware`。203 个 hermetic 单测可离线跑。README 写明 mock 指标是规划层，不是物理结果。

## 当前提炼状态

- [x] arXiv 摘要与实验节已对齐
- [x] 仓库 README 与入口已核查
- [x] wiki 映射：`wiki/entities/paper-physical-agentic-ai.md` 新建
