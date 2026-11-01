# RoboSkill 官方代码仓库

> 来源归档（repo）

- **标题：** Explore, Execute, Evolve: A Skill Acquisition and Reuse Loop for Embodied Agents
- **类型：** repo（具身智能体技能获取与复用；论文复现实验代码）
- **链接：** <https://github.com/SII-dannyXSC/RoboSkill>
- **论文：** <https://arxiv.org/abs/2609.37810>（见[论文来源归档](../papers/roboskill_arxiv_2609_37810.md)）
- **许可证：** MIT（官方 LICENSE）
- **入库日期：** 2026-10-27
- **一句话说明：** 官方源码提供论文使用的原生 Agent CLI 执行路径，以 LIBERO Gateway、单 agent Harness 和批量实验调度器连接任务运行与技能获取/复用。

## 仓库结构与运行入口（依据 README）

| 路径 | 用途 |
|---|---|
| `libero/` | LIBERO Gateway、受信 SDK / proxy 与 observation profiles；Gateway 管理模拟器侧信息和 agent-facing 接口 |
| `harness/` | 单 agent 生命周期，负责连接一个 LIBERO lease 与原生 Codex CLI 或 Claude Code |
| `agent_runner/` | 批量 task/seed 调度、并发限制、结构性启动故障重试及实验配置 |
| `scripts/run_libero_gateway.sh` | 启动 LIBERO Gateway 的 README 示例入口 |
| `agent_runner/run.py` | 启动 baseline、acquisition、program-prior 或 tactile 等实验配置 |

## 运行与复现边界

- README 的基本流程是准备固定的 LIBERO 环境、启动 Gateway、复制并修改实验配置，然后通过 agent_runner 启动 batch。
- 论文中的技能获取配置会启用同一 session 的成功后 reviewer；reviewer 读取完成的运行记录，输出经校验的 action ledger、keyframes、scene-local binding、summary 和实际运行代码的逐字节副本。
- 发布仓库为 source-only：不包含运行工作区、日志、session、视频、凭据、历史结果以及生成的经验包或技能。运行时生成的文件位于忽略的 `.runtime/` 下。
- README 提供 `make smoke` 与 `make verify-publication` 检查入口；完整数值复现还需要对应模型版本和等效 GPU 资源。
- README 标注的 tactile setting 使用 GPT-6 Astra，在 LIBERO-10 的 10 个任务和 seeds 0–4 上评测；严格条件移除力相关观测与夹爪状态，保留模拟器和七维动作空间。

## 对 wiki 的映射

- 统一实体：[wiki/entities/paper-roboskill.md](../../wiki/entities/paper-roboskill.md)
- 论文来源：[sources/papers/roboskill_arxiv_2609_37810.md](../papers/roboskill_arxiv_2609_37810.md)
