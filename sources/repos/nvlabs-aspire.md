# NVlabs/ASPIRE — 官方源码仓库

> 来源归档（ingest）

- **类型：** repo（机器人 coding-agent 持续学习 / code-as-policy）
- **仓库：** <https://github.com/NVlabs/ASPIRE>
- **项目页：** <https://research.nvidia.com/labs/gear/aspire/>
- **论文：** ASPIRE: Agentic /Skills Discovery for Robotics，arXiv:2607.00272
- **许可证：** Apache-2.0（项目材料；NOTICE 中列出的第三方组件保留各自许可）
- **核查日期：** 2026-10-18
- **关联归档：** [官方项目页](../sites/nvidia-research-aspire.md)；[论文摘录](../papers/aspire_nvidia_gear_2026.md)
- **一句话说明：** ASPIRE v1.0 将 coding-agent 机器人程序调试、rollout 验证和可复用技能库整理成开源工作流；README 提供 Codex / Claude Code 的仿真 Agent Skills 与多个基准的复现入口。

## 官方发布信息

- README 标注 2026-10 发布 v1.0：支持真机魔方在线技能学习，并展示仿真到真机的长时程桌面整理、叠杯、药片入瓶和拧瓶盖案例。
- 仓库报告在 80 个 LIBERO-Pro Goal / Object / Spatial / Long 任务上达到 99% 成功率，并覆盖部分 Robosuite 与 BEHAVIOR-1K 任务。该数值是仓库发布摘要，须结合具体模型、任务集合和 held-out 评测协议解读。
- 仿真技能以 in-context 指导迁移到不同具身和 API 的 YAM 真机；官方描述的是程序 / know-how 迁移，不是策略权重直接部署。
- 提供 iterative-debugging 与 evosearch 技能及基准 runbook；仓库 README 指向 CUDA、uv 与 gated SAM3 权重 token 的主机准备说明。

## 安全与复现边界

- README 明确指出，ASPIRE 执行 LLM 生成的 Python，代码具有完整 import 权限；trial / watchdog 不是安全沙箱。
- 官方建议在隔离主机运行、限制网络与敏感挂载，并且**不要给仿真 agent 真实硬件访问**。真机任务需遵循独立操作规程与操作员授权。
- 复现论文级实验前应明确 suite、experiment、任务/种子、预计 trial 数与耗时、GPU / 凭据 / 服务及输出路径；按仓库 Quick Start 要求先汇报这些信息。
- 开源状态：代码仓库可访问；具体第三方依赖与数据/权重访问条件以仓库 LICENSE、NOTICE 和对应 runbook 为准。

## 对 wiki 的映射

- [ASPIRE](../../wiki/methods/aspire.md) — 三组件闭环、v1.0 新案例、基准与运行安全边界
- [NVIDIA GEAR Lab](../../wiki/entities/nvidia-gear-lab.md) — 研究团队锚点

## 相关来源

- [NVIDIA Research 项目页归档](../sites/nvidia-research-aspire.md)
- [论文摘录](../papers/aspire_nvidia_gear_2026.md)
