# NVIDIA Research — ASPIRE（GEAR Lab）

> 来源归档（ingest）

- **标题：** ASPIRE: Agentic /Skills Discovery for Robotics
- **全称（论文）：** Agentic Skill Programming through Iterative Robot Exploration
- **类型：** site（官方项目页 + PDF 白皮书；2026-10 起关联 v1.0 源码）
- **发布方：** NVIDIA GEAR Lab；合作方含 UMich、UIUC、UC Berkeley、CMU
- **原始链接：**
  - 项目页：<https://research.nvidia.com/labs/gear/aspire/>
  - PDF：<https://research.nvidia.com/labs/gear/aspire/assets/Aspire.pdf>
  - 代码：<https://github.com/NVlabs/ASPIRE>
  - 仓库归档：[`sources/repos/nvlabs-aspire.md`](../repos/nvlabs-aspire.md)
- **入库日期：** 2026-07-01；**更新核查：** 2026-10-18
- **一句话说明：** 面向 **coding agent** 的 **持续学习机器人系统**：用 **闭环执行引擎** 暴露逐原语多模态 trace、**进化搜索** 改写控制程序，并把经真机/仿真验证的修复 **蒸馏进可扩展技能库**；第 100 个任务不再像第一个任务那样从零摸索。

## 摘录要点（与论文分工）

- **核心主张：** 传统机器人编程难在感知编排、接触动力学与失败恢复；现有 coding agent 往往只有 **粗粒度 rollout 反馈**，调试后经验 **不沉淀**。ASPIRE 让 agent 像人类工程师一样 **回放 trace → 定位失败原语 → 修补程序 → 把修复写成可复用技能**。
- **三组件闭环：**
  - **闭环机器人执行引擎** — 每次 perception / planning / grasp / control 调用记录观测、输入输出、视觉证据；agent 可 **选择性检查** 相关原语日志并 **重执行验证** 修补。
  - **持续扩展技能库** — 把经 rollout 验证的修复蒸馏为 **模块化、可检索** 的 in-context 指导（非固定人工原语表）。
  - **进化搜索** — 生成多样任务序列与控制程序，在单轨迹自改进之外做 **并行调试与迭代精炼**。
- **协调器–执行器架构：** 中央 coordinator 管理共享技能库并向各任务派发 actor coding agent；actor **不交换完整聊天历史**，经验经技能库蒸馏传递。
- **实验栈：** 仿真用 **Claude Code + Claude Opus 4.6** 在 **CaP-X**（MuJoCo Playground）上写 Python 控制程序；真机跨具身迁移用 **Codex GPT-5.5** 于双臂 YAM 工位。
- **基准与主结果（论文量级）：**
  - **LIBERO-Pro** 扰动套件：较 CaP-Agent0 等基线最高 **+77 pp**；
  - **Robosuite** 双手递送：20% → **92%**；
  - **BEHAVIOR-1K** 长时程家务：最高 **+32 pp**（如 navigate-and-pick-up-radio 56% → 88%）；
  - **LIBERO-Pro Long** 零样本：在 LIBERO-90 积累技能库后达 **31%**，基线约 **4%**；
  - 仿真侧在 **150+** 任务上积累技能（bowl_on_plate、handover、can_in_trash 等）。
- **真机跨具身：** 从 Franka 仿真发现的技能作为 in-context 指导迁移到 YAM 真机（非直接权重部署）；**token 成本显著下降**（如 soda-can 总 token 约 **62M → 6.6M**，成功率 13/20 → 19/20）。
- **消融：** 执行引擎单独将 LIBERO-Pro 宏平均成功率 **14% → 62%**；进化搜索进一步提升难例。
- **局限（页面原文）：**
  - 真机部署仍依赖成功检测、安全复位、监控与标定，**非完全自主**；
  - 依赖 **冻结 frontier LLM**（Claude Opus 4.6），未验证弱模型能否维持调试环；
  - **预定义原语 API** 限制可表达行为但保证调试安全；
  - 技能库 **长期记忆管理** 未完全解决（陈旧、冗余、误导风险）；
  - 调试与进化搜索 **LLM 调用与 rollout 成本高**。

## 论文 / 代码状态

- 项目页列出论文 [arXiv:2607.00272](https://arxiv.org/abs/2607.00272)。官方代码仓库已公开：<https://github.com/NVlabs/ASPIRE>（v1.0，2026-10 README 更新）；更新前的 “coming soon” 状态已过时。
- ASPIRE 项目材料采用 Apache-2.0；仓库提示第三方材料保留各自许可。详见仓库归档与 NOTICE。
- 主基线 **CaP-Agent0**（Fu et al., 2026）为 CaP-X 框架上的 coding-agent 对照。

## 2026-10 官方发布更新

- 仓库 README 宣布 v1.0：魔方真机在线学习、长时程桌面整理仿真到真机迁移，并报告在 80 个 LIBERO-Pro Goal / Object / Spatial / Long 任务上 99% 成功率及部分 Robosuite、BEHAVIOR-1K 结果。该数据是仓库发布摘要，不应与论文表格的任务/种子协议混为一谈。
- 项目页补充了模型成本—成功率前沿、跨具身技能迁移和在线学习效率图表；详细范围以各自评测协议为准。
- 仿真 Quick Start 要求 CUDA、uv 与 gated SAM3 权重的 Hugging Face token。由于运行的是 LLM 生成 Python，官方建议隔离主机、限制网络和敏感挂载，且不授予仿真 agent 真机硬件访问。

## 对 wiki 的映射

- [ASPIRE](../../wiki/methods/aspire.md) — 三组件闭环、技能库持续学习、进化搜索与跨具身 token 迁移读点
- [ENPIRE](../../wiki/methods/enpire.md) — 同属 GEAR coding-agent 真机/仿真自改进谱系，但 ENPIRE 侧重 **策略训练范式搜索**，ASPIRE 侧重 **程序调试 + 技能库复利**
- [NVIDIA GEAR Lab](../../wiki/entities/nvidia-gear-lab.md) — 研究组锚点
- [Manipulation](../../wiki/tasks/manipulation.md) — LIBERO / Robosuite / BEHAVIOR-1K 操作基准语境
