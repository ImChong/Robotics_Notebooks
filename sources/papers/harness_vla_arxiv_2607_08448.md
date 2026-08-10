# Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents（arXiv:2607.08448）

> 来源归档（ingest）

- **标题：** Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.08448>（**v3**，2026-07-15 更新；2026-07-09 首发）
  - PDF：<https://arxiv.org/pdf/2607.08448v3>
  - 项目页：<https://harnessvla.github.io/>
  - 代码：<https://github.com/RLinf/RPent>
- **作者：** Yixian Zhang*, Huanming Zhang*, Feng Gao, Xiao Li, Zhihao Liu, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, Zhengru Fang, Yi Nie, Changxu Wei, Yu Wang, Wenbo Ding, Chao Yu†（*Equal contribution；†Corresponding）
- **机构：** 清华大学（Tsinghua）、跨步智能（Striding AI）、普渡大学（Purdue）、中国科学院自动化研究所（CASIA）、无问芯穹（Infinigence AI）、香港科技大学（HKUST）、中关村人工智能研究院（ZGCA）
- **入库日期：** 2026-07-22
- **复核日期：** 2026-08-10（用户指定 PDF v3 + 项目页；对照 arXiv:2607.08448v3 与 harnessvla.github.io / RLinf/RPent）
- **一句话说明：** **冻结 VLA** 作可重试接触原语 `vla_act`，与固定解析原语库（定位/过渡/搬运/导航/释放）由 **记忆增强 agentic planner** 编排；参考种子探索写入 Task Specific Memory + Global Memory，部署时对扰动场景 **重绑定** 而非重放轨迹；LIBERO-Pro **82.4%**（相对最强相关基线 **+38.6 pp**）、RoboCasa365 **55.4%**（相对 RLDX-1 **+25.4 pp**）、RoboTwin C2R **58.4%**。

## 开源状态（项目页核查 2026-08-10）

- **已开源：** 项目页 Code 仍链到官方仓 [`RLinf/RPent`](https://github.com/RLinf/RPent)（Recursive Physical Agent / Agentic Infrastructure）；README 提供 `pip install -e ".[full]"`、`rpent` CLI（LIBERO / LIBERO-Pro）、RoboCasa 脚本与 HF π₀.₅ checkpoint 入口；文档站 <https://rpent.readthedocs.io/en/latest/>（含[中文文档](https://rpent.readthedocs.io/zh-cn/latest/)）；HF 组织 <https://huggingface.co/RLinf>。
- **部分能力：** Feature Matrix（2026-08-10 README）标 Pi0.5 + LIBERO-PRO ✅、Claude Code / Codex / Custom Planner ✅；RLDX-1、RoboCasa、真机 Franka·SO-101、WAM/DreamZero 等条目尚未打勾——复现以 README 当前矩阵为准。
- **互指：** [`sources/sites/harnessvla-github-io.md`](../sites/harnessvla-github-io.md) · [`sources/repos/rpent.md`](../repos/rpent.md)

## 核心论文摘录（MVP）

### 1) 问题：端到端 VLA 与纯解析原语各自错位

- **链接：** <https://arxiv.org/abs/2607.08448> §1
- **摘录要点：** 预训练 VLA 擅长局部分布内接触控制，但在语义重定向、目标重绑定、布局扰动与不稳定接触下易失效；LLM coding agent / harness 擅长组合推理，但纯解析原语难以覆盖不规则抓取、受限放置与铰接交互。需要在 **不微调 VLA、不膨胀技能库** 的前提下扩展其可用分布。
- **对 wiki 的映射：**
  - [Harness VLA（论文实体）](../../wiki/entities/paper-harness-vla.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 框架：固定原语库 + `vla_act` 接触专家

- **链接：** arXiv §2.1–2.3；Table 1
- **摘录要点：** 规划器 Π 只发出 JSON 原语调用；库含 `move_to` / `move_pose` / `rotate_*` / `set_gripper` / `release` 与 **`vla_act`**（短 burst 冻结 VLA）；RoboCasa365 另加 `navigate_to` / `move_base`。部署期禁止发明新原语；reset 仅用于 bootstrapping。
- **对 wiki 的映射：**
  - [Harness VLA](../../wiki/entities/paper-harness-vla.md) — 原语表与流程。
  - [行为树 × VLA 编排](../../wiki/concepts/behavior-tree-vla-orchestration.md) — 编排层对照。

### 3) 两阶段记忆：Bootstrapping → Deployment

- **链接：** arXiv §2.2
- **摘录要点：** 参考种子上搜索过渡顺序、预接触位姿、`vla_act` 时机、停止谓词与恢复策略；成功轨迹参数化为 **Task Specific Memory**（JSONL，坐标→感知查询符号化）；可复用成功规则与失败模型写入 **Global Memory**。部署禁用 reset、预算收紧，对当前 RGB-D **重绑定** 空间参数。
- **对 wiki 的映射：**
  - [Harness VLA](../../wiki/entities/paper-harness-vla.md)
  - [ASPIRE](../../wiki/methods/aspire.md) — 扩展技能库 vs 固定原语对照。

### 4) 主结果与机制分析

- **链接：** arXiv §3；附录 Table 18–19；项目页 Results
- **摘录要点：**
  - **标准 LIBERO：** Harness VLA (CC) **96.0%**，与冻结 π_RLinf **95.3%** 相当（保持分布内竞争力）。
  - **LIBERO-Pro：** CC **82.4%** / Codex **72.1%**；相对 RATS 报道总体 **+38.6 pp**；相对同 backbone 冻结 π_RLinf **50.0%** 显著提升。
  - **RoboCasa365：** Codex 任务加权总体 **55.4%** vs RLDX-1 **30.0%**（**+25.4 pp**）；CC **48.6%**。
  - **RoboTwin C2R：** CC **58.4%** vs 同冻结 LingBot-VLA 直接 rollout **50.4%**；相对 π₀.₅ **47.9%**。
  - 机制：规划器语义重绑定；稀疏可重试 `vla_act`；解析原语承担非接触结构。
  - **原语用量（Table 18/19）：** LIBERO 上 `VLA_ACT` 仅 **15.8%** 调用、解析原语 **84.2%**；RoboCasa365 `VLA_ACT` **35.3%**；RoboTwin C2R **47.4%**——冻结 VLA 不作单体端到端控制器，而是接触原语。
- **对 wiki 的映射：**
  - [Harness VLA](../../wiki/entities/paper-harness-vla.md)
  - [DreamSteer](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md) — 另一类冻结 VLA 部署 steering。
  - [LingBot-VLA](../../wiki/entities/lingbot-vla.md) / [RLDX-1](../../wiki/entities/rldx-1.md) — 评测用冻结后端。

### 5) 局限与未来（§5，v3）

- **链接：** arXiv §5 Conclusion and Limitations
- **摘录要点：** 规划器↔低层 VLA 仍为开环反馈；缺环境奖励 / 人类偏好的联合微调（指向样本高效 RL，如 GRPO）；稠密长程场景缺细粒度 caption 约束结构推理；未来可与 [ASPIRE](../../wiki/methods/aspire.md) 式自动技能发现互补——在固定可审计词汇上按需接纳新技能。
- **对 wiki 的映射：**
  - [Harness VLA](../../wiki/entities/paper-harness-vla.md)
  - [ASPIRE](../../wiki/methods/aspire.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-harness-vla.md`](../../wiki/entities/paper-harness-vla.md) — 主实体页
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — 冻结 VLA 作为原语的部署路线
- [`wiki/concepts/behavior-tree-vla-orchestration.md`](../../wiki/concepts/behavior-tree-vla-orchestration.md) — 编排对照
- [`wiki/methods/aspire.md`](../../wiki/methods/aspire.md) — 技能库扩张对照
- [`wiki/overview/vla-open-source-repro-landscape-2025.md`](../../wiki/overview/vla-open-source-repro-landscape-2025.md) — RPent 复现入口
