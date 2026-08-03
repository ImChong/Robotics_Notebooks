# RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning（arXiv:2607.18060）

> 来源归档（ingest）

- **标题：** RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.18060>（v2，2026-07-28 更新；2026-07-20 首发）
  - PDF：<https://arxiv.org/pdf/2607.18060>
  - 项目页：<https://www.robo-harness.com/>（CNAME / 静态站；apex `robo-harness.com` 部分环境可能失败）
  - 代码仓：<https://github.com/markli1hoshipu/RoboHarness>
- **作者：** Jinbang Huang, Yuanzhao Hu\*, Zhiyuan Li\*, Ran Qi\*, Yixin Xiao, Zhanguang Zhang, Mark Coates, Tongtong Cao, Yingxue Zhang（\*实习期间于华为诺亚方舟实验室完成）
- **机构：** 华为诺亚方舟实验室（Huawei Noah’s Ark Lab）、英属哥伦比亚大学（UBC）、多伦多大学（University of Toronto）、麦吉尔大学（McGill）、二零一二实验室基础模型部（Department of Foundation Model, 2012 Labs）
- **入库日期：** 2026-08-03
- **一句话说明：** 将 **独立开发的异构机器人策略**（VLA / RL / TAMP 等）封装为可调用 agentic skills，用 **理解 / 记忆 / 自进化** 三类辅助技能做能力边界感知的任务分解与路由，并以 **Memory Bridge** 在策略交接处把机器人引导到下一策略的 in-distribution 状态区——无需联合重训；LIBERO-LoHo 平均成功率 **95.2%**（单策略 π₀.₅ 仅 **6.4%**），LIBERO-Plus 平均 **93.2%**，另含 500 定制任务消融与 135 次真机试验。

## 开源状态（项目页核查 2026-08-03）

- **部分开源（占位仓）：** 项目页 Code 按钮链到 [`markli1hoshipu/RoboHarness`](https://github.com/markli1hoshipu/RoboHarness)；仓内主要为 **项目页静态资源**（`docs/` HTML / 图 / 演示视频）与 README，**无可辨识的训练 / 推理 / 部署 CLI 或 `train.py` / `eval.py` 入口**。
- **底层策略权重另开源：** 仿真实验使用的 π₀.₅（[`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi)）与 OpenVLA-OFT GRPO checkpoint（Hugging Face `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90`）可公开获取；**编排 harness 本身尚无可运行实现**。
- **互指：** [`sources/sites/robo-harness-com.md`](../sites/robo-harness-com.md) · [`sources/repos/robo-harness.md`](../repos/robo-harness.md)

## 核心论文摘录（MVP）

### 1) 问题：异构策略编排 ≠ 同质技能规划

- **链接：** <https://arxiv.org/abs/2607.18060> §1–§3
- **摘录要点：** 长时程任务需要语义、闭环、几何精度与 OOD 鲁棒性的组合，但没有任何单一策略同时占优。既有规划多假设 **同质、边界清晰** 的预定义技能；异构策略在架构、I/O、执行历史上不同，能力边界 **上下文依赖**，且一策略终端状态可能落在下一策略分布外，直接 handoff 会级联失败。
- **对 wiki 的映射：**
  - [RoboHarness（论文实体）](../../wiki/entities/paper-robo-harness.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 框架：三类辅助技能 + coding agent 路由

- **链接：** arXiv §4；Figure 1
- **摘录要点：** Coding agent（论文实现为 Codex + GPT-5.5）作高层规划/路由器；**Understanding Skills**（位姿不确定性、视觉/语义上下文、状态–策略兼容、图像质量）把原始观测变成可决策的量化证据；**Memory Skills** 维护按策略组织的多模态轨迹记忆与全局执行统计；**Evolution Skills**（策略适配 / harness 精炼 / 参数调优 / metadata 更新）在持续失败时用在线证据更新。底层策略保留原生接口，无需共享动作空间或联合训练。
- **对 wiki 的映射：**
  - [RoboHarness](../../wiki/entities/paper-robo-harness.md)
  - [Harness VLA](../../wiki/entities/paper-harness-vla.md) — 同为 harness，但冻结单族 VLA + 固定原语 vs 本文异构策略族
  - [行为树 × VLA 编排](../../wiki/concepts/behavior-tree-vla-orchestration.md) — 确定性 BT 编排对照

### 3) Memory Bridge：无联合训练的策略交接

- **链接：** arXiv §4.2.2；Figure 2
- **摘录要点：** 对下一子任务检索 top-K 轨迹节点 → 沿轨迹前后扩展机器人状态 → 拟合局部进度估计 \(f_{\mathrm{score}}\) 与支持域 \(\mathcal{R}_{\mathrm{conf}}\) → 在可行运动集合内最大化「进度 − 运动代价」选 handoff 目标 → 用现成运动规划生成桥接轨迹。实验中进度估计采用 pairwise ranking（SVM）。插件式，不改底层策略权重。
- **对 wiki 的映射：**
  - [RoboHarness](../../wiki/entities/paper-robo-harness.md) — Memory Bridge 流程
  - [GaP](../../wiki/entities/paper-gap-graph-as-policy.md) — 另一类 agentic harness（计算图 staging）对照

### 4) 主结果：长时程与扰动下的系统增益

- **链接：** arXiv §5–§8；Table 1–2；项目页
- **摘录要点：**
  - **LIBERO Original：** RoboHarness **98.7%**（协调 π₀.₅ + OpenVLA-OFT，略超构成策略）。
  - **LIBERO-Plus 七类扰动平均：** **93.2%**（六类第一）；相对 π₀.₅ **85.7%** / OpenVLA-OFT **67.9%** 显著提升；robot-state 扰动上 Memory Bridge 作用突出。
  - **LIBERO-LoHo：** 进度 **97.5%** / 成功 **95.2%**；π₀.₅ 仅 **55.3% / 6.4%**；最强分层基线 H-WM-π₀.₅ 为 **84.9% / 64.8%**。
  - **消融（500 定制任务）：** 去掉 Understanding 损伤最大；去掉 Evolution 易重复路由错误；去掉 Memory Bridge 时进度仍高但全成功从 **86.0% → 60.4%**。
  - **真机 UR5e × 135 试验：** TAMP + 微调 π₀.₅ 协作拼装；Bridge 结构约 **86.7%**，再藏块扰动降至 **66.7%**。
- **对 wiki 的映射：**
  - [RoboHarness](../../wiki/entities/paper-robo-harness.md)
  - [π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md) — 底层 VLA
  - [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) — 复现入口边界

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-robo-harness.md`](../../wiki/entities/paper-robo-harness.md) — 主实体页
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — 异构编排语境下的 VLA 定位
- [`wiki/entities/paper-harness-vla.md`](../../wiki/entities/paper-harness-vla.md) — 名称相近但问题设定不同的 harness
- [`wiki/concepts/behavior-tree-vla-orchestration.md`](../../wiki/concepts/behavior-tree-vla-orchestration.md) — 确定性编排对照
- [`wiki/entities/paper-gap-graph-as-policy.md`](../../wiki/entities/paper-gap-graph-as-policy.md) — agentic harness / staging 对照
- [`wiki/overview/vla-open-source-repro-landscape-2025.md`](../../wiki/overview/vla-open-source-repro-landscape-2025.md) — 开源边界提醒
