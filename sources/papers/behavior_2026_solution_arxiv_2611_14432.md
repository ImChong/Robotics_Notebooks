# One Policy for 100 Household Tasks｜BEHAVIOR Challenge 2026 方案

- **英文标题：** One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge
- **中文标题：** 单策略完成 100 项家务任务：2026 BEHAVIOR 挑战赛方案
- **论文：** https://arxiv.org/abs/2611.14432
- **全文：** https://arxiv.org/html/2611.14432v1
- **版本：** arXiv v1，2026-11-10
- **作者：** Ilia Larchenko、Gleb Zarin、Atindra Nair、Keerti Manney、Vitaly Kleban、Mirko Kiefer、Anjana Yodaiken
- **官网：** https://robot-learning-collective.github.io/behavior-1k-2026/
- **源码：** https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026
- **权重：** https://huggingface.co/IliaLarchenko/BEHAVIOR2026VLA
- **数据集：** https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
- **入库日期：** 2026-11-12
- **核对范围：** 官方论文 v1 全文的架构、数据准备、训练、推理、实验、赛事实绩、局限；结合作者官方 README、agent/README.md、scripts/train.py、scripts/serve_b1k.py、scripts/run_parallel_eval.py、scripts/eval_worker.py、src/b1k/models/pi_behavior.py、src/b1k/policies/policy_config.py 与 src/b1k/shared/eval_b1k_wrapper.py 核对可运行入口。未训练或跑仿真。
- **通读：** 已完成

## 核心模型与数据

- 以 **π₀.₅** 为起点，单个约 **2.9B 参数 VLA**：PaliGemma 视觉语言主干 + Gemma flow-matching 动作专家，在 OmniGibson/Isaac Sim 中为 Galaxea R1Pro 执行 100 类长时家务任务。
- **100 项任务、20,000 条示范**，LeRobot v3 数据集；不同任务共享同一策略权重，而非每任务训练一套策略。
- 使用学习的**任务嵌入与语义阶段嵌入**（总计 413 个阶段），状态离散 token 与连续状态特征并存；动作专家跨层 KV 混合、相关流噪声与多次采样的训练目标。
- 数据观测时间与评测器对齐，训练时去掉完成后的 idle tail 作为动作块锚点；训练包含不同相机及状态扰动。
- 训练配方约 **200k 步**；作者明确指出发布配方**并非已发布权重的完全精确重放**。

## 推理与评测

- 一次输出 **30 个动作**，典型流采样 **20 步**；每 13 个环境步执行前 26 个预测动作，保留 4 个未来动作用于滚动软 inpainting；配合阶段投票与块边界修正。
- 作者本地对**仅 VLA**的 100 项任务×10 回合评测：Q-score **0.392**；50 个旧任务 **0.371**、50 个新任务 **0.412**。
- 最终公开排行榜**第 9 名、44.6% Q-score**为**混合系统**：基策略加上针对 14 个弱项任务的 **GPT-6 Astra 工具型 Agent**。不可把 44.6% 当成单模型 VLA 的成绩。官方叙述的纯策略公开榜成绩为约 38.9%，与本地 39.2% 为不同评测口径。
- Agent 借助事先在各任务上进行的仿真练习、笔记与冻结会话进行 tool-driven 控制（observe、act、xyz）；这些**练习会话未公开**，源码不能单独完整复现 14 任务的 agent 路由实验。

## 代码可用性与限制

- 仓库公开训练、checkpoint 加载、WebSocket policy server、OmniGibson worker 评测和部分 Agent 架构；仓库公开**不等于所有竞赛运行工件与练习会话完整公开**。
- 官方 README 可运行示例需要配置 Isaac Sim、OmniGibson、openpi 子模块、checkpoint、训练示范及运行环境。
- 代码核对入口：`scripts/train.py::main/train_step`→`b1k.training.data_loader`、`PiBehavior.compute_loss`；`scripts/run_parallel_eval.py`→`eval_worker.py`→`SlotEvaluator`→WebSocket `MultiSessionPolicyServer`→`B1KPolicyWrapper`→`PiBehavior.sample_actions`。
- 作者结果仅由论文与 README 支持，未在当前环境独立复现。

## 对 wiki 的映射

- [BEHAVIOR 2026 实体页](../../wiki/entities/paper-behavior-2026-one-policy-100-tasks.md)
- [VLA 方法](../../wiki/methods/vla.md)
- [动作分块](../../wiki/methods/action-chunking.md)

## 参考来源

- [官方论文全文](https://arxiv.org/html/2611.14432v1)
- [官方论文记录](https://arxiv.org/abs/2611.14432)
- [官方仓库 README](https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026)
- [官方技术博客](https://robot-learning-collective.github.io/behavior-1k-2026/)
