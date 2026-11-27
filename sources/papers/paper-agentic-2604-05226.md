# RoboPlayground: Democratizing Robotic Evaluation through Structured Physical Domains

- **作者：** Yi Ru Wang*, Carter Ung*, Evan Gubarev, Christopher Tan, Siddhartha Srinivasa†, Dieter Fox†（*共同一作，†共同指导）
- **机构：** University of Washington；Allen Institute for Artificial Intelligence（AI2）
- **版本 / 载体：** arXiv:2604.05226v1 `[cs.RO]`（2026-04-06 提交）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2604.05226)（[arXiv 摘要页](https://arxiv.org/abs/2604.05226)）
- **项目页 / 代码：** [项目页](https://roboplayground.github.io)（**宣称将开源或未列链接**：截至 2026-11-27，项目页写明 "Everything is open-source" 并展示 `roboplayground.ai/create` 交互界面及示例生成代码片段，但论文正文与项目页均未提供可下载的官方 GitHub 仓库链接）
- **核对范围：** 通读 arXiv HTML 全文（I Introduction、II Related Works、III Methods、IV Results、V Ablative Studies、VI Discussions）、Appendix A–C（A-1 至 A-9 系统模块与参数、B-1 至 B-3 基线/用户研究/策略训练细节、C-1 至 C-2 消融指标与测试集）、Table I–XVI、Fig. 1–27，以及项目页 `roboplayground.github.io` 文本快照（2026-11-27）。
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 传统机器人操作评测（如 RLBench、LIBERO、RoboCasa、Simpler、ManiSkill 等）依赖少数专家硬编码固定任务实例与成功条件，外部用户无法低门槛地修改任务意图、约束或成功标准；而已有 LLM 任务生成工作（GenSim、Gen2Sim、RoboGen 等）主要服务于训练数据扩增，缺乏面向评测的可复现性与版本谱系控制。
- **形式化定义：** 单个操作任务定义为 $T = (\mathcal{A}, \rho_0, G, \ell, \mathcal{V})$，其中 $\mathcal{A}$ 为资产集合（`ColoredCube`、`SemanticCube`、`Table`、目标贴片等），$\rho_0$ 为初始状态重置分布，$G: \mathcal{S} \to \{0,1\}$ 为状态成功谓词（`_success()`），$\ell$ 为规范自然语言指令，$\mathcal{V}$ 为同义改写集合。论文强调语言层面的逻辑等价不等于任务实现的等价，容差、重置分布与成功判定时机均会改变评测结果。
- **系统流水线：** 由四大模块组成（均在统一 MuJoCo 物理域中实例化，桌面高度 $z=0.95\text{ m}$，工作空间 $x\in[0.40,0.70]\text{ m}, y\in[-0.25,0.25]\text{ m}$）：
  1. **Task Orchestration**：将自然语言填充为固定 `TaskSchema`，经 `task` 与 `feasibility` 检查（资产上限如 `SemanticCube` 最多 26 字母 / 10 数字、空间/排序/可见性冲突预检与修复建议）。
  2. **Code Generation**：检索 `api_review`、`common_errors` 与 `code_references`（10 个人工参考任务中选 2–3 个），先生成中间规范再编译为继承 `RoboEvalEnv` 的任务类（`_initialize_env`、`_on_reset`、`_success`）。
  3. **Validation & Iterative Repair**：依次经过 Basic Validation（AST 静态检查、隔离编译、实例化与 zero-action 冒烟测试）与 Success Check（支撑关系有向图环路/完整度/传递冗余分析、工作空间边界检查、目标态 50 步静置接触结算 + 50 步稳定性校验：漂移 $\le 1\text{ cm}$、垂直位移 $\le 2\text{ cm}$，以及 `SemanticCube` 5 射线可见性与法向 $\ge 0.97$ / 字形朝向 $\ge 0.94$ 可读性校验）；失败后由 Agent Orchestrator 路由至 6 类专项修复 agent（最多 5 轮、单 agent 最多 3 次，附负例防震荡）。
  4. **Context Steering**：通过 `SteeringRouter` 将多轮修改归类为 `Tweak` / `Extend` / `Modify` / `Pivot` / `Fresh`，结合 `TaskSnapshot`（含 `version_id`、`assets_used`、`goal_summary`、SHA-256 `code_hash`）与 `SteeringMeta` 支持跨资产兼容回溯与受控演化。
- **关键实验发现：**
  - **可用性（$N=26$ 被试内研究，Table I & X）**：SUS 为 $83.4 \pm 6.9$（高于 Cursor $68.8 \pm 7.8$、GenSim $52.5 \pm 9.3$），NASA-TLX 负载 $18.6 \pm 7.7$（显著低于 Cursor $36.7 \pm 10.4$ 与 GenSim $41.8 \pm 9.0$），偏好率 69%（18/26）。
  - **规范质量（Table IV）**：Compile / Smoke Test / Human Verification 均达 $100\pm 0\%$，LLM Intent Alignment 为 $73.5\pm 11.1\%$（Cursor $61.1\pm 14.0\%$，GenSim $38.8\pm 14.0\%$）。
  - **策略诊断（Table II）**：在 CuTAMP 演示微调的 6 个策略（4 个 Qwen3-VL-4B-Instruct 底座 StarVLA 变体：`Adapter`、`GR00T`、`Dual`、`Qwen-OFT`；2 个 `Pi-0.5` 变体：全微调与 LoRA）上测试 10 个分布内任务和 12 个受控扰动泛化任务。分布内简单放置最高达 96.0%（GR00T），但长程/组合训练任务普遍低于 22%；分布外泛化呈明显不对称——视觉扰动（V）迁移最好（GR00T 达 86%–90%），语义扰动（S）中等，行为扰动（B）则使全部模型几近崩溃（多任务 0%–2%）；`Pi-0.5 (LoRA)` 全线显著落后。
  - **人群扩展性（Fig. 4）**：单用户累计新增任务多样性快速饱和，而跨贡献者聚合多样性随人数单调上升。
  - **消融（Table III）**：关闭 Validation 时 Task Success 从 100.0% 跌至 12.0%；关闭版本历史追踪仅做意图路由时 Human-Verified 从 100.0% 跌至 76.9%。
- **对 wiki 的映射：** [RoboPlayground 实体页](../../wiki/entities/paper-agentic-2604-05226.md)
