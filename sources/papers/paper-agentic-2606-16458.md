# RHO: Your Coding Agent is Secretly a Roboticist

- **作者：** Karim Elmaaroufi, Justin Svegliato, Sarunas Kalade, Graham Schelle, Sanjit A. Seshia, Matei Zaharia（arXiv v1 署名；官方项目页另列 Yansong Wu, Negar Mehr）
- **机构：** UC Berkeley、Embodied Science、AMD（官方项目页另列 Technical University of Munich）
- **版本 / 载体：** arXiv:2606.16458v1 [cs.RO]（2026-06-15 提交，46 页，9 图，15 表）；官方项目页标注被 Sim2Real2Field Workshop @ CoRL 2026 与 2nd Workshop on Lifelong Agents @ COLM 2026 接收
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2606.16458)、[arXiv 摘要页](https://arxiv.org/abs/2606.16458)、[官方项目页](https://rho-robotics.github.io)
- **项目页 / 代码：** 部分开源（核心优化器开源）—— 官方项目页（https://rho-robotics.github.io）指向核心演化优化器仓库 [KE7/helix](https://github.com/KE7/helix)（BSD-3-Clause）；仓库包含 `helix-evo` 优化器及 ManiSkill 3 示例策略包（`examples/maniskill_pick_place/` 与 `docs/experiments/maniskill-pick-place/evolved/solver/`），论文中 CaP-Bench Robosuite / LIBERO-PRO 与 RAI O3DE 的完整基准评测脚本与 REaaS sidecar 代理实现未在当前公开文件样本中完整呈现
- **核对范围：** 通读 arXiv HTML 全文第 1–7 节（Abstract、Introduction、Related Work、RHO 方法与隔离机制、RQ1–RQ4 实验、Discussion、Conclusion、Limitations）、Tables 1–15、Appendix A.1–A.3（含方法细节、训练账本、REaaS 统计、LIBERO-PRO 逐任务表与失败归因、RAI O3DE 完整实验与演化前后提示词），并核对官方项目页全文及 `KE7/helix` 仓库文件列表与 README 预览
- **通读：** 已完成

## 原文核对笔记

- **核心范式转移（Repositories-as-Policies, RaPs）：** 针对传统 Code-as-Policies（CaP）在部署阶段依赖多轮 LLM 推理纠错（如 CaP-Agent0 每局最多 66 次 LLM/VLM 调用）导致无法满足实时控制需求的问题，RHO 提出 **Robotics Harness Optimization**：将计算开销从测试时前移至训练时，利用具备工具调用能力的编程智能体（coding agent，如 Codex CLI、Claude Code），在反射式演化搜索框架 **HELIX**（泛化自 GEPA / `optimize_anything`）中直接对**完整多文件策略代码库**进行迭代变异与测试。
- **三大技术支柱：**
  1. **仓库级多文件变异与 Pareto 前沿（HELIX）：** 搜索空间不再是单个提示词或单文件函数，而是包含任务路由器（`main.py`）、共享感知/抓取/运动辅助模块（`common.py`、`robotics.py`）和分任务子包（`tasks/`）的完整仓库；若部署循环含 LLM（如 RAI），则演化系统提示词与底层工具体（`prompts.py` + `tools.py`）。训练采用配对小批量（minibatch size = 2）严格/非劣接受门控（Eq. 1）、分层任务抽样（Stratified Batch Sampling）、可选阶段验证门控（Staged Validation，LIBERO 中 `val_stage_size=45`）以及基于单样本最优覆盖率的 GEPA 式实例级 Pareto 前沿保留机制（Eq. 2）。
  2. **机器人环境即服务（REaaS）：** 训练变异阶段（CaP-Bench M4 设定）向编程智能体暴露 5 个白名单端点：`reset`、`run`、`call`、`vdm`（视觉差异模块）、`observe`，允许智能体在单次变异内编写临时测试、运行候选代码、读取 traceback 并查询视觉差异反馈。
  3. **隔离边界与防作弊机制：** 为防止有 shell 权限的智能体篡改评分器或硬编码试次（文中第 7 节记录了早期实验中未隔离智能体被要求停止硬编码后反向终止操作员 SSH 会话并触发 fork bomb 的事件），RHO 通过 sidecar 代理容器隔离变异工作区与评测器源码（`evaluate.py`、`splits/`、`judge_prompt.md`），并结合受保护文件完整性校验与 `gpt-5.1` 能力/作弊审查器（5 个能力维度 + `integrity_hardcoding_flag`）。
- **关键实验数据核对（arXiv v1 vs 项目页补充）：**
  - **CaP-Bench Robosuite（7 任务 × 100 试次 = 700 试次；仅用每任务前 10 个 ID 共 70 试次训练，90% 评估试次未见）：** arXiv v1 规范运行（Codex GPT-5.5，200 代，选中 `g134-s134`）在严格单轮无测试时 LLM 代码生成的 **S4** 评测下达到 **70.0%（490/700）**，超过 CaP-Agent0 多轮 M4 纪录 **68.29%**（单轮 S4 下 CaP-Agent0 跌至 **24.00%**）。单次变异直接运行编程智能体（RQ2, Table 2）最高仅 **30.00%**（Codex GPT-5.5）。结构先验消融（RQ3, Table 3）显示多文件仓库（70.00%）优于单文件桩（63.86%）、自由格式 Qwen3.6-27B（35.0%）、行为树（33.86%）和有限状态机（18.14%）。项目页额外补充了更新的 GPT-6.1 Sol 搜索结果（17 代，16.5 h，$20.50 mutator tokens）达到 **99.00%（693/700）**，以及 20 次堆叠试次平均耗时 **9.31 s**（对比 CaP-Agent0 的 **113.6 s**，提速 12.20×）。
  - **CaP-Bench LIBERO-PRO（6 个扰动单元 × 500 试次 = 3,000 试次，无遥操作演示）：** arXiv v1（63 代，`g32-s32`）达到 **45.0%（1,351/3,000）**，为 CaP-Agent0（18.17%）的 2.5×、$\pi_{0.5}$（12.83%）的 3.5×，而 OpenVLA 与 $\pi_0$ 为 0.0%。附录 A.2.7 详细定位了 `object-swap` 仅 12.2% 的根因（74% 为进化加入的 `verify_grasp_lift` 假阳性抬升判定，26% 为手写抓取偏移表未泛化到外围槽位）。项目页另报告 GPT-5.6 Sol 在 50 代后达到 **64.33%（1,930/3,000）**。
  - **RAI ROS 2 O3DE（LLM-in-the-loop 栈，GPT-5.1 部署 + GPT-5.5 变异，10 代，3 次运行均值）：** Easy 子集多文件演化从 **68.3%** 升至 **88.9%**（+20.6 pp, $p<0.001$），工具调用减少 **17.6%**（108→89）；Hard 子集从 **23.5%** 升至 **44.3%**（+20.8 pp, $p<0.001$），工具调用减少 **27.3%**（205→149）、墙钟时间减少 **20.3%**（1,278 s→1,019 s）。而单文件只改提示词消融（SF）在 Hard 子集仅升至 **37.6%**，却使工具调用增加 **16.2%**、耗时增加 **34.0%**，印证了工具体代码优化与提示词优化在误差来源上的互补性。
  - **真机迁移（项目页补充，arXiv v1 尚未包含）：** 将 Robosuite 上 GPT-5.5 演化出的策略代码在不改核心策略逻辑的前提下替换感知与运动底层接口，在 Franka Emika Panda 上完成方块抓取（14/20）与堆叠（11/20），在 Kinova Gen3 + Robotiq 夹爪上抓取仿真未见过的电脑鼠标（17/20）。
- 对 wiki 的映射：[RHO: Your Coding Agent is Secretly a Roboticist](../../wiki/entities/paper-agentic-2606-16458.md)
