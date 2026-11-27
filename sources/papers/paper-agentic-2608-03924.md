# ETA: A New Agentic Paradigm for Embodied Tasks

- **作者：** Yitong Chen, Zezheng Huai, Sixian Li, Yubang Wang, Haozhe Zhang, Yifei Zhang, Hechang Chen, Jingjing Gong, Yu-Gang Jiang, Xipeng Qiu
- **机构：** 上海创智学院（Shanghai Innovation Institute）、复旦大学（Fudan University）、吉林大学（Jilin University）、南京大学（Nanjing University）、浙江大学（Zhejiang University）
- **版本 / 载体：** arXiv:2608.03924v1 [cs.RO]（2026 年 8 月 4 日提交，arXiv HTML 全文版本）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2608.03924) · [arXiv 摘要页](https://arxiv.org/abs/2608.03924)
- **项目页 / 代码：** 已开源 — [OpenMOSS/OpenETA](https://github.com/OpenMOSS/OpenETA)（项目主页 [openmoss.ai/OpenETA](https://openmoss.ai/OpenETA/)，Apache-2.0 协议；仓库包含完整的 `agent/`、`adapter/`、`sim/`、`real/`、`scripts/` 等可运行代码与测试；其中真机后端当前冻结于接口集成层级，未包含硬件批次统计结果）
- **核对范围：** 已通读 arXiv HTML 全文（Section 1–9 全部正文，含表 1–6 与图 1–7，以及 Appendix A–G 全部附录，含表 7–31 与图 8–12），并交叉核对 `OpenMOSS/OpenETA` GitHub 主页与 README 信息。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与范式定位：** 论文指出当前端到端 VLA（Vision–Language–Action）和 WAM（World Action Model）在跨任务/环境泛化上严重依赖机器人数据覆盖度，且长程执行过程难以控制与审计。作者提出 **Embodied Task Agent (ETA)** 范式及开源实现 **OpenETA**，将具身系统组织为 **Agent（Planner + Working Memory + Skill/Soul）– Interface（Host 契约与执行门控）– World（通过 MCP 暴露的仿真器或真机）** 三层闭环架构。
- **运行时不变量与信任边界：** 每次只允许执行一个改变物理世界状态（`world_mutating`）的原子动作（AtomAction），执行后必须先获取最新观测 $o_{t+1}$ 才能继续状态相关决策（连续 3 次观测刷新失败则安全停机）。Agent 仅提交 `tool_call` 或 `response`（`openeta.agent_command.v1`），无直接修改世界或伪造奖励的权限；只有宿主校验通过的 `openeta.environment_receipt.v1` 中的官方奖励/终止信号才能确立任务成功。
- **OpenETA 系统与两种配置：**
  - **Full OpenETA：** 默认注册 **44 个 Tool**（`main@135a7edc7e60` 快照下含 16 个 `read_only`、12 个 `planning`、9 个 `bookkeeping`、7 个 `world_mutating`），整合 SAM 3、UniDepth V2、MolmoPoint、AnyGrasp、Contact-GraspNet、GraspGenX、AnyPlace、cuRobo 等模块；Skill 仅为可审核文本经验，不能绕过 AtomAction 隐式执行。
  - **OpenETA for Codex：** 面向多模态推理模型的极简三工具接口，仅暴露 `observe`（支持 512×512 第三视角/腕部/XYZ 正交视图）、`mark_point`（单视图射线求交或正交多视图双点定 3D 坐标）与 `move_to`（以 approach 与 jaw 双向量指定夹爪朝向、闭爪前渲染预览并返回夹爪开度）。
- **核心实验数据（均来自论文冻结快照）：**
  - **Full OpenETA 固定矩阵基线（LIBERO 4 个标准 suite × 10 任务 × 10 种子 = 400 episodes，`gpt-5.6-luna` medium）：** 整体成功 **56/400 (14.0%)**（Spatial 8/100、Object 26/100、Goal 21/100、Long/LIBERO-10 1/100；18/40 任务至少成功 1 次；其中 3 个 LIBERO-10 单元因约 1800 s 仿真句柄 TTL 过期 `simulator_unknown_handle` 计为基础设施污染并保留在 400 分母中，排除后诊断值为 56/397 = 14.11%）。344 次失败中 `episode_timeout` 占 215 次（62.5%）、`unattended_ask_human` 占 66 次（19.2%）、`max_turns` 占 35 次（10.2%）。
  - **OpenETA for Codex（130 个 LIBERO 任务，seeds 0–4，未使用 VLA/任务专用策略）：** Pass@1 分别为 GPT-5.6 Luna 21/130、Terra 58/130、Sol 92/130；Pass@5 分别为 Luna **62/130 (47.7%)**、Terra **83/130 (63.8%)**、Sol **117/130 (90.0%)**（其中 Sol 在 Spatial 10/10、Object 10/10、Goal 8/10、LIBERO-10 10/10、LIBERO-90 79/90）。
  - **受约束智能体自我演化（LIBERO Spatial）：** 提出四重晋升门控 $\mathrm{Promote}(u) = D(u) \land Q(u) \land R(u) \land H(u)$（契约检查、隔离审查、同种子回放、配对留出验证）。在 6 个审计批次（Task-local Skill 0/30 vs 0/30；Exact-task playbook 1/30 vs 4/30，McNemar $p=0.375$；Stage-local v1 0/3 vs 0/3；Contrastive v2 0/1 vs 0/1）中 **没有任何候选经验通过晋升门控**，论文将其定性为“控制与防回归机制成立，当前文本经验生成尚无稳定成功率提升”。
  - **真机部署（UR5e + Robotiq + RealSense D435i/L515）：** 论文将其正式证据等级冻结在 **interface integration（接口集成）**，给出 1 条完整 sponge-to-tray 与 1 条部分 bell-pepper 定性演示视频，并记录了真实深度缺失及侧向抓取触发加速度保护停机两类 Sim2Real 瓶颈。
- **对 wiki 的映射：** [ETA: A New Agentic Paradigm for Embodied Tasks](../../wiki/entities/paper-agentic-2608-03924.md)
