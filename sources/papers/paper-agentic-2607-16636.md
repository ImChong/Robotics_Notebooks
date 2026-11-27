# PhyAgentOS: A Self-Evolving Operating System for Embodied Agents with Decoupled Cognitive Planning and Physical Execution

- **作者：** Yang Liu, Weixing Chen, Xinshuai Song, Tao Pu, Siwen Mo, Yongjie Bai, Zihao Chen, Qianran Sun, Liruo Zhong, Ying Shen, Liang Lin
- **机构：** X-Era Lab；HCP Lab, Sun Yat-sen University（中山大学）；Peng Cheng Laboratory（鹏城实验室）
- **版本 / 载体：** arXiv:2607.16636v1 [cs.RO]（Submitted on 18 Jul 2026，arXiv HTML 版，论文标注版本 `v0.1.6`）
- **原文：** <https://arxiv.org/abs/2607.16636>（HTML 全文：<https://arxiv.org/html/2607.16636>）
- **项目页 / 代码：** 已开源 — 官方核心仓库 <https://github.com/PhyAgentOS/PhyAgentOS-core>（MIT 协议，`v1.2.0` 发布于 2026-11-22；论文亦提及项目页 <https://phy-agent-os.net>、仓库入口 <https://github.com/PhyAgentOS/PhyAgentOS> 与配套 DSH 插件 <https://github.com/PhyAgentOS/PhyAgentOS-DSH-Plugin>；仓库详情见 [paper-agentic-2607-16636.md](../repos/paper-agentic-2607-16636.md)）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 引言、§2 相关工作、§3 系统设计、§4 五大核心机制、§5 实验与评测、§6 结论与局限、Appendix A–D、Table 1–7、Fig. 1–11 图注及参考文献 [1]–[136]），并核对官方 GitHub 仓库 `PhyAgentOS/PhyAgentOS-core` README 与目录结构快照。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与定位（§1–§2）：** 指出 VLA 模型、世界模型与 Agentic 规划器各自孤立发展，缺少统一的跨层状态表示、语义任务完成度验证（execution termination vs. semantic completion）与跨会话经验沉淀；ROS/ROS 2 提供通信中间件，但缺少任务级调度、持久化记忆、语义验证与分层安全约束。
- **双平面解耦与 State-as-a-File 协议（§3）：**
  - **Agent Layer（决策“做什么”）**：`Goal Planner`、`Goal Graph and Session Compiler`、`SkillRuntime-Target Selector`、`ContextBuilder`、`ToolRegistry`，禁止直接下发底层硬件指令，输出结构化会话契约。
  - **Protocol Layer**：以“Markdown + 嵌入式 YAML”实现跨层文件协议边界（`SESSIONS.md`、`SKILLRUNTIME.md`、`TARGETS.md`、`ENVIRONMENT.md`、`LESSONS.md`，以及 `KNOWLEDGE.md`、`TASK.md`、`LOG.md`、`RUNTIME_SIGNAL.md`；辅以 `sensors.yaml`、`perception.yaml`、`runtime_contract.yaml`、`safety.yaml`），形成零代码依赖与只追加审计轨迹。
  - **Runtime Layer（决策“怎么做”）**：四级收敛链 `WatchdogSupervisor` → `SessionRunner` → `SkillRuntime` → `Target`（`BaseRolloutTarget`）。支持 **Policy-driven（`PolicySkillRuntime`）** 与 **Agent-directed（`BuiltinSkillRuntime` / `TargetSessionHandle`）** 两条执行流。
- **五大核心机制（§4）：**
  1. **统一认知状态空间与云边协同（§4.1）**：对齐意图、能力、环境、执行与历史经验；高算力规划/记忆在云端或远端，高实时控制与安全联锁在边缘本地。
  2. **SessionVerifier 语义验收（§4.2）**：判决函数 $\mathcal{V}(G, S_0, S_T, \tau, H) \rightarrow \{\mathtt{success}, \mathtt{failure}, \mathtt{replan}\}$ 基于初始/终态观测与环境快照组成的 evidence bundle 进行裁决；`replan` 保留原始尝试不可变并派生子会话。
  3. **闭环试错与认知记忆巩固（§4.3）**：`Execute → Verify → Diagnose → Revise → Re-verify → Consolidate` 六阶段闭环；情景/语义/方法三级记忆，成功模式进 `KNOWLEDGE.md`、经二次验证有效的纠错经验进 `LESSONS.md`，无需重训神经网络即可实现跨会话自我演化。
  4. **评测即运行时编排（§4.4）**：通过 `Availability Gate` → `Session Compiler` 复用生产级调度与验证路径，区分 `First`（冻结策略首次尝试）与 `Final`（验证器触发恢复后），并生成结构化错误分类（perception/policy/adapter/execution/timeout/safety/verifier）。
  5. **纵深五层安全架构（§4.5）**：兼容性预检（`AdapterPlan` + `TargetToolManifest`）→ `ActionBridge`（坐标/单位/限幅转换）→ `SafetyGuard`（NaN/Inf、关节/空间/速度/频率/急停拦截）→ 心跳监控（Watchdog 超时与取消传播）→ 目标本地硬约束。
- **渐进式验证与关键数据（§5）：**
  - **游戏层认知验证（§5.1）**：Optimus-67（Wood `0.99`、Stone `0.96`、RedStone `0.30` 优于基线；Gold `0.06`、Armor `0.15` 仍具挑战）；StarDojo Lite-100（纯文本 `deepseek-v4-flash` 总成功率 `22.0%`、Crafting `50.0%`，超越 SPIKE 等基线）；DST-Dojo（10 轮测试平均生存天数 `1.02 → 2.10`，第 3 天存活率 `0% → 30%`）。
  - **仿真层物理回归（§5.2）**：固定策略权重，对比 `First` vs. `Final`——LIBERO（OpenVLA `74.5%→75.5%`、$\pi_0$ `92.8%→93.2%`、$\pi_{0.5}$ `97.0%→97.8%`、X-VLA `97.3%→98.6%`）；CALVIN ABC→D（5/5 全链完成率 X-VLA `74.3%→75.7%`、$\pi_0$ `38.9%→45.6%`、$\pi_{0.5}$ `85.3%→89.4%`）；RoboCasa365 `target50`（250 episodes，$\pi_{0.5}$ `17.6%→26.8%`、RLDX-1 `35.6%→42.8%`、WorldDreamer `34.0%→42.4%`）。
  - **真机与跨本体部署（§5.3，Table 7）**：列出 19 个仿真/真机平台（如 Agilex PIPER、Franka FR3/Panda、Fourier GR-3、Unitree Go2/G1/R1、Huibo Astra-Pro、Lekiwi、SO100/SO101、Stella Gaia Hand 20、ViperX300、XLerobot、Zerith_H1_PRO 等，正文亦提及 Dobot Nova 2），重点验证跨本体零样本切换（Franka/PIPER/Dobot）与安全机制（预检拒绝、SafetyGuard 截断、急停响应），但论文正文与 Table 7 未给出大规模真机成功率统计或毫秒级急停数值。
- **局限（§6.2–§6.3）：** 轮询式文件协议存在与轮询间隔成正比的延迟；记忆文件长期增长缺乏理论压缩上界；`Goal Graph / Session Compiler` 的多机 DAG 调度在论文初版（`v0.1.6`）中尚未完全实现；真机实验侧重安全机制验证而非大规模任务成功率统计。
- **对 wiki 的映射：** [PhyAgentOS 实体页](../../wiki/entities/paper-agentic-2607-16636.md)
