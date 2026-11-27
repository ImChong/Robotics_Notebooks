# ManipEvalAgent: Promptable and Efficient Evaluation Framework for Robotic Manipulation Policies

- **作者：** Yiteng Chen\*, Huiping Zhuang\*, Wenbo Li\*, Shiyi Wang, Xiangyu Zhao, Qingyao Wu†（* 共同一作，† 通讯作者）
- **机构：** 华南理工大学（South China University of Technology, SCUT）、复旦大学（Fudan University）
- **版本 / 载体：** ICLR 2026 会议论文（Published as a conference paper at ICLR 2026，共 25 页含附录）
- **原文：** [ICLR 2026 摘要页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8833c8aa10542d24d693bbaf6a4598f5-Abstract-Conference.html) · [ICLR 2026 会议全文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/8833c8aa10542d24d693bbaf6a4598f5-Paper-Conference.pdf)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查 ICLR 2026 摘要页及会议全文，均未列出独立项目页或官方 GitHub 代码仓库；实验所用 RoboTwin 2.0、LIBERO 与 ACT / DP / DP3 / RDT-1B / $\pi_0$ 为第三方既有开源基准与被测策略）
- **核对范围：** 全文通读第 1–5 节正文（含 Figure 1–6、Table 1–3）及 Appendix A.1–A.7（含实验超参、公平性计时、误差统计口径、Table 4–10、Open User Query 示例与分类体系、LightRAG 与 LIBERO 适配细节、局限性讨论）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 传统机器人操作仿真基准（如 RoboTwin 2.0、LIBERO、RLBench、CALVIN）依赖固定任务集与大规模采样，耗时高、不响应用户开放式需求，且仅输出单一成功率分数，缺乏可解释诊断。
- **三阶段多轮闭环（§3）：**
  1. **Proposal Stage（Plan Agent）：** 读取系统提示（仿真器能力约束 $\Omega,\Gamma$ 与被测策略元信息）与用户自然语言查询 $\Psi$，结合历史评测数据库检索的相似拆解，将查询动态拆分为子维度 $a_j \in A$，根据中间观测结果 $Y_{1:t}$ 迭代调整下一轮探索方向并输出诊断总结。
  2. **Generation Stage（TaskGen Agents & ToolGen Agents）：**
     - `TaskGen` 遵循复用优先原则，基于 Task Library $\mathcal{K}_{\text{task}}$、Asset List $\mathcal{K}_{\text{asset}}$ 与文档库 $\mathcal{K}_{\text{doc}}$ 生成可运行的任务 Python 文件（含场景构建与 `check_success`），并叠加三项增强：基于 LightRAG（本地 `Qwen2.5-7B-Instruct`）的 RAG、首帧渲染视觉自检（Visual Self-Reflection）与面向智能体的结构化文档 `README.Agent`。
     - `ToolGen` 遵循检索优先原则维护可扩展工具库 $\mathcal{T}$，按需复用或生成两类评测工具：基于仿真器 API 的规则函数 $r(\zeta)$ 与基于 VLM 关键帧问答的 VQA 工具 $q(I_{0:T}, Q)$。
  3. **Execution Stage：** 在仿真引擎中对每个构造任务默认执行 $M_j = 5$ 次 rollout（可配置），聚合规则工具标量信号与 VQA 视觉分析结果后回传给 Plan Agent。
- **核心量化结论（§4 & Appendix A.2）：**
  - **耗时与样本量：** 单任务设定下（Table 1），5 种策略总耗时降至 42–63 分钟、15,336–16,927 样本（RoboTwin 2.0 需 159–210 分钟 / 51,087–56,592 样本，LIBERO 需 103–132 分钟 / 26,732–29,546 样本）；多任务设定下（Table 4），RDT 从 11,102 / 4,909 分钟降至 97 分钟，$\pi_0$ 从 7,900 / 4,330 分钟降至 68 分钟。
  - **结论一致性与排序相关性：** 10 次独立运行中结论多数落在基准结果 1–3 个标准差内（Table 2, 5）；5 策略在 7 个任务簇上的 Spearman 排序相关系数 $\rho$ 在 10/20/50 rollouts 下分别稳定在 0.76–0.86、0.73–0.91、0.79–0.85（Table 9）。
  - **模块性能与系统误差：** 子维度拆解与人工一致性 Precision 为 0.943（GPT-4o）/ 0.927（Gemini 1.5 Pro）/ 0.924（GPT-4o mini）（Table 6）；扰动下 VQA 准确率 0.980–0.998、AUROC 0.966–0.987（Table 7–8）；TaskGen 完整版代码生成成功率 98%，ToolGen 完整版 96%（Table 3）；系统整体约 5% 运行受错误影响，其中 TaskGen 占 42.8%、ToolGen 占 27.0%、Simulation Engine 占 13.5%、Plan Agent 占 9.5%、其他占 7.2%（Figure 6）。
- **定位与局限（Appendix A.5–A.6）：** ManipEvalAgent 定位为传统静态基准的**互补型诊断评测器**而非替代品（Table 10 明确指出其不提供静态基准的“绝对确定性”）；当前依赖仿真器已有 3D 资产与 RGB 视角，跨仿真器迁移仍需手工适配（如 LIBERO 需同时合成 BDDL 与 `Problem` 类），尚未部署到真机闭环评测。
- 对 wiki 的映射：[ManipEvalAgent：可提示的高效机器人操作策略评测框架](../../wiki/entities/paper-agentic-survey-iclr-2026-8833c8aa.md)
