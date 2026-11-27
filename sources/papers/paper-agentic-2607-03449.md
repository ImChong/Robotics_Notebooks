# HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control

- **作者：** Li Ji, Siyin Wang, Pengfang Qian, Xiaopeng Yu, Yihai Tian, Zhaoye Fei, Jingjing Gong, Xipeng Qiu
- **机构：** Fudan University（复旦大学）、Shanghai Innovation Institute（上海创智学院）、East China Normal University（华东师范大学）
- **版本 / 载体：** arXiv:2607.03449v1 [cs.RO]（2026-07-03 提交），HTML 全文（CC BY 4.0）
- **原文：** [arXiv 摘要](https://arxiv.org/abs/2607.03449) · [arXiv HTML 全文](https://arxiv.org/html/2607.03449)
- **项目页 / 代码：** **已开源**（分层记忆与任务服务端框架、提示词模板及 WidowX 参考推理客户端） — [GitHub: HappyWaterXP/HiMe](https://github.com/HappyWaterXP/HiMe)（README 另提供网页入口 [happywaterxp.github.io/HiMe](https://happywaterxp.github.io/HiMe/)；注：60 条真机示教数据与微调后 $\pi_{0.5}$ 低层策略权重未随仓发布，低层训练沿用 OpenPI / LeRobot 流程）
- **核对范围：** 已通读 arXiv:2607.03449v1 HTML 全文 §1–§6（Abstract、Introduction、Related Work、Methods、Experiments、Further Analysis、Conclusion、Limitations）与 Appendix §7–§10（Task Design and Metrics、Model Initialization/Training/Deployment、Additional Experiments、Prompts Details），逐项核对 Tables 1–11 与 Figures 1–9 图注；同步核对官方 GitHub 仓库 `HappyWaterXP/HiMe` 主页、`README.md` 与 `files_sample` 文件列表。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[HiMe 论文实体页](../../wiki/entities/paper-agentic-2607-03449.md)

### 1. 研究动机与“频率-能力悖论”（§1）

- 现有视觉-语言-动作（VLA）策略大多建立在马尔可夫假设 $p(a_t \mid o_t, l)$ 上，仅依据当前瞬时观测与语言指令输出动作，在非马尔可夫长时程操作中无法保持对历史状态或隐含约束的持续信念。
- 面向物理世界的长时程系统存在 **frequency-competence paradox**：具备强推理能力的大模型延迟高、难以满足高频实时控制；而高频策略或受限小模型缺乏足够世界知识与长程逻辑推理能力。此外，真实人机交互具有多模态约束（如用户偏好）与环境动态性，仅靠被动追加（append-only）或固定长度 FIFO 的记忆会出现知识陈旧与信息冲突。

### 2. 分层记忆结构与三层策略解耦（§3.2–§3.4）

HiMe 将具身智能拆分为时间分辨率逐级降低的三层：

1. **Executor（$\pi_e$）— Transient Memory（$\mathcal{T}_t$）：** 仅依赖当前观测 $o_t$ 与当前激活子目标 $\tau_t \in \mathcal{E}_t^p$ 生成动作 $a_t \sim \pi_e(a_t \mid o_t, \tau_t)$。实机采用在 DROID 检查点基础上微调的 $\pi_{0.5}$，2 Hz 推理预测 10 步 action chunk，在 10 Hz 控制环中开环执行 5 步。
2. **Sentry（$\pi_s$）— Working Memory（$\mathcal{W}_t$）：** 维护近期观测滑动窗口 $\mathcal{W}_t = \{o_{t-h_s}, \dots, o_t\}$（$h_s = 8$），由轻量 VLM（Qwen3-VL-8B）按固定监控间隔（§3.4 写 $n_m = 5$，§4.1 实机评测说明每执行 10 步 $\pi_e$ 查询一次）输出 XML 状态 `<status>done|not_done</status>`。仅当 $u_t = \mathbf{I}[\pi_s(\mathcal{W}_t, \tau_t) > \delta] = 1$（检测到子目标完成或状态偏离）时触发 Planner；因 Sentry 偏保守（8 帧下 Precision ~82%、Recall ~35%），系统保留固定间隔 Planner 兜底机制以防死循环。
3. **Planner（$\pi_p$）— Episodic Memory（$\mathcal{E}_t = (\mathcal{E}_t^c, \mathcal{E}_t^p)$）：** 采用高能力 VLM（主实验 GPT-4o；补充实验 Qwen3-VL-30B），按两轮交互协议（Turn 1 仅 `QUERY`；Turn 2 执行 `CREATE`/`UPDATE`/`DELETE` 并输出更新后的 `<plan_list>`）维护两类长期记忆：
   - **Contextual Memory（$\mathcal{E}_t^c$）：** 跨模态语义模式键值存储 $\mathcal{M}_t$（含 `id`、2–5 个对象/位置/用户偏好标签 `tags`、文本事实 `text` 与图像索引 `image_path`），结合 OpenAI `text-embedding-3` 向量库与标签检索。
   - **Procedural Memory（$\mathcal{E}_t^p$）：** 带 `[done]` / `[current]` / `[pending]` 状态标签的有序语言子目标列表。

### 3. 真机实验、消融与效率数据（§4–§5, Appendix §7–§9）

- **实验平台与任务（Table 3–4, §8）：** WidowX-250 / WidowX-250s 机械臂 + 前向 Intel RealSense D435 第三人称相机 + 腕部相机（左半拼腕部图、右半拼全局图，25 Hz 采集、缩放至 $224 \times 224$ 并降采样至 10 Hz）。低层 $\pi_{0.5}$ 在 60 条遥操作示教（50 常规 + 10 边界情况）上按 OpenPI 配方训练 30k 步（AdamW，峰值 LR $5\times 10^{-5}$，batch size 256）。三项任务每法评测 20 次试验：Object Search（6 子目标，~1450 步）、Counting（7 子目标，~1200 步）、Rearrangement（6 子目标，~1035 步）。
- **主实验对比（Table 1–2, Fig. 4）：** 平均任务进度方面，Transient Memory 为 14%，Transient Memory w/ Sentry 为 26%（Counting 上仅 23%），Flat Memory 为 65%，HiMe w/o Sentry 为 68%，完整 HiMe 达 **90%**。Table 2 给出分任务精确指标（HiMe vs. Flat Memory）：Object Search 进度 92% vs. 64%、API 调用 1.8 vs. 5.4、记忆命中 94% vs. 68%；Counting 进度 92% vs. 58%、API 调用 2.6 vs. 4.8、命中 98% vs. 61%；Rearrangement 进度 87% vs. 73%、API 调用 1.4 vs. 6.2、命中 92% vs. 76%。
- **模态与管理消融（§5.1–§5.2, Table 9）：** Object Search 偏重空间/视觉证据（Only image 86% > Only text 74%），Counting 偏重语义计数（Only text 91% > Only image 78%），跨模态组合平均 90%。管理机制上，FIFO（68%）< No Management 仅追加（86%）< 完整 Add/Update/Delete（90%）；在 Rearrangement 固定 8 条容量限制下（Table 9），Limited Active Management 仍取得 80.0%，显著高于同容量 FIFO 的 64.0%（无限制主动管理为 87.0%）。
- **可扩展性与延迟（Table 10–11）：** 多轮 Object Search 随轮数增加出现性能下降：1 轮（4 子目标，平均记忆 5.5 条）92.5% → 2 轮（8 子目标，9.9 条）76.3% → 3 轮（12 子目标，14.0 条）66.7%。Planner 端到端单次调用延迟（Table 11）：GPT-4o API P50 为 38.59 s（P99 57.28 s）；本地 vLLM 部署 Qwen3-VL-30B（2×H100 80GB）P50 为 6.83 s；Qwen3-VL-8B（1×H100 80GB）P50 为 6.10 s。

### 4. 局限与核对边界（Limitations）

- 缺少配套标准仿真基准对照，主要在自建真机桌面操作（以 inspect / pick-and-place / reset 原语为主）上评测；灵巧技能、超大环境与超长时程持续运行尚未充分验证。
