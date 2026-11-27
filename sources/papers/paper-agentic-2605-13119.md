# Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models

- **作者：** Zixing Lei, Changxing Liu, Yichen Xiong, Minhao Xiong, Yuanzhuo Ding, Zhipeng Zhang, Weixin Li, Siheng Chen
- **机构：** 上海交通大学（Shanghai Jiao Tong University）、北京中关村学院（Zhongguancun Academy）、北京航空航天大学（Beihang University）
- **版本 / 载体：** arXiv:2605.13119v2 [cs.RO]（2026-05-13 提交 v1，2026-10-12 修订 v2）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2605.13119) / [arXiv HTML 全文](https://arxiv.org/html/2605.13119)
- **项目页 / 代码：** 宣称将开源或未列链接（摘要声明 `"Code will be released."`，截至 2026-11-27 原文及索引未列出独立项目页或公开 GitHub 仓库链接）
- **核对范围：** 全文通读（Abstract、§1–§6、References、Appendix A–G，含 Table 1–14 与 Figure 1–8 图注）
- **通读：** 已完成

## 原文核对笔记

- **核心定位与问题拆解（§1–§3）：**
  - 端到端 VLA（如 OpenVLA-OFT、$\pi_{0.5}$）擅长局部连续动作执行，但在长时域任务中面临「长程闭环规划 + 多样化物理操作」双重负担；而传统具身智能体（SayCan、Code as Policies 等）虽具备高层规划能力，却依赖手工或刚性低层技能。
  - 本文提出 **VLAs-as-Tools**：由高层 VLM 智能体 $\Pi_\phi$ 负责场景分析、目标分解、状态跟踪与失败恢复，并将 VLA 封装为带工具族标签的有限时域可调用工具集 $\mathcal{T} = \{T_g\}_{g \in \mathcal{G}}$。
  - 双向交互接口 $\mathcal{I} = (\mathcal{C}, \mathcal{R})$：智能体发出调用消息 $c_k = (g_k, z_k) \in \mathcal{G} \times \mathcal{Z}$（离散工具族标签 + 面向当前场景的子任务指令）；工具返回连续局部进度反馈 $p_t \in [0, 1]$，通过阈值监视器实现事件触发式低频重规划（避免高频轮询 VLM）。
- **工具对齐后训练 TAPT 与接口机制（§4, Appendix A–D）：**
  - **自动标注流水线（Appendix B）：** 全局视频理解 → 基于夹爪开合与运动变化的状态切分 → 基于操作定义的工具族标注与一致性校验。在 DROID 上得到 11 个基础工具族（`reach`, `grasp`, `move`, `release`, `flip`, `rotate`, `push`, `pull`, `insert`, `press`, `strike`），LIBERO 与 RoboTwin 各使用其中 7 个。
  - **调用对齐训练单元：** 模仿学习窗口 $x_{\mathrm{IL}} = (o_{1:T}, z, g, a_{1:T}, p^\star_{1:T})$，进度目标为归一化时间代理 $p^\star_t = (t-1)/(T-1)$；强化学习（以 RLinf 中的 GRPO 实现）构建子任务有界 MDP，从边界/前序成功状态池 $\mathcal{S}^0_{z,g}$ 重置，采用二值完成谓词 $\psi_{z,g}(s)$，并在首轮子任务 RL 后引入 **Reset-state pool refresh** 缓解子任务衔接分布偏移。
  - **工具族残差参数化：** 共享预训练 VLA 主干，在线性层注入按 $g$ 路由的低秩 LoRA 残差 $\Delta W_g = B_g A_g$，附加轻量进度头 $D \to 1024 \to 256 \to 1$ MLP + sigmoid（$\lambda_{\mathrm{prog}} = 0.1$，2 Hz 与动作块同步预测；OpenVLA-OFT 仅增 +9% 参数与 +7% 推理延迟）。
- **主要实验结果（§5, Appendix E–G）：**
  - **仿真监督微调（Table 1）：** 在 LIBERO-Long 与 RoboTwin-SA 上，直接套用高层规划器会导致标准 SFT 掉点（LIBERO-Long 上 OpenVLA / OpenVLA-OFT / $\pi_{0.5}$ 分别变化 $-0.2$ / $-11.8$ / $-12.0$ 个百分点）；而 TAPT + 工具族接口在 $\pi_{0.5}$ 上分别达到 **97.2%**（+4.8）与 **62.50%**（+23.12），在 OpenVLA-OFT 上达到 **95.6%**（+3.6）与 **48.75%**（+31.87）。
  - **目标域 RL（Figure 3）：** LIBERO-Long 上 $\pi_{0.5}$ 从 Standard RL 80.0% 升至 TAPT + 接口 **91.2%**（+11.2）；CALVIN_D（原生子任务划分）上从 76.0% 升至 **82.3%**。
  - **现代长时域基准（Table 2）：** 2,500 步预算下，RoboMemArena 全任务 SR 从 26.15% 升至 **30.00%**（+3.85），部分进度从 48.99% 升至 **58.16%**（+9.17）；RoboLab-120 全任务 SR 从 28.5% 升至 **34.8%**（+6.3），部分进度从 36.7% 升至 **42.8%**（+6.1）。
  - **指令忠实度 LIBERO-IF（Table 3）：** $\pi_{0.5}$ Faithful Rate 24.8% → **54.8%**（+30.0），Non-biased Rate 39.6% → **54.6%**（+15.0）；OpenVLA-OFT Faithful 19.4% → **54.0%**（+34.6），Non-biased 31.2% → **47.4%**（+16.2）。
  - **真机实验（Table 5, Appendix F）：** Franka Research 3 + 单俯视 RealSense D435（每任务 30 条演示、30K 步、20 次评测），三任务平均成功率由微调 $\pi_{0.5}$ 的 **16.7%**（10/60）提升至 **58.3%**（35/60），其中存在视觉别名状态的 `Conditioned` 任务从 **5.0%** 升至 **65.0%**，24 步长调用序列的 `Long` 任务从 **25.0%** 升至 **45.0%**。
  - **规划效率与扩展性（Appendix C, E）：** 事件触发重规划使每回合运行时 VLM 调用由 109.5 / 58.2 / 57.2 次降至 **1.988 / 0.304 / 0.228** 次；增量新增工具族可保留 **97.6%** 旧技能；更换四款开源/闭源 VLM 规划器（Gemini 3 Flash / GPT 5.4 / Qwen3.5-397B-A17B / Qwen3VL-235B-A22B）波动仅 0.6 个百分点。
- **对 wiki 的映射：** [Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models](../../wiki/entities/paper-agentic-2605-13119.md)
