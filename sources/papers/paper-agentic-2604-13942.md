# Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Progress Verification

- **作者：** Zhen Liu, Zhe Hu, Xinyu Ning, Hanlin Wang, XinXin Xie, Weize Li, Zhipeng Tang, Chongyu Wang, Zejun Yang, Yitong Liu, Zhongzhu Pu（通信邮箱：`puzz@inspireomni.ai`）
- **机构：** Beijing University of Posts and Telecommunications; InspireOmni AI; Tsinghua University
- **版本 / 载体：** arXiv:2604.13942v2 [cs.RO]（2026-04-15 提交 v1，2026-10-13 修订 v2；v1/索引标题为 *Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection*，v2 HTML 标题更新为 *Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Progress Verification*）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2604.13942) / [arXiv Abs](https://arxiv.org/abs/2604.13942)
- **项目页 / 代码：** 未确认开源代码（arXiv v2 文本页眉包含 `\metadata [GitHub] InspireOmniRobot/Goal2Skill`，但正文、附录及外部候选仓库扫描 `candidate_repos` 均未提供已验证的可访问 GitHub URL、独立项目页或可运行源码文件）
- **核对范围：** arXiv:2604.13942v2 全文（Abstract、§1–§6、References、Appendix §7 训练与推理细节 Table 3–4、§8 轨迹案例分析 Fig. 6–12、§9 消融与完整评测结果 Table 5–9，以及主文 Table 1–2、Fig. 1–5 图注与正文解析）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与双系统设计（§1, §3.1）：** 现有端到端 VLA 策略依赖有限观测窗口与前馈动作预测，在长程、部分可观测、存在遮挡与多阶段依赖的操作任务中容易遗忘历史信息并缺乏错误恢复能力。Goal2Skill 构建「高层 VLM 推理大脑 $\Phi$ + 底层 VLA 执行器 $\Pi$」双系统闭环，并在三个时间尺度上解耦运行：高频动作块（action chunk）执行、周期性进度验证（每 $n$ 个 chunk 触发一次）与稀疏任务规划/重规划。
- **高层规划与协调（§3.2）：**
  1. **任务分解与自适应规划（§3.2.1）：** 给定自然语言目标 $\mathcal{G}$ 与初始观测 $o_0$，规划器生成有序子任务序列 $\mathcal{P} = \langle \tau_1, \dots, \tau_K \rangle = \Phi_{\mathrm{plan}}(\mathcal{G}, o_0)$，每个子任务表示为 $\tau_k = (\ell_k, \delta_k, j_k)$（子任务语言指令、单次尝试步数预算、底层技能索引）。
  2. **主动记忆管理（§3.2.2）：** 维护三元组情景级结构化记忆 $\mathcal{M}_t = (\mathcal{H}_t, \mathcal{W}_t, \mathcal{E}_t)$：情景历史 $\mathcal{H}_t$ 压缩记录跨子任务状态变化与后续步骤所需的历史信息；工作记忆 $\mathcal{W}_t$ 跟踪当前子任务的起始参照状态、近期执行证据与局部进度；错误寄存器 $\mathcal{E}_t$ 记录失败偏差、假设原因、纠正方案与结果。记忆更新 $\mathcal{M}_t^+ = \Phi_{\mathrm{mem}}(\mathcal{M}_t, \mathcal{P}_{t:T}, \mathcal{Z}_t, v_t)$ 显式参考剩余计划 $\mathcal{P}_{t:T}$，实现面向未来步骤的主动信息保留。
  3. **记忆接地进度验证与自进化验证器（§3.2.3）：** 每执行 $n$ 个动作块，验证器依据检索到的验证技能 $\theta_k$ 输出 $v_t = (e_t, c_t) = V_{\theta_k}(\mathcal{Z}_t, \tau_k, \mathcal{M}_t)$，其中 $c_t \in \{\textsc{completed}, \textsc{ongoing}, \textsc{failed}\}$。其混合结合规则检查、轻量模型与 VLM 推理；积累轨迹可识别验证盲区并更新/新增验证技能，经**轨迹回放（trajectory replay）验证**后再提交至验证技能库（如 Fig. 4 中的摆动计数到达/去重逻辑）。最终子任务结束后还需核验全局目标 $\mathcal{G}$。
  4. **反思修正与重规划（§3.2.4）：** 失败、预算耗尽或全局目标未达成时触发反思 $(d_t, \rho_t, \tau'_t) = \Phi_{\mathrm{reflect}}(\mathcal{F}_t, \mathcal{M}_t)$，按 `Retry`（原指令重试） $\to$ `Adjust`（局部改写指令/定位/参数） $\to$ `Replan`（重写剩余子任务序列）逐级升级，并保留已完成的合理进展。
- **底层执行器与任务条件视觉过滤（§3.3）：**
  - 参考 HSC-VLA（Liu et al., 2026c），子任务开始时通过空间定位识别干扰物体边界框 $\mathcal{B}_{k,t}$，由分割模型 $\mathcal{S}$ 生成二值干扰掩码 $Q_t$ 并轻量追踪，构造掩码过滤图像 $\hat{I}_t = I_t \odot (1 - Q_t)$。仅底层动作策略接收 $\hat{I}_t$，高层模块仍保留原始观测 $I_t$ 以免丢失全局证据。
  - 底层策略基于 $\pi_{0.5}$ 以滚动时域（receding-horizon）输出长度为 $H=20$ 的连续动作块 $\mathbf{A}_t = (a_t, \dots, a_{t+H-1})$。
- **训练与推理配置（§7，Table 3–4）：**
  - **高层 VLM 后端：** 所有高层推理模块统一使用在线模型 `qwen3.8-flash`（经 OpenRouter API 调用）。全局预算 $T=1300$ 步，单子任务单次尝试预算 $\delta_k=320$ 步，单子任务最多尝试 $K_{\mathrm{sub}}=3$ 次，计划序列耗尽后全局恢复与重规划最多 3 轮。
  - **RoboMME 低层策略：** 基于 $\pi_{0.5}$（LIBERO 配置）联合多任务训练 16 个任务，每任务 100 条演示（其中 20 条施加视觉掩码过滤、80 条保留原图，不引入额外参数），$H=20$，batch size 64，AdamW，恒定学习率 $5\times 10^{-5}$，warmup 0.05，EMA 0.999，80,000 步，冻结 SigLIP2 视觉编码器，8× NVIDIA A800 训练。
  - **RMBench 低层策略：** 基于 $\pi_{0.5}$ 基础检查点微调，每任务 50 条专家演示（10 条掩码 [20%] + 40 条原图 [80%]），batch size 64，AdamW，余弦退火（$2.5\times 10^{-5} \to 2.5\times 10^{-6}$），warmup 1,000 步，EMA 0.99，20,000 步，`bfloat16` 精度。
- **核心实验与消融数据（§5, §9）：**
  - **RoboMME（16 任务 × 50 测试集 = 800 局，Table 1 & Table 7–9）：** 6 个代表任务均值 **70.0%**（最强非 Oracle 基线 MemER 为 44.7%，GroundSG+QwenVL 40.3%，MME-VLA FrameSamp 38.7%，$\pi_{0.5}$ 19.7%；Oracle 参考 GroundSG+Oracle 89.0%）；全 16 任务均值 **51.5%**（MemER 43.6%，FrameSamp 42.3%，GroundSG+QwenVL 32.8%，$\pi_{0.5}$ 17.8%）。在 `Reference`（均值 69.5% vs. MemER 41.5%）与 `Permanence`（均值 57.0%）类别优势显著，如 `VideoPlaceOrder` 在 Easy/Medium/Hard 分别达到 96.2%/91.7%/75.0%（其余非 Oracle 方法 Hard 最高 33.3%）。
  - **RMBench（5 任务，Table 2）：** 总体均值 **32.4%**（MemER 10.2%，X-VLA 9.8%，ACT 9.8%，$\pi_{0.5}$ 8.8%，DP 4.2%）；M(1) 任务均值 23.0%（Observe and Pick Up 8.0%，Rearrange Blocks 38.0%），M(n) 多阶段任务均值 38.7%（Battery Try 46.0%，Blocks Ranking Try 60.0%，Press Button 10.0%，而全部对比基线在 Press Button 上均为 0.0%）。
  - **结构化记忆消融（RMBench 3 任务子集，Table 5）：** 完整系统均值 **35.3%** $\to$ 去掉 $\mathcal{E}_t$ 降至 **28.0%** $\to$ 再去掉 $\mathcal{W}_t$ 降至 **27.7%** $\to$ 再去掉 $\mathcal{H}_t$ 暴跌至 **6.7%**（`Blocks Ranking Try` 从 54% 跌至 6%）。
  - **反思修正消融（RMBench 2 任务子集，Table 6）：** 完整系统均值 **28.0%**（Battery Try 46% / Press Button 10%） $\to$ 去掉 Reflection 降至 **17.5%**（30% / 5%）。
- **核查边界与局限：** 全文仅汇报 RoboMME 与 RMBench 仿真基准评测，未包含真机实验；Counting（52.0% vs. FrameSamp 66.0%）与 Imitation（27.5% vs. FrameSamp 42.5%）两类更依赖密集视觉轨迹历史的任务仍落后于直接采样历史帧的方法；官方源码仓截至 2026-11-27 未能通过可运行文件核查。
- **对 wiki 的映射：** [Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection](../../wiki/entities/paper-agentic-2604-13942.md)
