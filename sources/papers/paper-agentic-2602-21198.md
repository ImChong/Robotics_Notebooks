# Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs

- **作者：** Yining Hong, Huang Huang, Manling Li, Li Fei-Fei, Leonidas Guibas, Jiajun Wu, Yejin Choi
- **机构：** Stanford University（Yining Hong, Huang Huang, Li Fei-Fei, Leonidas Guibas, Jiajun Wu, Yejin Choi）、Northwestern University（Manling Li）
- **版本 / 载体：** arXiv:2602.21198v3 [cs.LG]（2026-02-24 提交 v1，2026-05-25 修订 v3；CC BY 4.0）
- **原文：** [arXiv HTML v3](https://arxiv.org/html/2602.21198) / [arXiv Abs](https://arxiv.org/abs/2602.21198)
- **项目页 / 代码：** 部分开源 — [官方 GitHub 仓库](https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning)（附项目主页 [reflective-test-time-planning.github.io](https://reflective-test-time-planning.github.io)）；仓库含基于 OmniGibson/BEHAVIOR-1K 的修改环境、`BEHAVIOR-model` 样例推理脚本及 MuJoCo Cupboard 交互环境 `cupboard/roboworld/envs/franka_cupboard_interactive.py`，但 README 说明 BEHAVIOR 完整评测集将于接收后发布、Cupboard 推理代码尚在清理中将于年内发布。
- **核对范围：** 通读 `arXiv:2602.21198v3` HTML 全文（摘要、§1–§6、References、Appendix A–R，含 Algorithm 1、Table 1–18 与 Figure 1–8 图注）及 GitHub 仓库主页 README 与样例文件目录列表。
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 现有具身大语言模型（Embodied LLM）在部署期参数固定，将多步部署变成相互独立的试错，无法从失败中积累经验；纯文本反思（如 Reflexion、Self-Refine）难以转化为持久化参数更新，而经典 RL 依赖多回合标量奖励与环境重置，难以在单次长程部署内诊断和纠正错误。
- **方法架构：** 提出 Reflective Test-Time Planning（反思式测试时规划），部署期实例化三个具身 LLM（由同一 SFT 底座初始化）：动作生成模型 $\pi_\theta$、内部反思模型 $V_{\phi_i}$、外部反思模型 $V_{\phi_e}$（冻结）。
  - **Reflection-in-Action (RIA，行动中反思)：** 执行前以高温 $T$ 从 $\pi_\theta$ 采样 $N$ 个候选动作，由 $V_{\phi_i}$ 生成自然语言内部反思与 $[0,100]$ 评分并择优执行（Test-Time Scaling）。
  - **Reflection-on-Action (ROA，行动后反思) 与 Retrospective Reflection（事后回溯反思）：** 每步执行后由 $V_{\phi_e}$ 结合结果反馈 $e_t$ 生成局部外部反思并存入工作记忆 $\mathcal{W}$（容量 $K=5$）；当 $|\mathcal{W}|=K$ 或触发关键里程碑（如离开房间 `EXIT`）时，利用事后信息对 $\mathcal{W}\cup\mathcal{D}_{\text{retro}}$ 中的历史动作重新打分归因。
  - **测试时训练（TTT）：** 将回溯反思数据与未探索动作正则化锚定数据 $\mathcal{D}_{\text{reg}}$（50%/50% 混合）构成 $\mathcal{D}_{\text{train}}$，通过监督学习（NLL）更新 $V_{\phi_i}$，通过 REINFORCE 策略梯度（奖励 $r=2(s_r/100)-1$）更新 $\pi_\theta$。
- **核心基准与结论：**
  - **Long-Horizon Household（BEHAVIOR-1K / LLaVA-3D-7B）：** 平均成功率 33.65%（Fitting 44.7%、Selection 32.4%、Preparation 31.7%、Hybrid 25.8%），显著优于 Reflexion（9.92%）、Self-Refine（11.20%）、3DLLM-Mem（11.13%）、PPO（3.53%）、DreamerV3（10.02%）及零样本商用模型 GPT-5（20.7%）与 Gemini 2.5 Pro（19.7%）；在 HM3D 逼真场景零样本迁移上 Preparation 达到 19.5%（次高 3DLLM-Mem 为 7.32%）。
  - **MuJoCo Cupboard Fitting（Qwen2.5-VL-3B）与 Franka Panda 真机：** 仿真 Fit 率 60.2% / Correct 率 25.3%（LoRA TTT 与全参数非视觉微调 57.4% 相近但可训练参数减少 >95%）；Franka Panda 真机微调后全模型达到 44.2% Fit / 16.6% Correct（无反思消融仅 20.7% / 7.2%）。
  - **关键机制发现：** RIA 与 ROA 相互依赖（单缺其一在多类任务上反而劣于两者皆无）；单步动作生成优于滚动时域规划（Receding Horizon 5× 计算量下 Fit 率反而降至 57.8%）；SFT 初始化呈倒 U 型且主要提供格式与基本具身先验；过大 LoRA 秩（如 $(32,32)$）在少量样本高学习率 TTT 下会发生模式坍缩。
- **对 wiki 的映射：** [Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs](../../wiki/entities/paper-agentic-2602-21198.md)
