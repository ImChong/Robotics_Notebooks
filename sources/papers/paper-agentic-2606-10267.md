# What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents

- **作者：** Jiaheng Hu, Mohit Shridhar, Caden Lu, Dhruv Shah, Hao-Tien Lewis Chiang, Jie Tan, Annie Xie
- **机构：** Google DeepMind（第一作者 Jiaheng Hu 于 Google DeepMind 实习期间完成）
- **版本 / 载体：** arXiv:2606.10267v2 [cs.RO]（Submitted 2026-06-09，last revised 2026-09-06；arXiv HTML 全文及官方项目页）
- **原文：** <https://arxiv.org/html/2606.10267>（Abstract：<https://arxiv.org/abs/2606.10267>；项目页：<http://jiahenghu.github.io/hi-vla>）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查官方项目页 `http://jiahenghu.github.io/hi-vla` 与 arXiv 原文，仅发布论文内容与演示视频，未列出 GitHub 仓库或源码发布链接；跨栈验证使用了外部开源组件 Qwen3-VL、$\pi_{0.5}$ 与 LIBERO，并非本论文独立开源框架）
- **核对范围：** 通读 arXiv:2606.10267v2 HTML 全文（Abstract、§1–§5、Alg. 1、Table 1–2、Appendix A–K 含 Table 3–8 及 Prompts E–I）与官方项目页 `http://jiahenghu.github.io/hi-vla`（Key Results、Real Robot Experiment、Findings、BibTeX）。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与统一形式化（§1, §3, Alg. 1）：**
  - 论文聚焦双层分层视觉-语言-动作系统（Hierarchical VLA, Hi-VLA）：高层 VLM 规划器 $\pi_{\text{VLM}}(l \mid \boldsymbol{o}, I)$ 根据任务指令 $I$ 与经过表征/记忆处理的观测 $\boldsymbol{o}$ 生成语言子目标 $l$，底层 VLA 控制器 $\pi_{\text{VLA}}(a \mid o_t, l)$ 在终止条件 $\beta(o_t, t)$ 触发前以更高频率输出低层动作 $a$。
  - 作者用半马尔可夫/Options 框架将既有 Hi-VLA（如 RT-H、Hi Robot、$\pi_{0.5}$、Gemini Robotics 1.5、HAMSTER、Helix 等）统一到同一控制循环：引入观测表征模块 $o'_t \leftarrow \phi(o_t)$ 与记忆模块 $\boldsymbol{o} \leftarrow \text{mem}([o'_i]_{i \le t})$，实现受控单变量消融。

- **受控实验设计与主要发现（§4, Appendix A–J）：**
  - **基准与任务划分：** 在 MuJoCo ALOHA 双臂桌面操作套件（每类 5 个任务、每任务 200 次试验、单次 2 分钟预算）上划分为短时域（Short-Horizon）、长时域（Long-Horizon）与语义推理（Reasoning）三类；标准对照组为 Gemini 2.5 Flash (thinking) + GROD-3B (real-data only) + 8 s ($T=400$) 固定周期切换 + 含接触信息的场景描述 + 窗口长度为 3 的原始记忆（无摘要）。
  - **高层 VLM（Table 3）：** 开启思维链推理（thinking）带来稳定提升（如 Gemini 2.5 Flash-Lite 在推理任务上由 58.51% 升至 75.20%，长时域由 48.73% 升至 58.21%）；在开启 thinking 时，Lite / Flash / Pro 随参数规模增大并未呈单调增益。
  - **底层 VLA（Table 4）：** 模型规模（GROD-3B 显著优于 GROD-1B）与**指令可操纵性（steerability）**是决定性瓶颈；用同域仿真演示微调 GROD-1B 后，因对多样化/改写子目标指令的遵循能力退化，长时域成功率从 41.30% 暴跌至 7.50%。
  - **切换/终止条件（Table 5, Appendix C）：** 基于成功检测（Success Detector）在长时域（57.39%）和推理任务（80.89%）上最优；由 VLM 预先估计执行时长最差（长时域 43.50%）；固定频率切换选 4–8 s 可在调用开销与成功率间取得折中。对成功检测器注入合成噪声发现，10% 随机误差几乎无损，但高假阳性（FP）会因提前切走子目标而严重破坏性能。
  - **观测表征（Table 6）：** 仅靠 VLM 朴素图像描述（35.70% 长时域）不如原始图像（38.84%）；加入 2D 边界框（47.90%）或特权接触信息（52.36%）生成结构化场景描述则带来显著提升，弥补了 VLM 在复杂任务下容易忽略图像局部几何与接触细节的问题。
  - **记忆机制（Table 7–8）：** 单回合内改变原始上下文窗口长度（1 / 3 / 5 / Full）或做单步/当前回合摘要，增益有限甚至为负；而跨回合经验总结（滚动运行 10 回合后由 VLM 提炼底层 VLA 的语言 affordance 摘要）在短/长/推理三类任务上均带来明显增益（相对无摘要分别为 75.81%→79.45%、52.36%→60.00%、72.62%→80.30%）。
  - **组合系统与真机/跨栈验证（Table 1–2, Appendix B）：** 组合最优设计（Best Hierarchy）在仿真短/长/推理任务上分别达到 78.22% / 67.08% / 80.89%，远高于朴素分层（69.57% / 40.56% / 66.49%）与扁平 VLA（69.63% / 25.30% / 50.90%）。在非特权真机 ALOHA 水果配色放置任务上，Best Hierarchy（基于边界框观测与纯视觉成功检测器）取得 12/15 正确放置，高于朴素分层（9/15）与扁平 VLA（3/15），且能在第 6 步误放红葡萄后恢复纠正。在 Qwen3-VL + $\pi_{0.5}$ + LIBERO 跨栈实验（Table 2）中，8× VLM 扩参仍无单调增益，而窄指令微调 $\pi_{0.5}$ 导致成功率从 73.3% 骤降至 32.0%。

- **对 wiki 的映射：** 升格并完善独立论文实体页 [paper-agentic-2606-10267](../../wiki/entities/paper-agentic-2606-10267.md)，系统记录 Options 控制形式化、五大设计轴受控消融结论、仿真/真机/跨栈定量结果与无官方代码边界。
