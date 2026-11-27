# Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control

- **作者：** William Chen, Jagdeep Singh Bhatia, Catherine Glossop, Nikhil Mathihalli, Ria Doshi, Andy Tang, Danny Driess, Karl Pertsch, Sergey Levine
- **机构：** UC Berkeley、Stanford University、Physical Intelligence
- **版本 / 载体：** arXiv:2602.13193v3 [cs.RO]（v1 提交于 2026-02-13，v3 修订于 2026-04-06）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2602.13193) / [arXiv HTML 全文](https://arxiv.org/html/2602.13193)
- **项目页 / 代码：** 部分开源；[官方项目页](http://steerable-policies.github.io)、[OpenVLA 版训练与推理代码仓库 `steerable-policies-bridge`](https://github.com/steerable-policies/steerable-policies-bridge)（MIT）、[配套机器人环境 `steerable-gym`](https://github.com/steerable-policies/steerable-gym)、[HuggingFace 模型权重 `steerable-policy-openvla-7b-bridge`](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge) 与 [Bridge 标注数据集 `steering_features_bridge`](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge)。注：仓库 README 中高阶 Embodied Reasoner 权重链接仍处于注释占位状态，且论文附录中的 $\pi_{0.5}$ 微调实现未随 `steerable-policies-bridge` 单独释放。
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections I–VII、Appendices A–F 含 Table I–III 与 Fig. 1–24 说明文字）、官方项目页文本，以及 `steerable-policies/steerable-policies-bridge` README 与目录结构。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与动机：** 分层机器人策略通常由高阶 VLM 输出任务级/子任务级自然语言指令来驱动低阶 VLA，但标准机器人数据集（如 Bridge）的原始任务标签过于公式化、单一且语义粗糙，限制了 VLM 的视觉/常识推理对低阶动作的精细调控（steerability）。
2. **六类操控指令抽象（Section IV-A）：**
   - Tasks（标准任务指令）
   - Subtasks（语义子任务，如 `"reach for the carrot"`）
   - Atomic motions（原子运动指令，如 `"move left"`、`"open gripper"`）
   - Gripper traces（归一化 `[0, 255]` 像素坐标序列，如 `"move from [x1, y1] to [x2, y2]"`）
   - Points（关键目标/位置指向，如 `"grasp at [x, y]"`）
   - Combinations（多模态混合指令）
3. **大规模合成标注管线（Section IV-B & Appendix A）：**
   - 运动语言：按 Zawalski et al. (ECoT) 程序化提取。
   - 开放词汇物体分割与追踪：MolmoE-1B-0924 识别最多 4 个场景物体 → Molmo-7B-D-0924 预测首帧坐标 → SAM 2 跨视频追踪掩码并转为 `<loc><seg>` 表征。
   - 夹爪轨迹：人工标注 100 张 Bridge 图像训练轻量 DETR 提取全库夹爪质心。
   - 子任务与指令生成：调用 Gemini 2.0 先切分子任务，再改写为上述多种风格的转向指令，将 Bridge 从约 38k 原始任务标签扩展到 206k 子任务与近 2M 条转向指令；另生成事后推理依据（rationalizations）用于训练高阶具身推理器。
4. **两种高阶控制方案（Section V）：**
   - **微调高阶具身推理器（Section V-A）：** 基于 Prismatic VLM 7B 自回归输出 rationale 与转向指令，每 $N=5$ 个环境步重查询一次，分别驱动 OpenVLA 版（80k 步，8×H100）与 $\pi_{0.5}$ 版（30k 步，8×H200，OpenPi 配方、无 Knowledge Insulation、无本体感觉输入、action chunk=4 仅执行首步）Steerable Policy。
   - **现成 VLM 上下文推理（Section V-B & Appendix B）：** 使用 Gemini 3.0（1024 thinking budget），以历史观测图像与已执行指令作为上下文，自主选择最优抽象层级，每 $N=20$ 步重查询；其中 `<object>` 占位符由另一次 Gemini 3.0 定位调用转为 `[x, y]`，且实验中禁用了视觉模型较不稳定的完整夹爪轨迹输出，仅保留指向点等其他抽象。
5. **实机评测结果（Bridge WidowX，累计 >650 episodes）：**
   - Human Oracle（Section VI-A）：全风格自由组合接近 100% 成功率；单风格下原子运动总体最强但各风格互补。
   - 具身推理套件（Table I）：全任务聚合成功率上，Steerable Policy + Embodied Reasoner（OpenVLA）达 `84.6 ± 3.4%`，$\pi_{0.5}$ 版达 `83.7 ± 3.4%`，优于标准 OpenVLA（`50.5 ± 4.7%`）、$\pi_{0.5}$（`61.3 ± 4.6%`）、ECoT-Lite Reasoning Dropout（`60.4 ± 4.6%`）、ECoT-Lite Reasoning Pre-training（`69.4 ± 4.4%`）、Full ECoT（`77.5 ± 4.0%`）及无推理消融（`66.7 ± 4.5%`）。
   - 多步上下文推理套件（Table II，平均任务进度分）：Full Approach 达 `84 ± 3.7%`，优于标准 OpenVLA（`48 ± 5.1%`）、仅限子任务的 SayCan-like 基线（`64 ± 5.5%`）以及 0 thinking tokens 的无推理消融（`78 ± 4.0%`）。
6. **对 wiki 的映射：** 归纳为实体页 [Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control](../../wiki/entities/paper-agentic-2602-13193.md)。
