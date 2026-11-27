# When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering

- **作者：** Jessie Yuan*, Yilin Wu*, Andrea Bajcsy（* 共同一作）
- **机构：** Carnegie Mellon University（CMU）
- **版本 / 载体：** arXiv:2602.22474v2 [cs.RO]（2026-02-25 提交 v1，2026-05-12 修订 v2）；Robotics: Science and Systems（RSS）2026 收录论文
- **原文：** https://arxiv.org/abs/2602.22474 （HTML 全文：https://arxiv.org/html/2602.22474 ；项目页：https://jessie-yuan.github.io/ups/ ）
- **项目页 / 代码：** 已开源（官方仓库 https://github.com/CMU-IntentLab/uncertainty_aware_policy_steering 公开了 RoboMimic NutAssembly 仿真任务的完整代码、Dreamer-v3 世界模型、扩散/残差策略训练与共形预测校准脚本，并在 Hugging Face 公开了 `jzyuan04/ups_nutassembly_*` 预训练权重与演示/rollout 数据集；硬件流水线与仿真一致但仓库未附带硬件数据与脚本，且未附带 LICENSE 文件）
- **核对范围：** 通读 arXiv:2602.22474v2 HTML 全文第 I–VII 节（Abstract、Introduction、Related Work、Problem Formulation、Approach IV-A/IV-B/IV-C、Simulation & Hardware Experiments V-A/V-B、Conclusion & Limitations）及附录 A（A-A 覆盖保证证明、A-B 评分函数对比、A-C 硬件实验与指令表 I–III、A-D 仿真设置与表 IV、A-E 校准直方图与 Logits 消融、A-F 残差策略与表 V–VIII）、项目页文本以及官方代码仓 `CMU-IntentLab/uncertainty_aware_policy_steering` 的 README 与文件列表。
- **通读：** 已完成

## 原文核对笔记

- **核心问题（VLM 验证器过度自信）：** 现有 VLM 策略引导（Policy Steering，如 FOREWARN）在面对高层次任务指令语义歧义或底层预训练策略本身能力不足时，往往输出过度自信的判别分布——在歧义指令下盲目选定某一行为模式，或在所有候选动作均无法完成任务时仍强行选中错误动作而不停机求助。
- **核心架构（UPS）：**
  1. **交错生成与世界模型推演（Sec. IV-A）：** 针对短视距动作块（action chunk，$H \ll T$）结果难以区分的问题，将基础扩散策略 $\pi(a \mid o_t)$ 与 Dreamer-v3 潜在世界模型 $\mathcal{W}_\phi = (\mathcal{E}_\phi, f_\phi, \mathcal{D}_\phi)$ 交替循环 $T/H$ 次，对 $K$ 条并行样本生成长视距预测观测序列 $\{\hat{o}_{t:t+T}^k\}_{k=1}^K$。
  2. **视频文本转述与多选验证：** 使用 `Gemini-3-flash-preview` 作为转述器 $\mathcal{T}^{\text{VLM}}$ 将预测轨迹转写为文本描述 $\ell_t^k$，语义相同的描述归并后附加第 $K+1$ 项 `"none of the above samples are correct"`，交由 `Gemini-2.0-flash` 进行多选题概率打分。
  3. **贝叶斯意图分解与序列级共形预测（Sec. IV-B）：** 为避免直接询问 VLM 概率或使用 CoT 导致的过自信/过保守，按公式 $\mathbb{P}^{\text{VLM}}(y\mid\{\ell^k\}_{k=1}^K;\mathcal{L})=\sum_{\theta\in\Theta}\mathbb{P}(y\mid\{\ell^k\}_{k=1}^K,\theta)\mathbb{P}(\theta\mid\mathcal{L})$ 先生成潜在用户意图集 $\Theta$ 并边缘化。在包含 $M$ 步验证的序列校准集（$N=80$，$1-\epsilon=0.85$）上定义完备性非一致性分数 $\kappa(\bar{x},\bar{y}) = 1 - \min_{x_i\in\bar{x}}\min_{y\in\mathcal{Y}_i^*}\mathbb{P}^{\text{VLM}}(y\mid x_i)$（区别于 KnowNo 的存在性 $\max$ 分数），在测试时以经验分位数 $\hat{q}$ 构建预测集 $C(x_i^{\text{test}})$（Theorem 1 保证 $\mathbb{P}(\mathcal{Y}_i^* \subseteq C(x_i^{\text{test}})) \ge 1-\epsilon$）。
  4. **三级不确定性消解策略（Sec. IV-C）：**
     - **Act（执行）：** $|C|=1$ 且不含 $K+1$，直接执行对应动作序列。
     - **Ask（语言澄清）：** $|C|>1$，向用户发起自然语言澄清提问并将回答总结为新指令 $\tilde{\mathcal{L}}$ 后重新验证，直到 $|C|=1$。
     - **Learn（残差持续学习）：** $C=\{K+1\}$，判定底层策略无能为力，仅在此时请求 SpaceMouse 遥操作纠偏；记录 $\Delta a = a_{\text{human}} - a_{\text{base}}$ 与干预标签，训练干预门控分类器（BCE）和残差策略 $\pi^{\text{residual}}$（MSE + tanh）；再部署时在解码后的世界模型图像上对 $K/2$ 个样本启用残差策略、保留 $K/2$ 个基础策略样本以防灾难性遗忘，并用残差 rollout 重训世界模型与重校准 $\hat{q}$。
- **核心实验结果（Sec. V & Appendix A）：**
  - **不确定性量化（Figs. 3–5, Appendix A-E）：** 在 Robomimic 仿真、`PnP Cup` 与 `Insert Block` 真机任务中，`CP + Bayesian Intent`（自生成概率分数）在三类场景均达到 $\ge 85\%$ 目标覆盖率，且在明确与无能场景保持低澄清率；一阶 token logits + softmax（$T=4$）存在选项偏置，表现差于自生成概率。
  - **策略引导与持续学习（Tables I & IV, Fig. 6）：**
    - **Hardware `PnP Cup`（Table I, 各 20 条测试）：** `Base Policy` 在明确/歧义场景 TP 均为 0.35；`Forewarn` 为 0.80 / 0.50；`UPS w/ Clarification` 升至 0.85 / 0.65；`UPS w/ Clarification + Residual` 达到 **0.90 / 0.80**（总成功率 85%，相对歧义场景未校准 `Forewarn` 提升 **30%**）。
    - **Simulation `NutAssemblySquare`（Table IV, 各 20 条测试）：** `Base Policy` TP 为 0.45 / 0.25；`Forewarn` 为 0.65 / 0.0；`UPS w/ Clarification` 为 0.60 / 0.55；`UPS w/ Clarification + Residual` 为 **0.70 / 0.65**。
  - **人类干预率（Sec. V-B1）：** 在 `PnP Cup` 中 UPS 为 **0.06**（HG-DAgger 0.10，EnsembleDAgger 0.16）；`Insert Block` 中 UPS 为 **0.05**（HG-DAgger 0.06，EnsembleDAgger 0.26）；仿真中 UPS 为 **0.058**（HG-DAgger 0.20，EnsembleDAgger 0.27）。
- **局限与边界（Sec. VI & V-B2）：** 假设世界模型推演画面与 VLM 视频转述始终准确完整，尚未建模世界模型本身的不确定性；在仿真与精细接触任务 `Insert Block` 上受限于当前世界模型对精密接触动力学的建模精度，覆盖率与成功率略低于 85% 目标。
- **对 wiki 的映射：** [When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering](../../wiki/entities/paper-agentic-2602-22474.md)
