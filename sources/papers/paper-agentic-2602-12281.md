# Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment

- **作者：** Jacky Kwok, Xilun Zhang, Mengdi Xu, Yuejiang Liu, Azalia Mirhoseini, Chelsea Finn, Marco Pavone
- **机构：** Stanford University, NVIDIA Research
- **版本 / 载体：** arXiv:2602.12281v2 [cs.RO]（2026-02-12 提交 v1，2026-02-18 修订 v2）
- **原文：** <https://arxiv.org/abs/2602.12281> / <https://arxiv.org/html/2602.12281>
- **项目页 / 代码：** 部分开源（Step 2.5 核查：官方主页 <https://cover-vla.github.io> ；官方主仓库 <https://github.com/cover-vla/cover-vla> 已开源 Bridge/SIMPLER 推理与评测管线及预训练验证器权重 `cover-vla/cover-vla-bridge`；配套仓库 <https://github.com/cover-vla/cover-vla-polaris> 已开源 PolaRiS/DROID 评测脚本与权重 `cover-vla/cover-vla-polaris`；主仓库 README To-Do 标明验证器训练管线 `Release verifier training pipeline` 与独立验证器服务 `Develop CoVer verifier server` 尚未发布）
- **核对范围：** 通读 arXiv:2602.12281v2 HTML 全文（Abstract、§1 引言、§2 相关工作、§3 测试时缩放定律分析、§4 方法与 Algorithm 1–2、§5 实验与 Table 1–2、§6 结论、§8.1–§8.10 附录与 Table 3–9、§9 符号表），并核对官方仓库 <https://github.com/cover-vla/cover-vla> README 与目录采样。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与测试时缩放分析（§1, §3）

- **核心痛点：** 视觉-语言-动作（VLA）模型存在「意图-动作鸿沟」（intention-action gap），对同义改写或对抗性（red-teaming）指令极度敏感；单纯扩增策略预训练或指令改写微调收益有限，且易引发 VLM 多模态推理的灾难性遗忘。
- **Bridge V2 测试时缩放分析（§3）：** 从 Bridge V2 均匀采样 1,000 个 $(s, a, I)$ 三元组，对比四种候选动作生成方式（重复采样、高斯扰动、VLM 指令改写、结合改写与重复采样的混合采样），以预言机验证器下的 NRMSE 衡量动作误差。动作误差与总推理 FLOPs 呈幂律关系 $\log(e) \approx \log(a) + b \cdot \log(k)$；指令改写明显优于纯重复采样与高斯扰动，混合采样（同时扩展改写数 $K$ 与每条改写采样动作数 $M$）能跳出局部极小、产生最广的动作覆盖分布。

### 2. 核心方法：CoVer 与 CoVer-VLA（§4, Algorithm 1–2）

- **层次化提示-动作优化（§4.1）：** 在每个动作块（action chunk）边界先于语言空间搜索最优改写 $l^* = \arg\max_{l' \in \mathcal{L}_r} \mathbb{E}_{a \sim \pi(\cdot \mid o_t, l')}[r(o_t, a, l)]$，再于该分布内采样 $M$ 个候选动作块并选出验证得分最高者 $a_t^* = \arg\max_{j \in [M]} \mathcal{V}_\theta(o_t, h_t, l^*, a'_j)$，其中 $h_t \in \mathcal{A}^W$ 为近期历史动作窗口。
- **CoVer 离线对比学习训练（§4.2, Algorithm 1）：**
  - **两阶段流程：** （1）保持观测 $o_t$ 与真实动作 $(h_t, a_t)$ 不变，利用 Open-X Embodiment 改写集合与 GPT-4o 生成+Qwen3-Embedding-0.6B $k$-means 聚类（$8\times \sim 64\times$）扩充指令；（2）以批内双向 InfoNCE 损失训练对比验证器，将同批其他样本作为隐式负例，无需人工失败标注。
  - **模型架构与缩放：** 视觉-文本编码器采用冻结的预训练 SigLIP2（1B 版使用 ViT-L/16-SigLIP2；500M 用 ViT-B/16-SigLIP2；250M 用 ViT-B/16-CLIP）配合文本感知视觉注意力（text-aware visual attention）提取 $\mathbf{F}_{\text{combined}}(o_i, l_i)$；动作编码器为 Transformer，编码历史与未来动作块得到 $\mathbf{A}(h_i, a_i)$；$\ell_2$ 归一化后以余弦内积 $s_{i,j} = \langle \mathbf{f}_i, \mathbf{a}_j \rangle$ 度量对齐度。最终模型为 1B 参数、16× 合成指令的 Bridge V2（2,000 万样本）、batch size 32,768、8× NVIDIA H200 训练 2,000 步；部署使用 3 个随机种子模型的集成（嵌入取均值后算相似度）。
- **CoVer-VLA 测试时验证管线（§4.3, Algorithm 2, §8.9–8.10）：**
  - **启动时计算（Boot-time Compute）：** 回合开始时仅调用一次 VLM（GPT-4o，约 11 s 生成 8 条改写）对初始画面 $o_0$ 做场景描述、词法替换推理并缓存改写文本及其嵌入，避免在线控制回路阻塞。附录 §8.9 对比表明 VLM 改写能结合视觉颜色/属性锚定目标（如将 Redbull 改写为 "blue can"），避免纯文本 LLM 产生 "beverage" 类语义漂移。
  - **在线批处理验证：** 每步对 $K$ 条缓存改写各采样 $M$ 个动作块（默认 $K=8, M=5$），用 $\mathcal{V}_\theta$ 针对**原始用户指令** $l'$ 统一打分 $s_{k,j} = \mathcal{V}_\theta(o_t, h_t, l', a'_{k,j})$；先对各改写求均分 $S_k = \frac{1}{M}\sum_j s_{k,j}$ 选定 $k^* = \arg\max_k S_k$，再从中选最高分动作块 $j^* = \arg\max_j s_{k^*, j}$ 执行。

### 3. 实验与消融核对（§5, §8.1–§8.7）

- **验证器缩放与判别（§5.1, §8.4–8.5）：** Top-1 动作检索准确率随指令扩增（$8\times \to 64\times$）、模型规模（250M → 500M → 1B，文本编码器从 40M 扩至 280M 是关键）、批大小（2,048 → 8,192 → 32,768）、训练步数与集成数量（1 → 8）持续提升。在二分类测试上达到 Precision 0.765、Recall 0.780、F1 0.772；在动作样本数消融中（Table 6），随样本数从 1 增至 16，动作 RMSE 从 0.166 降至 0.147（下降 11%）。
- **SIMPLER（ERT 对抗指令，Table 3）：** 基于 $\pi_0$，ID 4 任务均值：$\pi_0$ 41.5%、$\pi_0$(rephrase) 44.0%、$\pi_0$+随机改写 42.3%、RoboMonkey 7.5%、$\pi_0$+CoVer 57.0%、$\pi_0$(rephrase)+CoVer 65.5%；OOD 3 任务均值：$\pi_0$ 29.7%、$\pi_0$(rephrase) 48.7%、$\pi_0$+随机改写 28.7%、RoboMonkey 24.7%、$\pi_0$+CoVer 61.0%、$\pi_0$(rephrase)+CoVer 62.0%。
- **PolaRiS（DROID，$\pi_{0.5}$，50 回合 × 3 种子，Table 1）：** 三任务平均任务进度从 40.0±6.4% 提升至 53.9±11.7%（+13.9%），平均成功率从 3.8±4.9% 提升至 13.1±14.1%（+9.3%）。
- **WidowX 真机（§5.5, Figure 9）：** 在 `pepto bismol on plate` 与 `redbull on plate` 两任务上相对基线 $\pi_0$(rephrase) 分别提升 30% 与 60% 成功率（平均绝对提升 45%），而基础 $\pi_0$ 在多数场景几乎无法启动有效动作（0% 成功）。
- **计算与时延（Table 2, Table 5, Table 7）：** 由于冻结 SigLIP2 视觉/文本编码器省去大骨干反向传播（反向 $\approx 1.0 \times 10^9$ vs 前向 $\approx 3.3 \times 10^{11}$ FLOPs），训练 CoVer 仅需 $1.3 \times 10^{20}$ FLOPs（基座 $\pi_0$ 的 3.8×），远低于对 $\pi_0$ 做 16× 指令扩增微调的 $5.4 \times 10^{20}$ FLOPs（16.0×）。推理时图像-文本编码与策略前向并行，CoVer 动作编码器在 BS=1–32 仅增加 7–8 ms（RTX 5090 上 $\pi_{0.5}$+CoVer 在 BS=16 为 453 ms ≈ 2.2 Hz，BS=32 为 873 ms；LeRobot $\pi_0$+CoVer 在 BS=32 为 756 ms ≈ 1.3 Hz）。

### 对 wiki 的映射

- [Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment](../../wiki/entities/paper-agentic-2602-12281.md)
- [官方仓库归档：cover-vla/cover-vla](../repos/paper-agentic-2602-12281.md)
