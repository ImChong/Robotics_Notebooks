# MINT: Mimic Intent, Not Just Trajectories（RSS 2026 / arXiv:2602.08602）

> 来源归档（ingest）

- **标题：** Mimic Intent, Not Just Trajectories
- **缩写：** **MINT**
- **类型：** paper / vision-language-action / imitation-learning / action-tokenization / generalization
- **会议：** Robotics: Science and Systems（**RSS 2026**）
- **arXiv：** <https://arxiv.org/abs/2602.08602>（PDF：<https://arxiv.org/pdf/2602.08602>）
- **项目页：** <https://renming-huang.github.io/MINT/>
- **代码：** <https://github.com/RenMing-Huang/MINT>
- **权重：** Hugging Face `huangrm/MINT-libero` 等（以项目页为准）
- **作者：** Renming Huang, Chendong Zeng, Wenjing Tang, Jintian Cai, Cewu Lu, Panpan Cai†（上海交通大学、上海创智学院；通讯作者以 PDF 为准）
- **入库日期：** 2026-11-07（首次入库 2026-07-03）
- **版本：** arXiv:2602.08602v3（2026-03-28）；本次 PDF 未标注会议名，RSS 2026 之说沿用旧归档、未在 PDF 内核实
- **核对范围：** 2026-11-07 下载 arXiv v3 PDF（21 页）并通读：摘要、§I 引言、§II 相关工作、§III 概览、§IV SDAT（式 1–3、训练目标、Algorithm 1）、§V MINT 策略（next-scale 自回归、intent-based ensemble 式 5–6、模型结构）、§VI 实验（Table I–IV、LIBERO-Plus、one-shot 迁移、真机、消融）、§VII 结论与局限；参考文献仅浏览；项目页、代码和权重未核对。PDF 未含附录。
- **通读：** 已完成
- **一句话说明：** 用 **频域多尺度动作分词（SDAT）** 把轨迹拆成 **意图 token（低频全局结构）** 与 **执行 token（高频残差）**，再以 **跨尺度自回归** 做意图→执行推理；支持 **单样本意图注入** 迁移，在 LIBERO / MetaWorld / CALVIN / LIBERO-Plus 与真机上报告 SOTA 级成功率、扰动鲁棒性与 one-shot 迁移。

## 摘录 1：问题与动机（摘要 / 引言）

- **痛点：** 端到端 IL 与 VLA 在封闭设定表现亮眼，但对 **环境变化、新任务实例、技能迁移** 泛化不足；根因之一是 **直接模仿原始轨迹信号**，未建模「**为何执行这一动作序列**」的 **行为意图（behavioral intent）**。
- **与表面泛化的区分：** 换物体、换背景、换光照只是 **基础域随机**；真正难的是 **组合泛化**（学会 A、B、C 后能自由组合 A→B、B→C）与 **小样本迁移**（几次示范而非上千条演示）。
- **现有 action tokenization 的缺口：** FAST、BEAST、VQ-VAE 等多作 **压缩** 或保留 **低层运动学**，缺乏 **与可解释意图对齐** 的显式约束；多尺度层次（如 CARP）若在 **时域** 重建，易优先 **局部保真** 而非 **意图–执行结构**。

**对 wiki 的映射：** 与 [VLA](../../wiki/methods/vla.md)「工程瓶颈·数据规模与泛化」及 [Action Tokenization](../../wiki/formalizations/vla-tokenization.md) 并列，把 **频域谱分解 + 渐进重建监督** 作为 **语义化动作分词** 的新轴。

## 摘录 2：SDAT 与 MINT 策略（§III–V 要点）

- **两阶段框架：** (1) **Spectrally Disentangled Action Tokenizer（SDAT）** 在演示轨迹上学 **多尺度离散动作表示**；(2) **MINT policy** 在 token 空间做 **意图→执行** 的 **next-scale autoregression**。
- **SDAT 机制：** 动作块经 **DCT** 转到频域；**多尺度残差 VQ** 得 \(S_1,\dots,S_K\)——**最粗尺度 \(S_1\) 仅 1 个 token**，解释为 **Intent token**；更细尺度为 **Execution tokens**。训练用 **逐尺度频域重建损失** \(\mathcal{L}_{\text{freq.}}\)：先用 \(S_1\) 重建频谱，再用 \(S_1+S_2\)、…、直至全尺度；迫使粗尺度捕获 **低频全局形状**，细尺度专攻 **高频残差**；辅以时域 \(l_1\) 重建保证可执行性。
- **MINT 策略：** 视觉–语言骨干 + **action expert**；在每一尺度内 **并行预测** token map，跨尺度 **自回归**；推理可用 **intent-based action ensemble** 平滑重叠 chunk 的意图一致性。
- **单样本迁移（MINT-Zero）：** 从 **一条演示** 抽取 **Intent token**，在自回归生成中 **固定粗尺度 token**，仅生成执行 token，即可向 **新布局 / 新任务 / 更长视界** 迁移。

**对 wiki 的映射：** 沉淀至 [`wiki/entities/paper-mint-vla.md`](../../wiki/entities/paper-mint-vla.md)；与 [DeFI](../../wiki/methods/defi-decoupled-dynamics-vla.md)（前向/逆向动力学解耦）、[CapVector](../../wiki/entities/paper-capvector-capability-vectors-vla.md)（参数空间能力向量）形成 **表示层 / 参数层 / 动力学层** 不同解耦维度的对照。

## 摘录 3：实验与主要结论（§VI 摘要）

- **基准：** **LIBERO**、**MetaWorld**、**CALVIN**、扰动更强的 **LIBERO-Plus**；对比 **π₀.₅**、**UniVLA**、**OpenVLA-OFT**、**ACT**、**Diffusion Policy** 等。
- **标准榜：** 多基准 **SOTA 级** 成功率；推理效率优于部分生成式 IL 基线（细节以 PDF 表格为准）。
- **鲁棒性：** 在 LIBERO 上训练、**LIBERO-Plus** 评测时，相对最强基线 **OpenVLA-OFT** 成功率约 **+15%**。
- **One-shot：** 意图注入式迁移在 **新任务与新环境** 上相对基线约 **+60%** 迁移性能（论文叙事）。
- **真机：** 每任务约 **20 条演示** 即可有效迁移；相对最强基线 **π₀.₅** 约 **+29%** 成功率。
- **规模：** **MINT-30M**（标准 Transformer 从头训）与 **MINT-4B**（预训练 VLM + 随机初始化 action head）两档。

**对 wiki 的映射：** 实体页表格化 **基准 × 扰动 × 迁移设定**；与 [Manipulation](../../wiki/tasks/manipulation.md) 评测语义互链。

## 通读核对结论（2026-11-07，对照 arXiv v3 全文）

- **SDAT 的结构：** 编码器 $E$ 对**时域**动作序列 $A\in\mathbb R^{H\times D}$ 编码得到 $f\in\mathbb R^{L\times C}$；用 VQ-VAE 式多尺度残差量化（Multi-Scale Residual Quantization，共享码本 $Z$，$K$ 个尺度、分辨率 $l_1<\dots<l_K=L$）得到各尺度 token map $s_k$；累计特征 $\hat f^{(k)}=\sum_{i\le k}\phi_i(\mathrm{Lookup}(Z,s_i))$。**DCT 是用在“重建结果”上的**：谱解码器 $D_{spec}$ 把每个累计重建 $\hat A^{(k)}$ 沿时间维做 DCT 得 $F^{(k)}$，与真值 $F=\mathrm{DCT}(A)$ 比较——并不是先把输入动作变到频域再量化。引言里写“multi-scale VAE”，§III–IV 写 VQ-VAE，二者表述略有出入，以 §IV 的残差量化为准。
- **损失：** $\mathcal L=\mathcal L_{freq}+\|sg(f)-\hat f\|_2^2+\|f-sg(\hat f)\|_2^2+\alpha\|A-\hat A\|_1^2$，其中 $\mathcal L_{freq}=\sum_{k=1}^K\lambda_k\|F-F^{(k)}\|_2$；即逐尺度谱重建 + 码本/承诺损失 + 时域 $\ell_1$ 辅助项。最粗尺度只有 1 个 token（Intent token $S_1$），其余为 Execution tokens。
- **策略：** next-scale 自回归，$p(s_1,\dots,s_K)=\prod_k p(s_k|s_{<k})$；同一尺度内所有 token 并行预测，混合注意力掩码使尺度 $k$ 只看 $\le k$ 的 token map；交叉熵训练。两个版本：MINT-30M（从零训练的 decoder-only Transformer，冻结 SigLIP+DINOv2，BERT 语言经 FiLM 注入）与 MINT-4B（PaliGemma-2.6B + 约 300M 从零训练的动作专家，沿用 π0/π0.5 的 VLM）。语言无关版 MINT-Zero 用于 one-shot 意图注入。
- **Intent-based ensemble（式 5–6）：** 对重叠 chunk 的预测按 intent token 的余弦相似度做 softmax 加权。
- **结果：** LIBERO（Table I）：MINT-30M 平均 97.1 vs 无预训练的 SmolVLA 88.8；MINT-4B 98.3 vs π0.5 96.9；CALVIN ABCD→D 平均长度 4.57（RoboVLMs 4.49，π0.5 4.15）；MetaWorld 平均 67.2（π0 50.8，Very Hard 56.0 vs 20.0）。LIBERO-Plus（只在 LIBERO 上训练）平均：MINT-4B 80.1，OpenVLA-OFT 71.4，π0.5 65.0。one-shot 迁移（Table III）：Intent 注入平均 0.77，对比微调 0.17、回放 0.11。真机（6-DoF Piper-X，每任务 20 条示范、每策略 20 次试验）：作者称相对最强基线（改动版 π0.5，动作头重新初始化）提升 29%；以 Bayesian 后验区分显著性。**消融（Table IV）：** 逐尺度谱损失 LIBERO-Long 93.4 / CALVIN 4.54；逐尺度时域损失反而降到 82.8 / 4.06；intent-based ensemble 93.2 / 4.57 优于 temporal ensemble 89.2、action-based ensemble 90.4。
- **原文自述局限：** 意图的多样性受限于示范数据集范围；未来方向是用网络数据与离散意图 token 重组合成新行为。
- **与旧归档的差别：** 旧归档摘录里“动作块经 DCT 转到频域；多尺度残差 VQ 得 $S_1,\dots,S_K$”应读作“对重建做 DCT 后在频域监督逐尺度重建”。
