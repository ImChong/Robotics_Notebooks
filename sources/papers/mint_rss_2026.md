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
- **入库日期：** 2026-07-03
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
