# APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies（arXiv:2606.12366）

> 来源归档（paper）

- **标题：** APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies
- **作者：** Kechun Xu, Zhenjie Zhu, Anzhe Chen, Rong Xiong, Yue Wang（浙江大学；Rong Xiong 另署浙江人形机器人创新中心）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.12366>（v1，2026-06-10，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.12366v1>
- **项目页：** <https://xukechun.github.io/papers/APT>（文中只给出此链接，未见 GitHub 仓库地址，故页面不写 `code:`）
- **入库日期：** 2026-11-12
- **一句话说明：** 把连续动作 VLA 的动作专家先在「视觉-动作对」上预训练成与语言无关的 VA 先验（冻结 VLM），再用门控融合注入语言做第二阶段，用于提升未见指令与组合任务的泛化。
- **核对范围：** 下载 arXiv 2606.12366v1 PDF（首页标题与编号已核对），通读正文第 1–5 节（引言、相关工作、方法、仿真与真机实验、结论与局限）、全部正文图表（Table 1–3、Figure 4–7），以及附录 A–E（实现细节、视觉捷径分析、基线、仿真与真机细节、Table 4–8）。参考文献列表只扫读未逐条核对；Figure 14 的补充案例图只读了正文描述；未运行任何代码，未找到公开代码。OpenReview 页面本环境打不开，未读。
- **通读：** 已完成

## 开源状态

- 文中只给出项目页，未给仓库地址；本次未核实代码是否开源。

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **问题陈述（§1、§3.1）：** 连续动作 VLA 的动作专家随机初始化、只能从语言远少于视觉-动作多样性的 VLA 数据里学（一条轨迹的 T 个视觉-动作帧共用一条指令），易走「视觉捷径」，并向 VLM 回传噪声梯度。
2. **贝叶斯分解（§3.2，式 2）：** π(a|v,ℓ) ∝ π^p(a|v)·L(ℓ|v,a)；VA 先验只用视觉-动作对训练。训练分三段：预训练 Stage 1（VA 先验，VLM 全冻结，只用一半预训练数据）→ Stage 2（加入语言，全部预训练数据，联合训练）→ 任务微调。
3. **动作专家设计（§3.3、附录 A）：** Qwen3-VL-2B-Instruct 作 VLM；动作专家 N=20 层 Transformer 扩散模型，层级 VLM 特征门控融合 h_in^(i+1) = h_out^(i) + σ(ŵ_i)·φ_i(v,ℓ)；Stage 1 只激活 N/2 层并屏蔽语言 token，Stage 2 在每层后插入新层扩展到 N 层，原 N/2 层从 Stage 1 初始化。与 BayesVLA 不同，APT 在 Stage 2 联合优化全部层（§3.3）。
4. **预训练数据（附录 A）：** DROID、AgiBotWorld-Alpha、InternData-A1、InternVLA-M1，采样权重 5:5:4:1；各阶段各 100k 迭代，batch 256，DDPM 100 步训练、DDIM 20 步推理，动作块长 32，动作为相机系下 10 维（3 平移 + 6D 旋转 + 夹爪）。
5. **LIBERO-PRO（§4.1.3，Table 1，成功率 %）：** 8 个子套件平均 APT 19、APT(Ft VLM) 27，π0.5 为 11、LangForce 14；OpenVLA 与 π0 全 0。APT 在 Object-Pos（7 对 17）、Goal-Pos（23 对 38，Ft VLM 为 10）、Long-Pos（6 对 8，Ft VLM 为 12）低于或不高于 π0.5；作者自述 Goal-Pos 主要败于避障。
6. **刚性物体抓放仿真（§4.1.3，Table 2，SO/UO/UC/UOUE）：** π0 42/30/26/16；π0.5 84/70/86/50；APT 行：仅 Ft VLM 88/56/66/34，仅 KI 90/58/40/40，KI+2-Stage 96/74/90/62，2-Stage+Ft VLM 98/84/92/58。（π0.5 行的勾选列在文本抽取中有歧义，文字只说它用 KI 并联合 VL 数据共训。）基准按 [6] 在 IsaacSim 搭建，UR5，10k 轨迹 500 场景（附录 D.1）。
7. **架构通用性与消融（§4.1.4，Figure 4、5）：** 两阶段训练对 π 式与 GR00T 式动作专家也大多有增益，对 GR00T 式与 APT 更明显；去掉大规模预训练、改用 Token Insertion 注入语言都更差（UO、UOUE 差距最大）。具体数值只在柱状图里，本档未逐项抄录。
8. **真机（§4.2、附录 E，Agilex Cobot + Piper 臂，每任务 30 条遥操作演示）：** 抓放 110 次：π0.5 63/110，APT 90/110（Table 3、7）；杂乱场景推-抓-放 80 次：π0.5 43/80，APT 60/80（Table 3、8）。组合任务：已见语言逐条指令二者相当；单条拼接指令（task chaining）下 T1、T2、T1+T2 成功率 π0.5 为 20、10、0，APT 为 90、70、65（Figure 6，读自柱状图，每组 20 次试验）。
9. **补充仿真（附录 D.2、D.3）：** 原 LIBERO 平均 96.1（π0.5 96.9，Table 4），LIBERO-Plus 总分 71.6（X-VLA 71.4，π0 53.6，Table 5）。作者自述这些不是用来宣称优越。
10. **作者自述局限（§5）：** 未显式建模长程记忆，难以跟踪多步进度；评测只限桌面操作，运动 / 移动操作未探索。

## 论文内部不一致或不明处（本次核对发现）

- 杂乱抓放真机：Table 3 的 UC 列为 18/30 与 22/30，而附录 E.2 写「SO 之后其余 50 次，UC、UO、UOUE 各 10 次」，与 UC 共 30 次对不上（Table 3 四列总和 80 次与 Table 8 的 43/80、60/80 一致）。
- 附录 B 题为「视觉捷径分析」，实际是在「P(ℓ|v) 近似 one-hot」假设下的信息论推导，不是对训练出的策略测量捷径程度的实验。

## 对 wiki 的映射

- [paper-apt-action-expert-pretraining](../../wiki/entities/paper-apt-action-expert-pretraining.md)
