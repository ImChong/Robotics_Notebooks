# Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers

- **作者：** Frederic Vatnsdal, Roshan Gopal, Romina Garcia Camargo, Vijay Kumar, Alejandro Ribeiro
- **机构：** University of Pennsylvania（Department of Electrical and Systems Engineering / School of Arts and Sciences / GRASP Laboratory）
- **版本 / 载体：** arXiv:2609.28247v1 [cs.RO]，2026 年 9 月 23 日提交（CC BY 4.0，arXiv HTML 版）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2609.28247) · [arXiv 摘要页](https://arxiv.org/abs/2609.28247)
- **项目页 / 代码：** 未确认开源代码（论文全文及 arXiv 页面均未提供官方代码仓库或独立项目页链接；截至 2026-11-27 检索 `candidate_repos` 为空）
- **核对范围：** 完整通读 arXiv HTML 全文（Abstract、Section I–V、Table I–III、Figure 1–7 说明及 References）
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers](../../wiki/entities/paper-agentic-2609-28247.md)

### 核心方法与公式核对

1. **问题设定（Section I / II）：** 面向规模高达 $N=1024$ 的双积分器机器人集群自然语言集群飞行（flocking），提出去中心化推理空间反馈控制架构 **COMPASS**（COmmanding Multi-robot Policies with Attention and Semantic Steering）。
2. **双时间尺度程序与推理回路（Section II-A，式 (1)–(3)）：**
   - 每个机器人携带冻结的小语言模型（默认 Qwen3-1.7B）和持久化嵌入程序 $\mathrm{PROG}_i(t) = [\mathbf{P}_i \,\|\, \mathbf{T}_i \,\|\, \mathbf{f}_i(t) \,\|\, \mathbf{g}]$，其中 $\mathbf{P}_i$ 为操作员提示，$\mathbf{T}_i$ 为语言思考轨迹（prompt 变更时按 $\phi_{\mathrm{gen}}$ 生成 160 tokens 后冻结并做 KV-cache），$\mathbf{f}_i(t) = \psi(\boldsymbol{\pi}_i(t))$ 为 1 Hz 刷新的学习反馈 token，$\mathbf{g}$ 为可学习目标 query token。
   - 潜在推理状态从前 $\ell$ 层（默认 $\ell=27$）末端 $\mathbf{g}$ 位置抽取：$\mathbf{z}_i(t+1) = \phi_{1:\ell}(\mathbf{P}_i \,\|\, \mathbf{T}_i \,\|\, \psi(\boldsymbol{\pi}_i(t)) \,\|\, \mathbf{g})$。因因果掩码及末端位置特性，语言 token 不关注 $\mathbf{f},\mathbf{g}$，两者无需落在自然语言流形上。
3. **多智能体空间 Transformer（MAST，Section II-B，式 (4)–(6)）：**
   - 节点初始特征 $\mathbf{h}_i^{(0)} = \mathrm{MLP}([\mathbf{z}_i \,\|\, \mathbf{x}_i])$，包含推理状态 $\mathbf{z}_i$、自身速度 $\mathbf{v}_i$ 及邻域统计 $\mathbf{s}_i = [\mu_{\mathcal{N}_i}, v_e]$（平均邻居距离与相对邻域平均速度误差）。
   - 第一层使用 2D RoPE（周期 $\lambda=96\text{ m}$）使注意力 logit 仅依赖相对位移 $\mathbf{p}_j - \mathbf{p}_i$，保证平移不变性与置换等变性；通信掩码 $\mathcal{M}_{ij}=1 \iff d_{ij} \le r_c$（$r_c=4\text{ m}$），边特征 $\mathbf{e}_{ij}=[\mathbf{p}_j-\mathbf{p}_i \,\|\, d_{ij}]/r_c$。
   - 4 层 MAST（$d_{\mathrm{MAST}}=256$、8 头 $d_h=32$）提供 4-hop 感受野，10 Hz 运行；输出 $\boldsymbol{\pi}_i$ 经双 MLP 头分别解码低层动作 $\mathbf{u}_i$（10 Hz）与反馈 token $\mathbf{f}_i$（1 Hz）。
4. **训练与数据集（Section III，Table I）：**
   - 以集中式势场（potential-field）专家（速度对齐、目标速度 $5\text{ m/s}$ 跟踪、质心凝聚、碰撞排斥）在 $N=[32,64]$、密度 $0.25\text{ robots/m}^2$、$dt=0.1\text{ s}$、$T=10\text{ s}$ 上生成轨迹，采用 Behavior Cloning + DAgger 重标注 + TBPTT 训练 200 epochs（单卡 RTX A5000 24 GB，AdamW $lr=10^{-4}$，cosine annealing），LM 权重全程冻结。
   - 语料含 7 类训练指令（Plain/Abbreviation/Opposite/Map/Clock 各 1920，Numeric 5760 含 720 个 $0.5^\circ$ 方位角，Hold 21）及 2 类仅测试零样本类（Vague/Conceptual 384，Secondary intercardinals 192）。

### 关键实验数据核对

- **规模扩展性（Section IV-A，Fig. 3–4，$N=8\dots 1024$，30 s）：** 训练于 $N=[32,64]$ 的 COMPASS 在“go east”指令下航向误差随规模增大而下降，从 $N=8$ 的 $2.79^\circ$ 降至 $N=1024$ 的 $0.34^\circ$（16 倍训练规模），全程保持单连通群体（$\mathrm{frags}=1$）与均匀近邻间距；而纯语言通信消融在 $N=64$ 即分裂为 15 个碎片且速度降至 $2.5\text{ m/s}$；集中式 Claude Opus 4.8 随 $N$ 增大无法生成协调加速度且频繁输出错误数量动作。
- **结构化多样性（Section IV-B，Fig. 5）：** 全队同用模糊表述或混用 4 种同难度模糊表述并不能提升准确率；但让 $1/4$ 机器人接收更易解码的直白表述时，MAST 反馈可将全队航向与速度拉回正确意图，且跨规模成立。
- **推理引擎与泛化（Section IV-C，Table II–III，Fig. 6–7）：** Qwen3-0.6B/1.7B/4B 相对误差为 1.00 / 0.80 / 0.65，延迟与显存为 3.7 s (1.2 GB) / 4.0 s (3.3 GB) / 5.0 s (7.7 GB)。$N=64$ 下零样本中位航向误差：训练内模板 $1.0^\circ$、留出模板 $1.9^\circ$、概念/模糊指令 $5.7^\circ$、未见数值方位角 $2.0^\circ$（均值 $2.2^\circ \pm 1.6^\circ$）、16 方位次间方位（zero-shot）$7.7^\circ$（均值 $7.96^\circ \pm 4.41^\circ$，18/24 落入目标扇区，ESE $1.7^\circ \pm 1.0^\circ$、ENE $3.8^\circ \pm 1.8^\circ$、NNE $13.2^\circ \pm 2.7^\circ$）。任务中由“go east”切换至“stop”时存在约 $0.2\text{ m/s}$ 残余漂移。
