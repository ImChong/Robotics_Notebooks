# Mamba 与 S4 一手论文精读与源码核对归档（arXiv:2312.00752 & arXiv:2111.00396）

> 来源归档（ingest + primary reading）：Gu & Dao (2023) *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*（arXiv:2312.00752v2）、Gu, Goel & Ré (2022) *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*（arXiv:2111.00396v3）以及官方仓库 [state-spaces/mamba](https://github.com/state-spaces/mamba) 一手核对记录。

- **入库日期：** 2026-11-14
- **沉淀到 wiki：** 是 → [`wiki/concepts/mamba.md`](../../wiki/concepts/mamba.md)
- **核对范围：** 全文通读 Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*（arXiv:2312.00752v2，正文 §1–§6 及附录 A–E）与前驱论文 Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces*（arXiv:2111.00396v3，正文 §1–§5 及附录 A–D），并逐文件核对官方仓库 [state-spaces/mamba](https://github.com/state-spaces/mamba)（`README.md`、`mamba_ssm/modules/mamba_simple.py`、`mamba_ssm/ops/selective_scan_interface.py`）。
- **通读：** 已完成

## 一句话说明

Mamba 在 S4 结构化状态空间的基础上解除线性时不变（LTI）限制，令步长 $\Delta_t$ 与投影矩阵 $B_t, C_t$ 成为当前输入 $x_t$ 的函数（S6 选择性扫描），并依靠 GPU SRAM 核融合、工作高效并行关联扫描与反向重算消除高带宽显存 IO 瓶颈，在语言、DNA 与音频长序列上以线性时间复杂度与 $O(1)$ 单步推理状态匹配同级 Transformer 质量。

## 一手论文核对摘要

### 1. 从 S4（arXiv:2111.00396v3）到选择性瓶颈

- **连续系统与双线性离散化（S4 Eq. 1–3）**：连续状态空间 $x'(t) = A x(t) + B u(t), y(t) = C x(t) + D u(t)$ 结合 HiPPO-LegS 连续记忆矩阵（Eq. 2）实现长程记忆；按步长 $\Delta$ 做双线性（bilinear）离散化后得到递推形式 $x_k = \bar{A} x_{k-1} + \bar{B} u_k$，并因线性时不变（LTI）可等价展开为长度 $L$ 的全局卷积核 $K = (C\bar{B}, C\bar{A}\bar{B}, \dots, C\bar{A}^{L-1}\bar{B})$。
- **S4 核心技巧**：直接对 HiPPO 矩阵对角化存在指数级病态数值放大（Lemma 3.2 证得其对角化矩阵元素达 $V_{3i,i} \approx 2^{4N/3}$）；S4 将 $A$ 分解为正规阵加低秩项（NPLR/DPLR：$A = V\Lambda V^* - P Q^T$），在频域用单位根求截断生成函数 + Woodbury 恒等式化简低秩项 + Cauchy 核加速与逆 FFT，将卷积核计算与单步递推复杂度降至 $\tilde{O}(N+L)$ 与 $O(N)$。
- **S4 成效与根本局限**：S4 在 LRA 六项均值达 **86.09%**（首次攻克 16k 长度 Path-X，达 **96.35%**）、WikiText-103 测试困惑度 20.95 且自回归吞吐比 Transformer 快 60×；但因参数 $(\Delta, A, B, C)$ 对所有时间步恒定（LTI），**无法依据输入内容决定写入或遗忘信息**，在语言等离散高信息密度模态仍落后 Transformer。

### 2. Mamba 选择性状态空间（S6，arXiv:2312.00752v2 §3）

- **状态压缩视角与合成任务诊断（§3.1）**：Attention 通过保存完整 KV cache 实现无损检索但带来 $O(n^2)$ 训练与 $O(n)$ 推理状态膨胀；传统循环模型依赖定长有限状态但缺乏内容筛选能力，导致无法解决随机间隔的 **Selective Copying** 或依赖上下文触发复制的 **Induction Heads**。
- **选择机制参数化（§3.2, Algorithm 2）**：保留对角矩阵 $A \in \mathbb{R}^{D \times N}$ 独立作用各通道，改为令 $B_t, C_t, \Delta_t$ 随输入 $x_t$ 变化（引入时间维 $L$）：
  - $B = s_B(x) = \mathrm{Linear}_N(x) \in \mathbb{R}^{B \times L \times N}$
  - $C = s_C(x) = \mathrm{Linear}_N(x) \in \mathbb{R}^{B \times L \times N}$
  - $\Delta = \tau_\Delta(\text{Parameter} + s_\Delta(x)) \in \mathbb{R}^{B \times L \times D}$，其中 $\tau_\Delta = \mathrm{softplus}$，$s_\Delta(x) = \mathrm{Broadcast}_D(\mathrm{Linear}_1(x))$（或更一般的低秩投影 $\mathrm{Linear}_D(\mathrm{Linear}_R(x))$）。
  - **ZOH 零阶保持离散化（Eq. 4）**：$\bar{A}_t = \exp(\Delta_t A)$，$\bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I)\cdot \Delta_t B_t$（实际实现在 $\bar{B}_t$ 常采用简化近似 $\Delta_t B_t$ 或精确展开）。
- **Theorem 1 与物理机理（§3.5 & Appendix C）**：当 $N=1, A=-1, B=1, s_\Delta=\mathrm{Linear}(x), \tau_\Delta=\mathrm{softplus}$ 时，S6 ZOH 离散化严格等价于门控 RNN：
  $$g_t = \sigma(\mathrm{Linear}(x_t)), \qquad h_t = (1 - g_t) h_{t-1} + g_t x_t$$
  由此解释三项核心选择性行为：大 $\Delta_t \to \infty$ 对应重置旧状态并聚焦写入当前输入；小 $\Delta_t \to 0$ 对应忽略当前噪声、持久化历史记忆，实现 **Variable Spacing**（跳过冗余词）、**Filtering Context**（长上下文不稀释有效信息）与 **Boundary Resetting**（多文档打包或 RL 回合边界处一键清零状态）。

### 3. 硬件感知并行扫描与简化同构块（§3.3–§3.4 & Appendix D）

- **突破 LTI 缺失卷积的算力挑战**：尽管时变动力学失去了全局 FFT 卷积等价性，但在现代 GPU 上算子瓶颈主要在于 **HBM ↔ SRAM 内存带宽**，而非 FLOPs（$O(BLDN)$ 在中等 $N$ 下常数更小）。
- **三项关键优化**：
  1. **Kernel Fusion**：仅从慢速 HBM 读入尺寸 $O(BLD + DN)$ 的 $(\Delta, A, B, C)$，在高速片上 SRAM 内完成离散化 $(\bar{A}, \bar{B})$、扫描递推（生成局部 $(B, L, D, N)$ 隐状态但不写回 HBM）并与 $C$ 乘加汇总为 $(B, L, D)$ 输出再写回 HBM，减少 $O(N)$ 倍访存 IO（序列过长时切块 chunk scan 并传递边界状态）。
  2. **Parallel Associative Scan**：利用线性时变一阶递推满足结合律的性质，以 Blelloch work-efficient 并行前缀扫描消除串行瓶颈。
  3. **Recomputation**：反向传播不从 HBM 读取 $(B, L, D, N)$ 中间态，而是在 SRAM 内即时重算，显存占用对齐 FlashAttention-2。
- **同构 Mamba Block（§3.4, Fig. 3）**：把 H3 序列块与 SwiGLU/Gated MLP 合并为单一均质结构——以扩展因子 $E=2$ 把隐藏维 $D$ 映射至 $ED$ 两条分支：主分支依次经过短因果 `Conv1d`、激活 $\sigma = \mathrm{SiLU}$ 与 `S6`，门控分支经 $\mathrm{SiLU}$ 后与主分支逐元素相乘，最终经输出投影回到 $D$（单块投影参数 $3ED^2$，双块叠加 $12D^2$ 对齐 Transformer 的 MHA+MLP）。

### 4. 主实验结果、消融与局限（§4–§5）

- **合成任务（Table 1–2）**：Selective Copying 上 Mamba+S6 达 **99.8%**（Mamba+S4 仅 56.4%，Mamba+Hyena 28.4%）；Induction Heads 在 $L=256$ 训练后零样本外推至 **$2^{20}=1,048,576$（4000×）**仍保持完美精度（注意 Transformer 家族在 16k 全部 OOM，xPos 也无法进一步外推）。
- **语言建模（Fig. 4 & Table 3，The Pile / 300B tokens）**：在 Chinchilla scaling laws 下首次有纯线性时间模型匹敌并随上下文追上强配方 Transformer++（LLaMA 式 RoPE+SwiGLU+RMSNorm）；下游零样本评估中 **Mamba-2.8B** 平均准确率 **63.3%** 超过同规模 Pythia-2.8B（59.1%）与 RWKV-3B（59.6%），并可比肩两倍规模的 Pythia-6.9B（61.7%）。
- **DNA 与音频（§4.3–§4.4）**：人类基因组 HG38 上控制训练总计算量时，随上下文从 1K 增至 1M，Mamba 困惑度持续下降（HyenaDNA 在长上下文反而退化）；SC09 语音生成 24.3M 参数 Mamba 达 **FID 0.67**（SaShiMi 1.42，DiffWave 1.92）；注：YouTubeMix 连续音频预训练（§4.4.1）是全篇唯一改用**复数** SSM 更优的任务。
- **速度与吞吐（§4.5 & App. E.5）**：A100 80GB 上融合扫描比朴素 PyTorch 实现快 **20–40×**，在序列长度超过 **2K** 后快于 FlashAttention-2；因无 KV cache 可使用更大批大小，推理吞吐达同参数 Transformer 的 **4–5×**。
- **关键消融（Table 6–10）**：选择性 $\Delta$ 单项带来最大增益（10.93 → 10.15 ppl），三者联合协同降至 **8.71 ppl**；开启 S6 后实数对角初始化 **S4D-Real**（$A_n = -(n+1)$，8.71 ppl）反超复数 S4D-Lin（9.16 ppl）；开启选择性 $B,C$ 后将状态维 $N$ 从 1 扩到 16 仅增加约 1% 参数即降低 **>1.0 ppl**（9.73 → 8.71），而在非选择性 $B,C$ 下扩大 $N$ 几乎无效（9.88 → 9.81）。
- **论文局限（§5）**：连续-离散谱的“无免费午餐”权衡（连续感知信号常仍受益于复数或 LTI 偏置）；生态成熟度与下游工程工具链（如微调、量化、RLHF、ICL 工具）相对 Transformer 起步更晚；论文实验上限至 ~2.8B（未在 7B+ 实证工程扩展表现）。

## 官方源码核对摘要（`state-spaces/mamba`）

- **`mamba_ssm/modules/mamba_simple.py` 逐模块核对**：
  - `Mamba.__init__` 默认 `d_state=16`、`d_conv=4`、`expand=2`、`dt_rank="auto"`（自动设 $\lceil d_{\text{model}} / 16 \rceil$）。
  - `in_proj` 输出 `2 * d_inner` 切分为 `(x, z)`；`conv1d` 采用深度可分离 `groups=d_inner`（`padding=d_conv-1` 并在前向切片 `[:seqlen]` 维持因果）；`x_proj` 联合线性映射出 `(dt, B, C)`（维度 `dt_rank + 2 * d_state`）；`dt_proj` 把 `dt_rank` 升维至 `d_inner`，其偏置初始化对应 $\tau_\Delta^{-1}(\mathrm{Uniform}([0.001, 0.1]))$ 并标记 `dt_proj.bias._no_reinit = True`。
  - 状态矩阵参数化为对数形式 `A_log = nn.Parameter(torch.log(arange(1, d_state+1)))`（保持 `float32` 存储并标记 `_no_weight_decay = True`，前向取 `A = -torch.exp(self.A_log.float())` 保证稳定负半平面衰减）；跳跃连接向量 `D = nn.Parameter(torch.ones(d_inner))` 同样禁用权重衰减。
  - **训练快路径 vs 慢路径**：当无缓存且开启 `use_fast_path=True` 并存在 CUDA 扩展时，调用融合核 `mamba_inner_fn`（`selective_scan_interface.py` 中 `MambaInnerFn`，`checkpoint_lvl=1` 反向重算 `conv1d_out` 与 `delta`）；否则回退分步执行 `causal_conv1d_fn` + `selective_scan_fn` 或纯 PyTorch 参考实现 `selective_scan_ref`。
  - **单步自回归推理 `step()`**：读取并原地更新 `inference_params` 中的固定滚动缓存 `conv_state`（长度 `d_conv`）与 `ssm_state`（维度 `(B, d_inner, d_state)`），在配备 Triton 时自动调度 `causal_conv1d_update` 与 `selective_state_update` 算子实现恒定 $O(1)$ 单步内存与计算。
- **`README.md` 工程与精度提示**：模型主权重必须保持 **FP32** 并配合 PyTorch AMP（BF16/FP16）混合精度训练（文档明确警告 DeepSpeed 纯 FP16 等直接半精度主权重方案会影响数值稳定性）；源码构建支持 `MAMBA_FORCE_BUILD=TRUE` / `MAMBA_KEEP_CUDA_BUILD=TRUE`。

## 可引用来源

- Mamba 原论文：Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752v2（2023/2024）。<https://arxiv.org/abs/2312.00752>
- S4 前驱论文：Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces*, arXiv:2111.00396v3（2022）。<https://arxiv.org/abs/2111.00396>
- 官方代码：state-spaces/mamba — <https://github.com/state-spaces/mamba>（核心文件：`mamba_ssm/modules/mamba_simple.py`、`mamba_ssm/ops/selective_scan_interface.py`、`README.md`）
- 关联二手索引：[`sources/papers/gu_mamba_arxiv_2312_00752.md`](./gu_mamba_arxiv_2312_00752.md)、[`sources/papers/gu_s4_arxiv_2111_00396.md`](./gu_s4_arxiv_2111_00396.md)、[`sources/papers/ai_architecture_foundations.md`](./ai_architecture_foundations.md)
