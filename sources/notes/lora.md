# lora

> 来源归档（ingest + primary reading）：**LoRA（Low-Rank Adaptation）** 参数高效微调方法的一手论文精读与官方源码核对归档，用于支撑 wiki 概念页对该术语的定义、机制分析、实验依据与工程实践边界。

- **入库日期：** 2026-07-22
- **精读更新：** 2026-11-14
- **沉淀到 wiki：** 是 → [`wiki/concepts/lora.md`](../../wiki/concepts/lora.md)
- **核对范围：** 全文通读 Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*（arXiv:2106.09685v2，正文 §1–§7 及附录 A–H），并逐文件核对官方仓库 [microsoft/LoRA](https://github.com/microsoft/LoRA)（`README.md`、`loralib/layers.py`、`loralib/utils.py`、`examples/NLG/src/model.py`）。
- **通读：** 已完成

## 一句话说明

LoRA 冻结预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，只训练其旁路低秩分解增量 $\Delta W = B A$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$），前向从 $W_0 x$ 变为 $W_0 x + \frac{\alpha}{r} B A x$，在推理阶段可将 $\frac{\alpha}{r} B A$ 离线合并回 $W_0$ 实现零额外延迟部署。

## 一手论文核对摘要（arXiv:2106.09685v2）

### 1. 核心公式与参数化（§4.1）

- **低秩前向公式**：对预训练线性层权重 $W_0 \in \mathbb{R}^{d \times k}$，约束其更新为两个低秩矩阵乘积，并以常数缩放因子 $\alpha$ 除以秩 $r$ 进行幅度归一化：
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
- **初始化保证平滑起点**：论文 §4.1 采用 $A \sim \mathcal{N}(0, \sigma^2)$ 随机高斯初始化、$B = 0$ 全零初始化，使训练第一步 $\Delta W = B A = 0$；引入 $\frac{\alpha}{r}$ 缩放在改变 $r$ 时有助于保持激活/梯度量级相对稳定，从而减少对学习率的重调。
- **推理无额外开销**：部署前直接计算 $W' = W_0 + \frac{\alpha}{r} B A$，前向拓扑与原始线性层完全同构；若需切换下游任务，仅减去旧任务 $\frac{\alpha}{r} B A$ 并加上新任务矩阵即可热切换。

### 2. 系统资源收益与推理延迟（§2, §5.4, Table 1）

- **GPT-3 175B 显存、存储与吞吐**：训练显存从全量微调（Adam 优化器状态+激活）的 **1.2 TB** 降至 **350 GB**（显存最多节省约 2/3）；只对 $W_q, W_v$ 插 $r=4$ 旁路时，单个下游任务检查点从 **350 GB** 缩至 **35 MB**（约 **10,000×** 存储压缩）；训练吞吐从 **32.5 tokens/s** 提升至 **43.1 tokens/s**（**+25%**，因绝大多数参数无需计算梯度与优化器步进）。
- **Table 1 推理延迟对比（GPT-2 medium，单卡 batch=1）**：在极低批大小/短序列（`seq=128`）实时解码场景，串行瓶颈型 Adapter 带来显著额外延迟（`AdapterL` **+20.7%**、`AdapterH` **+30.3%**；`seq=512` 时分别仍为 +2.9% 与 +4.1%），而 LoRA 合并权重后额外推理延迟为 **0%**。

### 3. 主实验结果与关键消融（§5, §7, Appendix E/F/H）

- **跨架构对比（Table 2–4）**：在 RoBERTa base/large（GLUE）、DeBERTa XXL（1.5B）、GPT-2 medium/large（E2E NLG / WebNLG / DART）及 GPT-3 175B（WikiSQL / MNLI-m / SAMSum）上，LoRA 与全量微调持平甚至更优（例如 GPT-3 上 WikiSQL **74.0 vs FT 73.8**，MNLI-m **91.7 vs FT 89.5**）。
- **权重矩阵选择与参数预算分配（Table 5，GPT-3 175B，固定约 18M 可训练参数预算）**：
  - 仅适配 $W_q$（$r=8$）：WikiSQL **70.4** / MNLI **91.0**；
  - 仅适配 $W_v$（$r=8$）：WikiSQL **73.0** / MNLI **91.0**；
  - 同时适配 $\{W_q, W_v\}$（$r=4$）：WikiSQL **73.7** / MNLI **91.3**；
  - 同时适配 $\{W_q, W_k, W_v, W_o\}$（$r=2$）：WikiSQL **73.7** / MNLI **91.7**。
  - **结论**：在相同参数预算下，将低秩容量分散到多种投影矩阵（优先 $W_q, W_v$ 或全部注意力投影）显著优于把高秩单押在单一矩阵上。
- **秩 $r$ 极低饱和（Table 6 & Appendix H.2）**：仅 $r=1$ 作用于 $\{W_q, W_v\}$ 即在 GPT-3 上达到 WikiSQL **73.4** / MNLI **91.3**；Appendix H.2 GPT-2 E2E 上 BLEU 在 $r=4$ 达峰值 **70.38**（$r=16$ 时验证损失最低 **1.16**），证明适配微调的“更新本征秩”远低于权重矩阵全秩。
- **子空间分析与特征放大效应（§7.3–7.4, Eq. 4, Table 7, Appendix H.4）**：
  - 定义归一化子空间 Grassmann 相似度 $\phi(A_{r=8}, A_{r=64}, i, j)$，实测不同 $r$ 学到的主右奇异方向在最顶部方向重合度超过 **0.5**。
  - 计算特征放大因子 $\|\Delta W_q\|_F / \|U^T W_q V^T\|_F$（其中 $U, V$ 为 $\Delta W_q$ 的左右奇异向量）：$r=4$ 时该比值高达 **$6.91 / 0.32 \approx 21.5$**（$r=64$ 时降为约 2）。说明 $\Delta W$ 并非重复 $W_0$ 原本最强的主奇异方向，而是**放大了预训练权重中存在、但未在下游任务主方向中凸显的特定子空间**。
- **低数据与 Prefix 组合（Appendix E & F.3）**：低数据 MNLI-100（每类仅约百条样本）上 LoRA 达 **63.8** > 全量微调 **60.2** >> PrefixEmbed **37.6**；在 GPT-3 WikiSQL 上，LoRA 与 Prefix-Embedding 正交组合（`LoRA+PE`）可达 **76.2**（优于单用 LoRA 的 73.7），而 `LoRA+PL`（Prefix-Layer）略微下降。

### 4. 一手论文指出的局限（§4.2 Footnote 6 & §7）

- **批内多适配器混批受限**：一旦为追求零延迟将 $BA$ 合并进 $W_0$，同一 forward batch 内若混有对应不同任务 LoRA 的样本，则无法复用单一合并权重，需要回退为不合并权重并按样本分组/掩码计算或以切换延迟换取灵活性。
- **大分布偏移下的低秩上限**：尽管多数组合下 $r=1\sim 8$ 足够，但在与预训练语料/模态截然不同的任务（如切换全新语言或剧烈动力学变化）上仍可能需要更大秩甚至全量微调。
- **层位与块选型依赖经验消融**：哪些投影矩阵（Q/K/V/O 或 MLP/Decoder）最受益仍需面向任务实证验证。

## 官方源码核对摘要（`microsoft/LoRA`）

- **`loralib/layers.py` 核心实现**：
  - `LoRALayer` 基类定义公共超参 `r`、`lora_alpha`、`lora_dropout`（`nn.Dropout(p)` 或 `Identity`）、`merge_weights`、`merged` 及 `scaling = lora_alpha / r`。
  - 提供四类层封装：`Embedding`、`Linear`、`MergedLinear`（针对多头注意力将 QKV 合并为单个大矩阵的层，如 GPT-2 的 `c_attn`）、以及 `ConvLoRA`（派生 `Conv1d`、`Conv2d`、`Conv3d`，将低秩矩阵重塑为等效卷积核）。
  - **初始化细节（源码 vs 论文差异）**：论文 §4.1 描述 $A$ 采用随机高斯分布、$B$ 全零；而源码 `loralib/layers.py` 中 `Linear`、`MergedLinear`、`ConvLoRA` 实际对 `lora_A` 使用 **`nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))`**、对 `lora_B` 使用 `nn.init.zeros_(self.lora_B)`；仅 `Embedding` 使用 `zeros_(lora_A)` 与 `normal_(lora_B)`。两者均严格保证初始 $\Delta W = B A = 0$。
  - `MergedLinear` 掩码分块逻辑：通过 `enable_lora`（如 `examples/NLG/src/model.py` 中 GPT-2 `Attention` 对 `c_attn` 设置 `enable_lora=[True, False, True]` 只适配 $Q$ 和 $V$、跳过 $K$）配合 `F.conv1d(..., groups=sum(enable_lora))` 实现分块独立低秩旁路，并用 `lora_ind` 布尔掩码在 `merge()` / `train(mode)` 时精确叠加或扣除权重增量。
- **`loralib/utils.py` 训练与导出接口**：
  - `mark_only_lora_as_trainable(model, bias='none'|'all'|'lora_only')`：按参数名中是否包含 `"lora_"` 冻结原模型主权重，支持不训练偏置、仅训练 LoRA 层偏置或训练全部偏置三种模式。
  - `lora_state_dict(model, bias=...)`：仅提取含 `"lora_"`（及可选 `"bias"`）的状态字典，直接产生 KB–MB 级可插拔增量检查点。
- **`examples/NLG/src/model.py` 模型集成模式**：在 `Attention` 与 `MLP`（若启用）中配置 `lora_attn_dim`、`lora_attn_alpha`、`lora_dropout`；在多适配器加载场景注释了可显式传入 `merge_weights=False` 以避免混批冲突。

## 为什么值得保留

本库多篇 VLA / world-model / 视频微调与具身控制页面（FADA、Any2Any、SONIC-Transfer、M4World、WAM-TTT、RLDX-1、MorphAct 等）均把 LoRA 作为**动力学敏感模块的低成本适配算子**或**跨具身补丁**直接引用；通过补齐论文公式、消融证据与 `loralib` 源码语义，各页回链概念页时可统一对齐初始化差异、推理合并边界、多任务混批约束及预算分配策略。

## 可引用来源

- 原始论文：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685v2（2021）。<https://arxiv.org/abs/2106.09685>
- 官方实现：microsoft/LoRA — <https://github.com/microsoft/LoRA>（核心文件：`loralib/layers.py`、`loralib/utils.py`、`examples/NLG/src/model.py`）
- 常见衍生：QLoRA（4-bit 量化基座 + LoRA）、DoRA（权重幅值-方向分解）、Safe LoRA 投影微调。

## 库内引用页面

- [`wiki/methods/mimic-video.md`](../../wiki/methods/mimic-video.md)
- [`wiki/entities/paper-fada-humanoid.md`](../../wiki/entities/paper-fada-humanoid.md)
- [`wiki/entities/paper-any2any-cross-embodiment-wbt.md`](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)
- [`wiki/entities/paper-sonic-transfer.md`](../../wiki/entities/paper-sonic-transfer.md)
- [`wiki/entities/paper-m4world.md`](../../wiki/entities/paper-m4world.md)
- [`wiki/entities/paper-wam-ttt-human-video-test-time-steering.md`](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)
- [`wiki/entities/rldx-1.md`](../../wiki/entities/rldx-1.md)
- [`wiki/entities/paper-morphact.md`](../../wiki/entities/paper-morphact.md)
- [`wiki/formalizations/safe-lora-update-projection.md`](../../wiki/formalizations/safe-lora-update-projection.md)
