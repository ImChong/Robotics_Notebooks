# Lion 一手论文精读与源码核对归档（arXiv:2302.06675）

> 来源归档（ingest + primary reading）：Chen et al. (2023) *Symbolic Discovery of Optimization Algorithms*（arXiv:2302.06675v4）与官方仓库 [google/automl/lion](https://github.com/google/automl/tree/master/lion) 一手核对记录。

- **入库日期：** 2026-11-14
- **沉淀到 wiki：** 是 → [`wiki/methods/lion.md`](../../wiki/methods/lion.md)
- **核对范围：** 全文通读 Chen et al., *Symbolic Discovery of Optimization Algorithms*（arXiv:2302.06675v4，正文 §1–§8 及附录 A–I），并逐文件核对官方仓库 [google/automl/lion](https://github.com/google/automl/tree/master/lion)（`README.md`、`lion_pytorch.py`、`lion_optax.py`）。
- **通读：** 已完成

## 一句话说明

论文把深度学习优化器设计形式化为命令式 Python/NumPy 风格符号程序搜索，借演化搜索、抽象执行剪枝、漏斗选择与化简发现 **Lion（EvoLved Sign Momentum）**——仅追踪单一动量并用 `sign` 生成等幅更新，显存较 AdamW 减半、步时提速 2–15%，同时明确了其相对 AdamW 的超参缩放定律与适用边界。

## 一手论文核心要点（arXiv:2302.06675v4）

### 1. 符号搜索方法论（§2, Appendix F/H/I）

- **搜索空间设计（§2.1 & App. H）**：将优化器编码为与 AdamW 共享输入输出签名的 `train(w, g, v1, v2, lr)` 函数（限制额外历史状态最多 2 个张量以保内存不高于 AdamW），基于 43 个作用于 n 维数组 / JAX PyTree 的数学与线性代数原语（含 `interp(x, y, a) = (1 - a)*x + a*y`）及可突变常数构造无限制长度的语句序列。
- **高效检索技术（§2.2 & App. I）**：
  - **热启动与双重启**：以 AdamW（Program 3）热启动正则化演化（种群 1K、锦标赛尺寸 2），结合从原点重启（探索）与从当前最佳算法重启（利用）。
  - **Abstract Execution 静态剪枝**：(1) 类型/形状静态推导过滤非法程序；(2) 构建功能哈希表对语义等价程序去重（缓存命中率 **$89.1 \pm 0.6\%$**，将每轮 200–300K 生成程序压缩至约 20–30K 实际评估，节省 ~10× 搜索算力）；(3) 依赖追踪剔除 **$69.8 \pm 1.9\%$** 冗余语句，使程序缩短 ~3×，便于人类解析。
- **弥合代理到目标的泛化鸿沟（§2.3 & App. F）**：代理任务在单颗 TPU V2 上 < 20 分钟完成（ViT on 10% ImageNet 与小 Transformer on LM1B），而目标任务规模放大 $> 10^4\times$。为对抗 **meta-overfitting**，作者设计逐级扩大 10×→100× 验证算力的 **Funnel Selection**，并对胜出原始程序（Program 8 → Program 4 → Program 1）剪除无影响函数（`cosh`、`arcsin`、`clip`）、合并两级线性插值（`~0.899` 与 `~1.109` 简化为单步 `0.99`）并把 `m / sqrt(m^2)` 等价化简为 `sign`。

### 2. Lion 算法更新规则与机制解析（Algorithm 2, §3, Appendix G）

- **核心更新流程（Algorithm 2）**：给定超参 $\beta_1, \beta_2, \lambda, \eta$（默认 $\beta_1 = 0.9, \beta_2 = 0.99$）：
  1. $c_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t$（结合旧动量与当前梯度决定本步更新方向）
  2. $\theta_t \leftarrow \theta_{t-1} - \eta_t (\mathrm{sign}(c_t) + \lambda \theta_{t-1})$（各维度统一幅度 $\pm 1$ 并加上解耦权重衰减）
  3. $m_t \leftarrow \beta_2 m_{t-1} + (1 - \beta_2) g_t$（以更慢衰减率更新用于后续步的动量状态）
- **关键结构洞见**：
  - **解耦记忆与当步权重**：通过 $(\beta_1=0.9, \beta_2=0.99)$ 将“记忆历史跨度”（$\beta_2=0.99$，比常见 0.9 长约 10×）与“更新时赋给当前梯度的权值”（$1-\beta_1=0.1$）分离；不跟踪二阶矩 $v_t$、无需 $\epsilon$ 与偏差修正。
  - **与 signSGD / NAdam 的区别**：`signSGD` 及常规动量版仅按单步或传统动量取符号；`NAdam` 虽同用当前梯度混入，但未利用 $\beta_2$ 分离记忆尺度与更新插值且保留二阶适应。
  - **损失几何与正则效应（Appendix G, Table 10）**：ImageNet ViT-B/16 上 Lion 终训损失略高于 AdamW（$L_{\text{train}}$：0.75 vs 0.61），但高斯参数扰动平坦度 $L_{\text{train}}^{\mathcal{N}} = \mathbb{E}_{\epsilon \sim \mathcal{N}}[L_{\text{train}}(w+\epsilon)]$ 明显更优（**1.37 vs AdamW 3.74**），验证 `sign` 二值噪声起正则作用并推动收敛至更平坦极小值（验证集反高 +1.96%）。

### 3. 主实验结果、超参规律与消融（§4–§6, Table 1–12）

- **图像分类（Table 2–3）**：ImageNet 从头训练无强增强时 ViT-B/16 提升 **+1.96%**（77.44% vs 75.48%），Mixer-B/16 提升 **+1.99%**（70.11% vs 68.12%），加 RandAug+Mixup 后 CoAtNet-3 仍领先 +0.42%（84.87% vs 84.45%）；JFT-300M 预训练节省 **3–5× 计算量**使 ViT-L/16 追平或超越 AdamW 的 ViT-H/14；JFT-3B 上 ViT-G/14 达 **90.71%**。
- **多模态与扩散模型（Table 1, Table 4, §4.3）**：BASIC-L 零样本 ImageNet 达 **88.3%**（超 Adafactor 85.7% 达 +2.6%），微调达 **91.1%**；ImageNet 256×256 无条件扩散生成训练步数减少 **2.3×** 且 FID 更佳（**4.1 vs AdamW 4.7**）；Imagen 64→256 超分模型效果更稳且 CLIP 分数更高，但 64×64 base 模型无显著优势。
- **语言建模与微调（§4.4, Table 5–6）**：Wiki-40B 与 PG-19 小/中/大模型上取得 **1.5–2×** 收敛提速；T5 Base/Large/11B 在 GLUE 平均全胜（10–12/12 指标领先）；但在高质量 1.6T token 内部语料大语言模型预训练上困惑度与 Adafactor 无差异（尽管 1-shot NLG/NLU 略好），C4 MLM 仅微弱领先。
- **核心消融与缩放法则（§4.5–§5, Table 7, Fig. 8, Table 12）**：
  - **双插值系数消融（Table 7）**：固定单一系数 `m = interp(g, m, β); update = sign(m)` 的 `Ablation_0.9` 与 `Ablation_0.99` 均在 ViT-S/16、ViT-B/16 与 PG-19 上显著差于完整 Lion 甚至不如 AdamW，证明 $(\beta_1, \beta_2)$ 协同不可或缺。
  - **批大小敏感性（Fig. 8 Left）**：ImageNet ViT-B/16 上 AdamW 最优 batch size 为 256，Lion 最优 batch size 为 **4,096**；在 **32K** 超大批次下 Lion 领先 AdamW **+2.5%**（77.9% vs 75.4%），而在 batch size 64 时仍稳健但缺乏优势。
  - **内存与速度（§3.2）**：`bfloat16` 存动量训练 ViT-B/16（BS=4096）时，AdamW 至少需 16 颗 TPU V4，Lion 仅需 **8 颗**；各任务训练步速快 **2–15%**。
  - **超参换算指南（§5 & Table 12）**：由于 `sign` 输出全维度 $\pm 1$，更新 $L_2$ 范数远大于 AdamW，因此 **学习率 `lr` 需取 AdamW 的 1/3–1/10**（含初始、峰值、终值同步缩放），同时为了保住有效衰减力度 $lr \cdot \lambda$，**解耦权重衰减 `λ` 需同比放大 3–10×**（例如 ViT-B/16 强增强用 `lr=1e-4, λ=10.0` 对齐 AdamW 的 `1e-3, 1.0`；扩散模型 `3e-5, 0.1` 对齐 `3e-4, 0.01`）。语言任务可调为 $\beta_1 = 0.95, \beta_2 = 0.98$ 缩短历史记忆窗口以提升稳态。

### 4. 一手论文明确界定的局限（§6）

- 搜索空间仍受一阶算法偏置限制，缺少构建 K-FAC / Shampoo 等二阶优化器的高阶矩阵算子，且搜索成本约 ~3K TPU V2 days 并仍需人工后简化。
- 在 ResNet 等较易优化的 ConvNet、高强度数据增强场景，以及高质量海量语料/大语言模型困惑度与 C4 MLM、Imagen base 模型上，Lion 与 AdamW 差距不显著。
- 当训练批次很小（**batch size < 64**）时，Lion 往往不比 AdamW 更好。
- 超大模型上即使采用 `bfloat16` 一阶动量仍有一定显存开销（后续需考虑动量因子化）。

## 官方源码核对摘要（`google/automl/lion`）

- **`lion_pytorch.py` 核对要点**：
  - `Lion(Optimizer)` 默认参数：`lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0`，校验 `lr >= 0` 与 `0 <= betas < 1`。
  - **`step()` 更新执行顺序**：对每个有效梯度 `grad`：
    1. 先执行解耦权重衰减：`p.data.mul_(1 - group['lr'] * group['weight_decay'])`（与论文 Pseudocode 合并在同一减法括号内 $\theta_t \leftarrow \theta_{t-1} - \eta(\mathrm{sign}(c_t) + \lambda\theta_{t-1})$ 一致到一阶精度）；
    2. 按 $\beta_1$ 插值并取符号更新参数：`update = exp_avg * beta1 + grad * (1 - beta1)`；`p.add_(update.sign_(), alpha=-group['lr'])`；
    3. 按 $\beta_2$ 原地刷新一阶动量 EMA：`exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)`。
- **`lion_optax.py` 核对要点**：
  - 通过 Optax 组合链实现：`optax.chain(scale_by_lion(b1, b2, mu_dtype), add_decayed_weights(weight_decay, mask), _scale_by_learning_rate(learning_rate))`。
  - 显式支持 **`mu_dtype`**（如传入 `jnp.bfloat16` 将一阶动量存储内存直接再减半）；并在 `scale_by_lion` 中先将动量临时 cast 至 `grad.dtype` 计算 `update` 与 `jnp.sign(update)` 保证符号精度，更新后的新动量 `mu` 再 cast 回 `mu_dtype` 存盘。
- **`README.md` 实用建议**：除重申 `lr` 缩小 3–10×、`weight_decay` 放大 3–10× 及语言任务 `betas=(0.95, 0.98)` 外，提示梯度裁剪/学习率调度器可直接沿用 AdamW 配置（或按需在更新前对 `c_t`/梯度裁剪）。

## 可引用来源

- 原始论文：Chen et al., *Symbolic Discovery of Optimization Algorithms*, arXiv:2302.06675v4（2023）。<https://arxiv.org/abs/2302.06675>
- 官方实现：google/automl/lion — <https://github.com/google/automl/tree/master/lion>（核心文件：`lion_pytorch.py`、`lion_optax.py`、`README.md`）
- 关联二手索引：[`sources/papers/deep_learning_optimizers.md`](./deep_learning_optimizers.md)
