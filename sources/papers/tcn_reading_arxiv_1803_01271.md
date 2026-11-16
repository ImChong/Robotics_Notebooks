# TCN 一手论文精读与源码核对归档（arXiv:1803.01271）

> 来源归档（ingest + primary reading）：Bai, Kolter & Koltun (2018) *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*（arXiv:1803.01271v2）与官方仓库 [locuslab/TCN](https://github.com/locuslab/TCN) 一手核对记录。

- **入库日期：** 2026-11-14
- **沉淀到 wiki：** 是 → [`wiki/concepts/temporal-convolutional-network.md`](../../wiki/concepts/temporal-convolutional-network.md)
- **核对范围：** 全文通读 Bai, Kolter & Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*（arXiv:1803.01271v2，正文 §1–§4 及附录 A–D），并逐文件核对官方仓库 [locuslab/TCN](https://github.com/locuslab/TCN)（`README.md`、`TCN/tcn.py`、`TCN/adding_problem/model.py`、`TCN/adding_problem/add_test.py`）。
- **通读：** 已完成

## 一句话说明

论文以 `TCN = 1D FCN + causal convolutions` 给出通用时序卷积基线架构，在 11 项序列基准与有效记忆长度评测上推翻了“通用序列建模必选 LSTM/GRU”的默认习惯，并严格剖析了固定感受野带来的评估历史存储与跨域迁移限制。

## 一手论文核心要点（arXiv:1803.01271v2）

### 1. 架构定义与核心公式（§3.1–§3.3）

- **基本构造**：
  1. **一维全卷积输出等长（1D FCN）**：隐藏层与输入层保持相同时间长度 $T$，通过零填充保证张量维度对齐。
  2. **严格因果卷积（Causal Convolution）**：时刻 $t$ 的输出仅依赖同一层及前驱层时刻 $\le t$ 的输入，绝不泄漏未来信号。
  3. **膨胀卷积（Dilated Convolution，Eq. 2）**：对一维输入序列 $\mathbf{x} \in \mathbb{R}^n$ 与卷积核 $f: \{0, \dots, k-1\} \to \mathbb{R}$，在索引 $s$ 上的膨胀卷积算子为：
     $$F(s) = (\mathbf{x} *_{d} f)(s) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{s - d \cdot i}$$
     其中 $d$ 为膨胀因子（随网络深度按指数增长 $d = O(2^i)$，即第 $i$ 个残差块 $d=2^i$），$k$ 为卷积核大小；单层有效回溯步数为 $(k-1)d$。
  4. **残差块结构（Fig. 1）**：每个 `Residual Block` 内包含两层串联的「膨胀因果卷积 → WeightNorm → ReLU → Spatial Dropout（按整通道置零正则化）」，并在输入输出通道数不等时加入可选 $1 \times 1$ 卷积以对齐残差维度，最后输出 $o = \mathrm{Activation}(x + \mathcal{F}(x))$。

### 2. 理论优劣对比（§3.4–§3.5）

- **相比 RNN（LSTM/GRU）的优势（§3.4）**：
  - **训练并行性**：前向不必等待 $h_{t-1}$ 完工，整条序列所有时间步共享滤波器并行计算。
  - **灵活可控的感受野**：改变卷积核 $k$、膨胀底数与块数 $L$ 即可精确设定历史窗口。
  - **梯度稳定性**：反向传播路径深度由网络层数决定，与序列长度 $T$ 解耦，缓解 BPTT 梯度爆炸/消失。
  - **较低训练显存**：同一时间步权重共享，不需像 LSTM/GRU 那样为每个时间步缓存全部门控中间激活张量。
- **论文明确指出的固有缺点（§3.5）**：
  - **评估/推理期的原始历史存储成本**：RNN 只需维护固定大小的汇总状态 $h_t$ 并丢弃旧输入；而标准 TCN 在推理阶段必须保留直到有效感受野深度的原始输入/中间激活历史，从而消耗更多运行时缓冲内存。
  - **跨域迁移时感受野不可自适应扩展**：将模型迁移到需要更长历史记忆的新任务时，若原先固定的 $(k, d)$ 与深度产生的感受野不足，TCN 性能会硬性受限而无法自动外推。

### 3. 基准实验、有效记忆与消融发现（§4, Table 1–2, Fig. 5–6, Table 5）

- **11 项跨模态基准（Table 1，默认超参下通用 TCN vs 通用 LSTM/GRU/RNN）**：
  - **Sequential MNIST**：TCN **99.0%** vs LSTM 87.2% / GRU 96.2%；**Permuted MNIST**：TCN **97.2%** vs LSTM 85.7% / GRU 87.3%。
  - **Adding Problem（$T=600$）**：TCN MAE/MSE 降至 **5.8e-5**（通用 LSTM/GRU/RNN 均未能收敛学成）。
  - **Copy Memory（$T=1000$）**：TCN 损失 **3.5e-5**（通用 LSTM/GRU 完全失效）。
  - **复调音乐（NLL↓）**：JSB Chorales **8.10**（LSTM 8.45）、Nottingham **3.07**（LSTM 3.29）。
  - **语言建模与字符级建模**：WikiText-103 困惑度 **45.19**（优于同尺度标准 LSTM 48.4）、LAMBADA 困惑度 **1279**（LSTM 4186）、Char PTB **1.31 bpc**、text8 **1.45 bpc**；但在 **Word-level PTB** 上 TCN 困惑度 **88.68** 仍落后于高度正则化特调 LSTM（78.93）。
- **有效记忆长度实测（Fig. 5）**：在 Copy Memory 任务随延迟长度 $T$ 增加时，TCN 在所有测试尺度保持 **100% 准确率**，而标准 LSTM 在 $T<50$、GRU 在 $T<200$ 之前即跌破 20% 准确率。
- **消融实验（Table 2, Fig. 6, Table 5）**：
  - **卷积核大小 $k$（Table 2）**：Copy Memory 与 P-MNIST 在 $k \le 3$ 时彻底失效、需要更大核宽（如 $k=7, 8$），而 Word PTB 在 $k=3$ 表现最佳，说明 $k$ 对任务信息跨度高度敏感。
  - **残差连接（Fig. 6）**：残差旁路显著加速深层 TCN 收敛并提升梯度稳定性。
  - **激活与门控对比（Table 5）**：将 ReLU 替换为 GLU（门控线性单元）使卷积计算量翻倍；在 Word PTB 略降困惑度（87.94 vs 88.68），但在 Copy Memory $T=1000$ 上反而显著退化（0.00508 vs ReLU 的 3.5e-5）。

## 官方源码核对摘要（`locuslab/TCN`）

- **`TCN/tcn.py` 逐模块核对**：
  - `Chomp1d(chomp_size)`：PyTorch `Conv1d` 默认双向等量 padding，因此前向设置 `padding = (kernel_size - 1) * dilation_size` 后，借助 `Chomp1d.forward(x)` 直接切片 `x[:, :, :-self.chomp_size].contiguous()` 裁掉右侧未来时间步，从而严格保障因果性。
  - `TemporalBlock`：包含两组串联的 `weight_norm(nn.Conv1d(...)) -> Chomp1d -> ReLU -> Dropout`，当 `n_inputs != n_outputs` 时构建 `downsample = nn.Conv1d(n_inputs, n_outputs, 1)` 对齐通道，最后执行 `self.relu(out + res)`；`init_weights()` 对两层卷积及可选 `downsample` 均使用 `normal_(0, 0.01)` 初始化（注：`README.md` 提醒新版 PyTorch 下可兼容或替换旧版 `weight_norm` 接口，原仓未对单步流式推理专门提供缓存环形队列 kernel）。
  - `TemporalConvNet(num_inputs, num_channels, kernel_size=2, dropout=0.2)`：按层级索引 `i` 循环构建 `num_levels = len(num_channels)` 个 `TemporalBlock`，严格按 `dilation_size = 2 ** i` 指数递增。
  - 理论感受野公式（对应 `TemporalBlock` 双卷积层结构）：设每层核宽为 $k$、共 $L$ 个残差块（膨胀 $d_i = 2^i, i=0,\dots,L-1$），每个 `TemporalBlock` 内含 2 个同等膨胀的因果卷积层，单块增加感受野 $2(k-1)2^i$，故整网总感受野为 $1 + 2(k-1)(2^L - 1)$。
- **`TCN/adding_problem/model.py` 下游读出模式**：针对序列回归/分类任务，直接取 `TemporalConvNet` 输出最后一步特征 `y1[:, :, -1]` 接入线性读出层（`self.linear`），保证不借助未来步。

## 可引用来源

- 原始论文：Bai, Kolter & Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv:1803.01271v2（2018）。<https://arxiv.org/abs/1803.01271>
- 官方代码：locuslab/TCN — <https://github.com/locuslab/TCN>（核心文件：`TCN/tcn.py`、`TCN/adding_problem/model.py`、`TCN/adding_problem/add_test.py`）
- 关联二手索引：[`sources/papers/bai_tcn_arxiv_1803_01271.md`](./bai_tcn_arxiv_1803_01271.md)、[`sources/papers/ai_architecture_foundations.md`](./ai_architecture_foundations.md)
