---
type: formalization
tags: [vla, machine-learning, robotics, math, tokenization]
status: complete
updated: 2026-11-07
related:
  - ../methods/vla.md
  - ../methods/behavior-cloning.md
  - ../formalizations/behavior-cloning-loss.md
  - ../entities/paper-mint-vla.md
  - ../entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md
  - ./cross-modal-attention.md
  - ../concepts/flow-matching-embodied-policy.md
original:
  - ../../sources/papers/fast_action_tokenization_arxiv_2501_09747.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/mint_rss_2026.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
sources:
  - ../../sources/papers/fast_action_tokenization_arxiv_2501_09747.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/mint_rss_2026.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
  - ../../sources/papers/perception.md
summary: "动作分词（Action Tokenization）是将机器人的高维连续动作空间映射为有限离散 Token 序列的过程，是使大语言模型架构能够直接预测物理动作的关键桥梁。"
---

# Action Tokenization (动作分词)

## 一句话定义

**动作分词（Action Tokenization）** 是把机器人连续动作（通常是一段动作块）映射成有限词表上的离散 token 序列、并能解码回连续动作的过程，使自回归的语言模型架构可以像预测下一个词一样预测动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |
| RT-2 | Robotics Transformer 2 | 将 web 规模 VLM 能力迁移到机器人控制的代表工作 |
| FAST | Frequency-space Action Sequence Tokenization | DCT + 量化 + BPE 的动作压缩分词 |
| DCT | Discrete Cosine Transform | 离散余弦变换，把信号分解成不同频率的余弦分量 |
| BPE | Byte-Pair Encoding | 字节对编码，无损合并高频符号对 |
| FSQ | Finite Scalar Quantization | 学习式量化（VQ-VAE 的简化替代） |
| VQ | Vector Quantization | 向量量化，用码本表示隐向量 |
| SDAT | Spectrally Disentangled Action Tokenizer | MINT 的谱解耦多尺度分词器 |

## 为什么重要

1. **统一架构：** RT-2 把动作写成文本 token（连续维度均匀离散为 256 个 bin，共 8 个整数），与网络图文数据一起 co-fine-tune，使 VLM 的语言接口直接输出动作；解码时只采样合法的动作 token。
2. **分词方式决定能否训练：** FAST 论文的案例研究显示，对同一条平滑轨迹，逐时间步分箱的自回归模型随采样率升高误差陡增，最终只会复制第一个动作；20 Hz 的 Table Bussing 与 50 Hz 的 T-Shirt Folding 上，分箱策略根本无法取得进展。
3. **与连续动作头的取舍：** π₀ 选择不分词，用 flow matching 动作专家输出连续动作；π₀-FAST 则证明，在 10k 小时、9.03 亿 timesteps 的数据上，自回归 + FAST 能达到与扩散 π₀ 相当的效果，训练算力约少 5 倍，但推理更慢（约 750 ms/块 vs 约 100 ms/块）。

## 核心原理

### 数学定义

假设原始动作空间为 $\mathcal{A} \subset \mathbb{R}^d$。FAST 论文的表述：分词器是映射 $T_a: a_{1:H}\to[T_1,\dots,T_n]$，把长度为 $H$、维度为 $|A|$ 的动作块映到词表 $|V|$ 上的 $n$ 个离散 token，**$n$ 可随序列变化**，就像同样长度的句子可被切成不同数目的文本 token。需要同时有解码器 $D$ 还原 $\hat a$，重建误差可用 $\|a-D(E(a))\|^2$ 衡量。

### 1. 标量分箱 (Scalar Binning)

对每个维度、每个时间步独立量化：把训练集取值范围均匀分成 $N$ 个 bin，最常用 $N=256$。一个 $H$ 步、$D$ 维动作块变成 $H\times D$ 个 token。

$$
b_{t,j}=\left\lfloor N\,\frac{a_{t,j}-a_j^{\min}}{a_j^{\max}-a_j^{\min}}\right\rfloor,\qquad t=1,\dots,H,\; j=1,\dots,D
$$

- $N$：bin 数（常取 256）。
- $H$：动作块的时间步数。
- $D$：动作维度数。
- **RT-1**：每个动作维度离散成 256 个 bin（在各变量范围内均匀分布），共 7 个手臂变量、3 个底盘变量和 1 个模式切换变量；用类别交叉熵训练，输出 3 Hz。
- **RT-2**：沿用 256 bin；PaLI-X 直接用整数 token，PaLM-E 覆盖 256 个最少使用的 token。
- **OpenVLA**：同样的逐维 256 分箱（据 FAST 论文对其的描述）。
- **局限：** 高频数据下一块可达数百个 token，训练困难、推理慢，相邻 token 高度相关使下一 token 预测的信息量趋近于零。

### 2. 向量量化与学习式压缩 (VQ / FSQ)

维护可学习码本 $\mathcal{C}=\{e_1,\dots,e_K\}$，把编码器输出量化到最近的码向量：$k=\arg\min_i\|z(a)-e_i\|_2$。优点是能压缩、能捕捉维度间相关；缺点是需要额外训练且对超参敏感。FAST 论文用 FSQ 作为对照：FSQ 与 FAST 相当或稍差，在低保真重建时更高效，但在高保真重建上扩展更差；MINT 把 VQ-VAE 类方法归入“学习式分词”，指出它们常保留低层运动学而非意图。本页此前把 Octo、π₀ 列为 VQ 代表是不准确的：π₀ 论文用 flow matching 输出连续动作，并把 Octo 称作用扩散过程生成动作的模型。

### 3. 频域压缩：FAST (DCT + BPE)

FAST（Pertsch et al., 2025）：先用 1% / 99% 分位数把每维归一化到 $[-1,1]$；对每个动作维度沿时间做 DCT；系数乘缩放因子 $\gamma$ 后取整，舍去不显著的高频系数；按“各维低频分量在前”的列优先顺序展平；再用 BPE 无损压缩。压缩率（1 秒块）：BridgeV2 35→20 token，DROID 105→29，Bussing 140→28，Shirt Fold 700→53，每个机械臂约 30 个 token，与控制频率几乎无关。只有两个超参（$\gamma=10$、BPE 词表 1024 在单数据集实验中通用）。**FAST+** 是在约 100 万条 1 秒动作块上训练的通用分词器，可作黑盒用于任意机器人。

### 4. 谱解耦多尺度分词：MINT 的 SDAT

MINT（Huang et al.）的 SDAT 不是先对输入做 DCT 再量化。编码器处理**时域**动作块，用多尺度残差量化（$K$ 个尺度，共享码本，最粗尺度只有 1 个 token）得到各尺度累计重建；**DCT 作用在重建结果上**，损失 $\mathcal L_{freq}=\sum_k\lambda_k\|F-F^{(k)}\|_2$ 要求先用 $S_1$ 重建频谱、再用 $S_1+S_2$……直到全尺度，迫使最粗 token 解释低频整体形状（Intent token），细尺度负责高频残差（Execution tokens），再加码本/承诺损失和时域 $\ell_1$ 辅助项。策略端做 next-scale 自回归（尺度间自回归、尺度内并行）。消融中，逐尺度**时域**损失反而使 LIBERO-Long 降到 82.8，而逐尺度谱损失为 93.4。

```mermaid
flowchart LR
  A["动作块 a(1:H)"] --> B["归一化"]
  B --> C["逐维 DCT"]
  C --> D["缩放 γ + 取整"]
  D --> E["低频优先展平"]
  E --> F["BPE 压缩"]
  F --> G["离散 action tokens"]
```

## 工程实践

- **选型经验（FAST 论文）：** 数据控制频率 ≥ 20 Hz 或块很长时，不要用逐步分箱；单数据集 FAST 或通用 FAST+ 都可，FAST+ 与专用版性能接近；去掉 BPE 会变差（重复 0 token 稀释学习信号并拖慢推理）；`AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)` 即可调用，建议对输入做分位数归一化并按 1 秒块分词。
- **本体状态不必精细分词：** FAST 论文里状态仍用 256 分箱写进文本，因为它是输入而不是需要精确输出的目标。
- **解码：** 单臂任务用贪心解码；双臂任务（T-Shirt Folding 等）用温度 0.7 才能让策略从初始姿态动起来。
- **推理开销：** 自回归需逐 token 解码（30–60 个动作 token），π₀-FAST 约 750 ms/块；扩散 π₀ 约 100 ms/块。作者认为 LLM 领域的投机解码、量化等可借用（未验证）。
- **与连续头的关系：** 见 [Flow Matching 与具身策略](../concepts/flow-matching-embodied-policy.md)；与注意力结构的关系见 [Cross-modal Attention](./cross-modal-attention.md)。

## 局限与风险

- **有损压缩：** FAST 在压缩率与重建误差间权衡，由 $\gamma$ 控制；不能默认无损。
- **评测范围：** FAST 论文的策略评测只在固定基座机械臂上；移动机器人、灵巧手、人形只做了离线压缩测试。
- **推理慢：** 自回归逐 token 解码，对高度动态任务不一定可用（作者承认只在静态操作上评测，慢推理未伤害性能）。
- **意图可解释性证据有限：** MINT 的 Intent token 解耦主要靠 t-SNE 可视化与消融；其“+15% LIBERO-Plus 鲁棒性、one-shot +60%”等数字出自作者自己的基准设置，且真机对手为改动版 π₀.₅。
- **未经原文核实的旧说法：** “离散化后的分布通常比连续回归更容易被建模，尤其是多模态分布”本页此前没有出处；多模态分布在 π₀ 等工作里恰恰是选 flow matching 的理由（推测，需另证）。

## 关联页面

- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [MINT（SDAT 频域意图分词）](../entities/paper-mint-vla.md)
- [FAST 论文实体（RCL 清单）](../entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md)
- [Behavior Cloning Loss](./behavior-cloning-loss.md)
- [Cross-modal Attention](./cross-modal-attention.md)
- [Flow Matching 与具身策略](../concepts/flow-matching-embodied-policy.md)

## 参考来源

- Pertsch, K., et al. (2025). *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. [归档](../../sources/papers/fast_action_tokenization_arxiv_2501_09747.md)
- Brohan, A., et al. (2023). *RT-2*. [归档](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md)
- Brohan, A., et al. (2022). *RT-1: Robotics Transformer for Real-World Control at Scale*. [归档](../../sources/papers/rt_1_arxiv_2212_06817.md)（本次只核对了动作 token 化与输出频率段落，未通读全文，因此不列入 original）
- Black, K., et al. (2024). *π0*. [归档](../../sources/papers/pi0_arxiv_2410_24164.md)
- Huang, R., et al. (2026). *Mimic Intent, Not Just Trajectories (MINT)*. [归档](../../sources/papers/mint_rss_2026.md)
- Padalkar, A., et al. (2023). *Open X-Embodiment: Robotic Learning at Scale*（旧页引用，本次未核对）。

## 推荐继续阅读

- [FAST 论文 arXiv:2501.09747](https://arxiv.org/abs/2501.09747) 与 [项目页](https://www.pi.website/research/fast)
- [MINT 论文 arXiv:2602.08602](https://arxiv.org/abs/2602.08602)
- [MINT 论文实体](../entities/paper-mint-vla.md)
