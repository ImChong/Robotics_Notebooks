# FAST: Efficient Action Tokenization for Vision-Language-Action Models（arXiv:2501.09747）

> 论文来源归档（ingest）。此前该论文仅有策展清单摘录（[RCL Awesome WAM 清单条目](rcl_awesome_wam_2501_09747_fast-efficient-action-tokenization-for-v.md)），本归档为 2026-11-07 通读全文的版本。

- **标题：** FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **作者：** Karl Pertsch, Kyle Stachowicz 等（Physical Intelligence / UC Berkeley / Stanford）
- **类型：** paper / vla / action-tokenization
- **arXiv：** <https://arxiv.org/abs/2501.09747> · PDF：<https://arxiv.org/pdf/2501.09747>
- **项目页（研究介绍，不是代码仓库）：** <https://www.pi.website/research/fast>
- **FAST+ tokenizer 官方代码与资产：** <https://huggingface.co/physical-intelligence/fast/tree/main>
- **π₀-FAST 官方训练/推理代码：** <https://github.com/Physical-Intelligence/openpi>
- **2026-11-08 资源复核：** [项目页归档](../sites/fast-action-tokenization.md) · [官方源码函数归档](../repos/fast-action-tokenizer.md)。两个实现层次分开：HF 提供分词器，openpi 提供策略实现及 checkpoint 入口。
- **入库日期：** 2026-11-07
- **版本：** arXiv:2501.09747（2025-01-16 提交；本次下载 PDF 共 19 页）
- **核对范围：** 2026-11-07 下载 PDF 并通读摘要、§I 引言、§II 相关工作、§III 预备（含分箱 tokenization 定义）、§IV 案例研究、§V 方法（DCT 压缩、FAST 算法、通用分词器 FAST+）、§VI 全部实验（对比分词器、Table I 压缩率、FAST+ 通用性、消融、与扩散 π0 对比、10k 小时规模化）、§VII 讨论与局限，以及附录 A（FAST+ 训练数据配比）、B（压缩-重建权衡）、C（策略训练）、D（DROID 设置）、E（评测任务）；Table III 的完整数据集列表和各任务细节评分表只浏览；参考文献仅浏览；未运行 FAST+ 代码。
- **通读：** 已完成
- **一句话说明：** 用 DCT + 量化 + BPE 压缩动作块得到离散 token，使自回归 VLA 能在高频灵巧数据上训练，π0-FAST 与扩散 π0 相当而训练算力减至约 1/5。

## 通读核对结论（2026-11-07）

- **问题表述：** 策略 $\pi(a_{1:H}|o)$ 输出长度 $H$ 的动作块；分词器是映射 $T_a:a_{1:H}\to[T_1,\dots,T_n]$，$n$ 随序列可变。常用做法（RT-1/RT-2/OpenVLA）是逐维、逐时间步把取值范围均匀分成 $N$ 个 bin，最常用 $N=256$，$H\times D$ 个 token，高频数据一块可达数百个 token。
- **案例研究（§IV）：** 对 25–800 步采样率的三次样条插值任务，分箱 + 自回归 Transformer 的预测误差随采样率陡增，最终退化为复制第一个动作；FAST 的 DCT 分词在各采样率下误差平稳。解释：自回归训练的学习信号与第 $i$ 个 token 对前面 token 的**边际信息量**成正比，而采样率越高相邻 token 越相关，边际信息趋近于零。OpenVLA 在低频 Bridge/RT-1 上可用、在较高频 DROID 上难以拟合与此一致。
- **算法（§V-B，Fig. 4、Algorithm 1）：** ① 用训练集每维 1% 与 99% 分位数把动作归一化到 $[-1,1]$；② 对每个动作维度独立做 DCT；③ 乘以缩放系数 $\gamma$ 后取整（舍去不显著系数）；④ 展平为一维整数序列——**按列优先**，先放各维的低频分量（作者发现先预测低频整体形状更稳）；⑤ 用 BPE 无损压缩成最终 token。可逆，解码快。超参只有 $\gamma$ 与 BPE 词表大小（单数据集实验取 $\gamma=10$、词表 1024）。
- **压缩率（Table I，1 秒动作块）：** BridgeV2（7 维、5 Hz）35 → 20 token（1.75×）；DROID（7 维、15 Hz）105 → 29（3.6×）；Bussing（7 维、20 Hz）140 → 28（5.0×）；Shirt Fold（14 维、50 Hz）700 → 53（13.2×）。作者观察到每个机械臂约 30 个 token，几乎与控制频率无关。
- **FAST+（§V-C）：** 在约 100 万条 1 秒动作块上训练的通用分词器（单臂、双臂、移动；关节、末端、相机系；多种频率；动作补零到 32 维），可作黑盒使用；在未见数据集上压缩率约 2× 以上，策略性能与数据集专用 FAST 接近。
- **实验设置（§VI）：** 骨干 π0（PaliGemma-3B）与 OpenVLA（Prismatic 7B）；7 个评测环境（LIBERO、Table Bussing 20 Hz、T-Shirt Folding 50 Hz、Grocery Bagging 20 Hz、Toast 50 Hz、Laundry Folding 50 Hz、DROID 零样本 15 Hz）。对照：分箱、FSQ（学习式压缩）、FAST、FAST+。结论：分箱在 20 Hz 以上的 Table Bussing 与 50 Hz 的 T-Shirt Folding 上无法取得进展；FAST 与 FSQ 相当或更好，且无需额外神经网络；FAST 使首个强泛化的 DROID 策略成为可能，可在三所大学的未见环境中零样本评测。在 OpenVLA 上用 FAST+ 也能显著改善 T-Shirt Folding。**去掉 BPE** 会降低性能（大量重复 0 token 稀释学习信号并拖慢推理），但仍优于分箱。
- **与扩散 π0 对比（§VI-E–F）：** 小数据集（LIBERO、T-Shirt Folding，<50 小时）二者相当；Table Bussing 上 π0-FAST 用约 1/3 训练步数达到同等性能；DROID 上 FAST 更好地遵循语言指令。10k 小时、9.03 亿 timesteps 的 π0 混合数据上，π0-FAST 与扩散 π0 持平（包括 Laundry Folding），训练 GPU 小时少约 5 倍。**局限：** 推理慢——π0-FAST 约 750 ms/块（解码 30–60 个动作 token，用 2B 级语言模型主干），扩散 π0 约 100 ms/块（4090，10 步去噪、300M 动作专家）。
- **压缩-保真权衡（附录 B、Fig. 12）：** FSQ 等 VQ 类方法在低保真时更高效，但 FAST 在高保真重建上扩展更好，更适合精细控制。
- **本地推理与训练细节（附录 C）：** 图像 224×224；本体状态也用 256 分箱写进文本；AdamW（β1=0.9，β2=0.95），恒定学习率 5e-5，EMA 0.999；推理贪心解码，双臂任务用温度 0.7。
- **原文自述局限与展望：** 只在固定基座机械臂上做了策略评测，移动/灵巧手/人形只做了离线压缩测试；非自回归（扩散）解码与压缩编码的结合、自回归 VLA 推理加速留作未来工作。
- **原文没有的内容：** 论文没有把 FAST 与 Octo、MINT 对比；“VQ 类分词（Octo、π0 部分组件）”之类说法不是本文内容（本文把 VQ 类称作另一类学习式分词，并实测了 FSQ）。

## 2026-11-08 源码核对补充

- 官方 FAST+ `UniversalActionProcessor.__call__` 使用 `dct(..., axis=1, norm="ortho")`、系数量化、BPE；`decode` 再反变换。如果 BPE 解码/矩阵形状错误，源码会退回**全零系数**，需在实际控制链路检测异常，而不是默认输出可信。
- openpi 的 `FASTTokenizer.tokenize/extract_actions` 实现动作 token ↔ PaliGemma 词表的映射，`Pi0FAST.compute_loss/sample_actions` 对应交叉熵训练与逐 token 解码。推理由 `Policy.infer` 调用，并由 output transform 将 tokens 解码为动作。
- 源码与 API 的阅读范围、开源边界见 [FAST / FAST+ 官方源码归档](../repos/fast-action-tokenizer.md)；本次未运行模型或真机，不将论文的 5× 训练加速视作在线推理加速。
