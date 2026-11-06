# Attention Is All You Need 论文通读（架构地图视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/ai-architecture-map.md`）

- **标题：** Attention Is All You Need
- **作者：** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin（Google）
- **类型：** paper / architecture / transformer
- **原文：** <https://arxiv.org/abs/1706.03762>（2026-11 通过 arXiv PDF v7 获取全文）
- **一句话说明：** 提出只用注意力、不用循环与卷积的 Transformer 编码器—解码器，是架构地图中"序列支"从 RNN/CNN 转向注意力的出处。
- **核对范围：** 通读 §1–§7 全文（含 §3 架构、§4 为何自注意力、§5 训练、§6 机器翻译 / 消融 / 句法分析实验）及参考文献表；附录的注意力可视化图（Figure 3–5）只浏览了图注，未逐词核对。未阅读 ViT、Mamba 等其他论文，地图中其余分支仅依赖 wiki 既有内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1–§2）：** 循环模型沿位置顺序计算，训练样本内部无法并行；卷积模型关联两个远距位置所需操作数随距离线性（ConvS2S）或对数（ByteNet）增长；Transformer 将其降为常数次操作，代价是注意力加权平均降低有效分辨率，用多头注意力补偿。
- **结构（§3.1）：** 编码器、解码器各 N=6 层；编码层 = 多头自注意力 + 逐位置前馈，每个子层外有残差连接与 LayerNorm，d_model=512；解码层多一个对编码输出的注意力子层，并对自注意力做掩码以保持自回归。
- **注意力（§3.2，式 1）：** Attention(Q,K,V)=softmax(QKᵀ/√d_k)V；缩放是因为 d_k 大时点积方差大、softmax 进入梯度极小区。多头 h=8，d_k=d_v=64，总计算量与单头全维度相近。
- **前馈与位置编码（§3.3、§3.5）：** FFN 为两层线性夹 ReLU，d_ff=2048；无循环无卷积，故加入正弦位置编码，与学习式位置嵌入结果几乎相同（Table 3 行 E）。
- **复杂度对比（§4，Table 1）：** 自注意力每层 O(n²·d)、串行步数 O(1)、最大路径 O(1)；循环 O(n·d²)、O(n)、O(n)；卷积 O(k·n·d²)、O(1)、O(log_k n)。n 小于 d 时自注意力比循环更省；超长序列可用邻域受限注意力（路径变 O(n/r)），作者列为未来工作。
- **训练（§5）：** WMT14 英德约 450 万句对、英法 3600 万句；8×P100，base 10 万步约 12 小时，big 30 万步 3.5 天；Adam，warmup 4000 步；残差 dropout 0.1 与标签平滑 0.1。
- **结果（§6.1，Table 2）：** 英德 BLEU：big 28.4、base 27.3，超过此前含集成的最好结果 2 BLEU 以上；英法：big 41.8（文字另写 41.0，原文自身不一致，表为 41.8），训练成本为此前最优的 1/4 以内。
- **消融（§6.2，Table 3）：** 单头比最优设置低 0.9 BLEU，头数过多也变差；减小 d_k 变差；更大模型更好；dropout 有效。
- **泛化（§6.3，Table 4）：** 4 层 Transformer 做英文成分句法分析，仅 WSJ 4 万句时 F1 91.3，半监督 92.7，未做任务特化调参。
- **结论与展望（§7）：** 计划扩展到文本以外的模态，研究局部受限注意力以处理图像、音频、视频等大输入，并减少生成的顺序性。

## 对 wiki 的映射

- [`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)：序列支（RNN→Transformer）的出处与复杂度对照
- 既有论文归档：[attention_is_all_you_need.md](attention_is_all_you_need.md)
