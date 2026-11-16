# Understanding Positional Embeddings in Modern LLMs / 现代大语言模型中的位置编码

> Sebastian Raschka 的原始技术教程归档（blog），不是独立新论文。

- **标题：** Understanding Positional Embeddings in Modern LLMs（现代大语言模型中的位置编码）
- **作者：** Sebastian Raschka
- **类型：** blog / LLM / Transformer / positional-encoding
- **发布日期：** 2026-11-15
- **入库日期：** 2026-11-16
- **原文：** <https://magazine.sebastianraschka.com/p/understanding-positional-embeddings>
- **原文版本：** 2026-11-15 发布的在线文章；2026-11-16 访问核对
- **配套代码（文章引用的独立 Qwen3 示例）：** <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb>
- **核对范围：** 原文 §1–§7 全部正文：位置注入位置；GPT-2 learned absolute 与原始 Transformer sinusoidal；RoPE 配对/旋转/相对位移/频率；PyTorch 示例；Partial RoPE、NoPE、learned relative position bias；2025–2026 架构图库趋势分析、结语及原文说明。另核对作者链接的 Qwen3 notebook 中 compute_rope_params、apply_rope、GroupedQueryAttention.forward、TransformerBlock.forward 与 Qwen3Model.forward。
- **通读：** 已完成
- **一句话说明：** 从绝对位置向量讲到注意力内旋转 Q/K 的 RoPE，再比较部分旋转、跳过显式位置编码与注意力分数相对偏置，解释位置建模的机制和实现边界。

## 原文机制摘录

1. **位置信息进入位置不同**：GPT-2 的 learned absolute embedding 是输入 token embedding 加上一张训练得到的位置向量表；原始 Transformer 将固定的正弦余弦向量加到输入嵌入（还将 token embedding 乘以维度平方根）；近年 RoPE 在注意力内部旋转 query/key 而不旋转 value。
2. **RoPE 相对性**：每对特征按 token 位置与频率旋转；假定未经旋转的 q/k 固定，二者的旋转内积依赖位置差而非共同偏移。文章例子 q 位于 5、k 位于 2，各转 100° / 40°，原始相同特征对 (1, 5) 的内积由 26 变成 26×cos 60° = 13。
3. **频率与配对**：标准频率 θ_i = b^(-2i/d)，d=8、b=10000 时为 [1, 0.1, 0.01, 0.001]（每 token 的弧度数）。split-half 和 interleaved 两种布局在适当重排 q/k 特征后数学等价；未经重排不能直接互换。
4. **近年变体**：Partial RoPE 只旋转部分维度，减少正余弦表规模而非 KV cache 本身；NoPE 依靠 causal mask 的可见前缀携带隐式顺序信号；RoPE/NoPE 可按 block 混用；Inkling 在注意力 logit 上加由 query 状态与相对距离决定的可学习位置偏置。
5. **2026 观察**：作者的 LLM Architecture Gallery 里每注意力层都用 RoPE 仍常见，但 Partial RoPE、混合 NoPE 的图库条目增多。**不是对所有公开模型的系统抽样**，不能把图库比例当成整个行业份额。

## 工程与局限核对

- GPT-2 learned absolute 向量表受到训练定义的位置范围约束；仅把正弦/RoPE 公式算到更长位置，不代表模型可靠泛化到新长度。
- RoPE 的相对位移不变性成立时，必须假定未经旋转的 query/key 值保持不变；在真实模型前插上下文可能改变隐藏状态，不能保证实际注意力分数不变。
- NoPE 长度泛化结果来自作者引用的特定小规模合成任务及代码模型实验，不可直接等同于所有真实长上下文任务的优势。
- Partial RoPE 的内存节省与模型 head_dim、旋转维度、上下文窗以及数值格式有关；原文作者认为其收益未必值得额外复杂度，属于作者观点，不是通用性能定律。
- 文章包含可单独运行的教学 PyTorch 片段，但没有宣称本文对应一个独立训练/发布项目。所引用 Qwen3 notebook 是作者已有 LLMs-from-scratch 仓库中的教学实现；已阅读其函数调用，**未执行模型推理或下载权重**。
- 机器人/VLA 中的语言、视觉和动作 token 的位置/时间坐标并不相同；把本文一维 token RoPE 直接推广到多模态时空坐标，需要额外设计，**此项是知识库工程延伸，非原文实验结论**。

## 对 wiki 的映射

- [位置编码：绝对、RoPE、NoPE 与相对偏置](../../wiki/concepts/positional-encoding.md) — 新建概念页，解释公式、实现、选型及边界。
- [Transformer](../../wiki/concepts/transformer.md) — 在架构基础页补入现代位置编码机制的链接。
- [模型架构纵深 Stage 1](../../roadmap/depth-model-architecture.md) — 在骨干族谱阶段安排位置编码学习入口。
