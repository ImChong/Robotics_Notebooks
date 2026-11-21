# Unified Multimodal Understanding via Byte-Pair Visual Encoding (Being-VL-0.5)

> 来源归档（paper / arXiv:2506.23639v1，ICCV 2025）

- **URL：** <https://arxiv.org/abs/2506.23639>
- **PDF：** <https://arxiv.org/pdf/2506.23639.pdf>
- **项目页：** <https://beingbeyond.github.io/Being-VL-0.5/>
- **代码仓库：** <https://github.com/BeingBeyond/Being-VL-0.5>
- **标题：** Unified Multimodal Understanding via Byte-Pair Visual Encoding
- **作者：** Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University、Renmin University of China
- **年份：** 2025（arXiv 2025-06；ICCV 2025）
- **核对范围：** 已获取并通读 `arXiv:2506.23639v1` PDF 抽取全文（Sections 1–5 及实验对比表，涵盖 VQ-GAN + Priority-guided vBPE 词表构造、FD/PD/RD/ID 四类指令配方、三阶段课程解冻策略与多模态理解评测）；未独立复现预训练。
- **通读：** 已完成

## 摘要与核心贡献

1. **纯离散多模态大语言模型（无需连续 ViT/MLP 投影器）**：以 Llama-3.1-8B 为统一自回归骨干，直接利用 VQ-GAN（码本大小 8192）对图像做空间量化，将视觉块与文本编码在**同一扩展离散词表**下统一建模。
2. **Priority-Guided Visual Byte-Pair Encoding（vBPE）**：针对原生 VQ token 二维局部语义割裂及单码信息稀疏问题，在合并优先级函数中引入**频次 $F(a,b)$ 与空间邻接得分 $S(a,b)$ 的加权和**：
   $$P(a,b) = F(a,b) + \alpha \cdot S(a,b), \quad (\alpha=0.3, \sigma=2.0)$$
   分别构建 +8K 与 +16K 扩展视觉词表（`Being-VL-0.5-8K` / `16K`），显著压缩单图视觉 token 长度并增强高频视觉结构共现建模（相对纯 VQ 无 BPE 方案在 MMBench 从 `38.2` 跃升至 `72.1`）。
3. **三阶段渐进解冻训练**：构造 Foundational（FD）、Positional（PD）、Region-level（RD）、Instructional（ID）四级课程数据；Stage 1 仅对齐新增视觉 embedding 层，Stage 2 解冻前 25% 浅层 Transformer 做多模态特征混合，Stage 3 全参指令微调。
4. **评测表现**：在 VQAv2（`80.2 / 80.6`）、MMBench（`71.8 / 72.1`）、MME-Perception（`1525.8 / 1536.3`）、POPE（`84.3 / 86.0`）上大幅超越 Being-VL-0（MMBench `44.0`）并与 LLaVA-1.5 等连续 ViT 编码架构对齐。

## 与知识库关联

- 实体节点：[cn-os-being-vl-0-5.md](../../wiki/entities/cn-os-being-vl-0-5.md)
- 代码归档：[being-vl-0_5.md](../repos/being-vl-0_5.md)
