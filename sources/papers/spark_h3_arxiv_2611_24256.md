# Spark-H3: Adaptive Block Sparse Attention for Video Diffusion Models（视频扩散模型的自适应分块稀疏注意力）

> 一手论文来源归档（paper）

- **类型：** paper
- **论文：** [arXiv:2611.24256v1](https://arxiv.org/abs/2611.24256)
- **HTML 全文：** https://arxiv.org/html/2611.24256v1
- **PDF：** https://arxiv.org/pdf/2611.24256
- **作者：** Zecheng Tang、Jiaye Fu、Youtian Lin、Xinyao Rao、Liuxuan Sun、Jian Zhang、Yao Yao、Wei Li
- **公开版本：** arXiv v1，2026-11-18，未经同行评审的预印本
- **项目主页：** https://zechengtang.github.io/Spark-H3/
- **代码：** https://github.com/zechengtang/Spark-H3
- **模型卡/演示：** https://huggingface.co/Aazeus/Spark-H3
- **核对范围：** 通读 arXiv HTML v1 的摘要、`1–5 全部正文（含相关工作、`3.1 双分支 BSA、`3.2 Reblock、`3.3 Reweight、`4.1–4.5 的 MiniMax-H3/LTX-2.5/Turbo 结果、Table 1–5、消融与 `5 结论）；阅读正文图题与参考文献。未下载 PDF 图像或复测指标，论文 HTML 无单列技术附录。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 论文核心摘录

1. **问题定位（`1）：** 视频 DiT 的长时空序列造成密集自注意力成本高；静态分块把注意力偏好不同的 token 放进同一 tile，有限精确预算捕获不到足够多的 attention mass；简单均值 KV 压缩又因为 Jensen gap 低估被压缩块的总注意力质量。
2. **双分支分块稀疏注意力（`3.1）：** 对每个 query tile 以 query/key tile 均值评分，按精确块密度 ρ 选 Top-K；选中块执行 token-level 精确注意力，其余块以压缩 KV 近似。两支共享 softmax 归一化，对已选块掩掉压缩贡献以免双计数。
3. **Spark-Reblock（`3.2、Algorithm 1）：** 从 Q/K 所诱导的注意力偏好距离出发，估计二阶矩，在每棵分治树节点由 landmarks 拟合局部划分方向，按 rank 平衡递归划分成等长 tile；可批量完成 4 层二分、16 子组，构建复杂度约 O(N log N)。Spark-H3 默认 32 landmarks、每次划分 2 次代表更新和 64-token tile，实际 H3 路径复用 key 排序到 Q/K/V。
4. **Spark-Reweight（`3.3）：** 以全局平均 query 为锚点，针对每个 KV 块计算 softmax 加权的 key/value 代表与 logit 偏置 β；这是压缩块 log-sum-exp 在平均 query 点的一阶近似。只保证在锚点处质量及梯度校准，**不是**对所有 query 的精确重构。
5. **MiniMax-H3（Table 1）：** 14.4s、345 帧、1344×768、20 次去噪、前 4 步密集热身。10% exact density：DiT 4061.8s（Dense）→ 1629.6s（Spark），2.49×，PSNR 24.81 dB、VBench 84.56；20%：1796.1s、2.26×、PSNR 26.76、VBench 84.50。
6. **方法横向与跨模型（Table 1–2）：** MiniMax-H3 上 10% Spark-H3 相对 10% PISA 的 PSNR 24.81 vs 20.20；LTX-2.5 仅对高分辨率第二阶段应用稀疏注意力，10% ρ 对该阶段 DiT 为 3.92×、PSNR 30.84 dB；不可表述为 LTX-2.5 全流程 3.92×。
7. **少步蒸馏共存（`4.3、Table 3）：** MiniMax-H3-Turbo 社区 8-step LoRA + 前 2 步密集，10% density 得 2.27×、24.80 dB；20% density 得 2.08×、26.48 dB。LoRA 本身减步数与本文的注意力替换是两个不同加速维度。
8. **消融（Table 4–5）：** 在固定 Reweight 下，10% 密度加 Reblock 的 attention-mass recall 61.36% → 79.69%，PSNR 20.30 → 24.81；20% 密度下同时开启 Reblock+Reweight，log-mass MAE 从 1.39 降到 1.16，PSNR 26.24 → 26.76。
9. **指标边界（`4.1）：** 延迟与速度只量测 **DiT 主干**，明确排除文本编码及 VAE 解码；PSNR/SSIM/LPIPS 是相对密集注意力结果的忠实度，不代表物理正确性或独立视频质量真值。

## 对 wiki 的映射

- [Spark-H3 独立论文实体](../../wiki/entities/paper-spark-h3.md)
- [项目主页归档](../sites/spark-h3-project.md)
- [代码入口与运行时核对](../repos/zechengtang-spark-h3.md)
- [MiniMax-H3 基座](../../wiki/entities/minimax-h3.md)
