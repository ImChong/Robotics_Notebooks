# 流策略高斯蒸馏（arXiv:2610.40271）

> 来源归档（paper）

- **标题：** Gaussian Distillation of Flow Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40271>
- **PDF：** <https://arxiv.org/pdf/2610.40271>
- **作者：** Saied, Youssef, Sheebaelhamd, Ziyad, Vernade, Claire, Fleuret, François
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** Flow 策略的高斯蒸馏。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** Flow Policy 需要多次网络前向传播才能生成动作，推理成本较高，也难以直接计算动作概率密度。
3. **周报所述创新点：** 提出 Gaussian Action Readout（GAR），用预测高斯均值和协方差的输出层替换 Flow Policy 最后一层，冻结其余网络。只使用约 0.02% 的可用训练状态拟合输出层，即可实现单次网络前向采样，并获得闭式动作概率密度。在 RoboCasa365 的 50 个任务中，蒸馏策略保持接近教师的成功率，推理速度最高提升约 2 倍。
4. **arXiv 摘要（英文节选）：** Flow policies are among the strongest robot controllers and are well suited to modelling multimodal action distributions. Our analysis shows that the continuous-action distributions of the studied flow policies are well approximated by Gaussians. We propose the Gaussian Action Readout (GAR), which replaces a flow policy's final layer with a Gaussian readout trained on just 0.02% of eligible states, while keeping the rest of the network frozen.

**对 wiki 的映射**

- [paper-gaussian-distillation-flow-policies](../../wiki/entities/paper-gaussian-distillation-flow-policies.md)
