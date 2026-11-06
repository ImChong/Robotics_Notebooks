# DDPM 通读归档（arXiv:2006.11239）

> 论文来源归档（通读版，供 diffusion-model 页 `original:` 引用）

- **标题：** Denoising Diffusion Probabilistic Models（Ho, Jain, Abbeel；NeurIPS 2020）
- **PDF：** <https://arxiv.org/pdf/2006.11239>
- **通读：** 已完成
- **核对范围：** 全文（摘要、§1 引言、§2 背景、§3 全部、§4 实验含消融/渐进编码/插值、§5 相关工作、§6 结论、附录 A 推导、B 实验细节、C 与 NCSN 对比）；附录 D 的额外样本图仅浏览。

## 核对要点

- 前向过程固定：$\beta_t$ 从 $10^{-4}$ 线性增至 0.02，$T=1000$；$q(x_t|x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$（式 4）。
- 逆向方差固定为 $\sigma_t^2=\beta_t$ 或 $\tilde\beta_t$（二者效果相近）；学习方差会使训练不稳、质量变差。
- 预测噪声 $\epsilon_\theta$ 的简化损失（Algorithm 1）与多噪声尺度去噪分数匹配相似；采样（Algorithm 2）形似 Langevin 动力学。预测 $x_0$ 在早期实验中样本质量更差。
- 骨干：基于 PixelCNN++ 的 U-Net，GroupNorm，16×16 分辨率自注意力，正弦时间嵌入加到每个残差块；CIFAR10 35.7M 参数，256×256 模型 114M。
- 结果：CIFAR10 FID 3.17（对训练集）；似然（码长）不如其他似然模型，多数比特用于不可感知细节。
- 训练细节：Adam，lr 2e-4（256×256 为 2e-5），EMA 0.9999，CIFAR10 dropout 0.1；TPU v3-8 上 256 张 CIFAR 图采样 17 秒，128 张 256×256 图 300 秒。
- 与自回归的联系：以掩码为前向过程时退化为自回归；高斯扩散可视为广义比特序的自回归解码。
