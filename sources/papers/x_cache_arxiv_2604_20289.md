# X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference（arXiv:2604.20289）

> 来源归档（ingest）

- **标题：** X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference
- **缩写：** **X-Cache**
- **类型：** paper / inference-acceleration / world-models / diffusion-cache / autonomous-driving
- **arXiv：** <https://arxiv.org/abs/2604.20289>
- **项目页：** <https://x-cache-1.github.io/en/>
- **代码：** 截至 2026-07-21 项目页仅链 arXiv，**未列 GitHub**
- **机构：** 小鹏（XPeng）AI Infra Team
- **作者（前若干）：** Yixiao Zeng, Jianlei Zheng, Chaoda Zheng, Shijia Chen, Mingdian Liu, Tongping Liu, Tengwei Luo, Yu Zhang 等
- **入库日期：** 2026-07-21
- **一句话说明：** 面向 **少步蒸馏** 的自回归驾驶世界模型（如 X-World）的 **免训练** 加速：沿 **跨 chunk** 而非跨去噪步缓存 DiT block 残差，双度量门控 + KV 写保护，约 **71% skip / 2.6–2.7×** 加速且 PSNR 几乎无损。

## 摘录 1：为何旧缓存全失效

- 交互式驾驶世界需 chunk 流式、动作条件、闭环无前瞻；X-World 类模型常蒸馏到 **S=4** 步。
- TeaCache / DeepCache / ΔDiT / BWCache 依赖 **跨去噪步** 冗余——少步蒸馏后步间几乎无可跳过冗余。
- chunk 边界动作 intentionally 非平滑；闭环禁止 look-ahead → 轨迹外推与 block-cascading 并行也不可用。

**对 wiki 的映射：** [`wiki/entities/paper-x-cache.md`](../../wiki/entities/paper-x-cache.md)；作为 [X-World](../../wiki/entities/paper-x-world.md) 的部署加速配套。

## 摘录 2：方法要点

- **轴切换：** 同一步、相邻 chunk 的物理场景连续性 → per-(t,b) block 残差缓存。
- **指纹：** 在 3D (F,H,W) latent 上采 32 token + 全局均值 + 展平动作向量。
- **双度量门：** cosine（全局方向）AND max token deviation（局部异常）；任一侧视角异常否决 reuse。
- **自适应阈值：** 每 cell EMA cosine 历史 + 硬地板 τ=0.97。
- **护栏：** KV-update chunk 强制全算（否则 PSNR 可从 ~53→21 dB）；anchor block(F_n=1) 永不跳；可选 step-0 保护。

## 摘录 3：数字与开源

- 城市场景 7-cam PSNR ≈ **51.4 dB**，skip **71.4%**，**2.7×**；高速/掉头类似。
- **开源：** 截至入库日 **未开源**。

**对 wiki 的映射：** 工程实践强调「少步蒸馏世界模型的缓存轴必须换到跨 chunk」。
