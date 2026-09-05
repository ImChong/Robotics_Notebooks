# haofeixu.github.io/pointdit（PointDiT 项目页）

- **标题：** PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation
- **类型：** site / project-page
- **URL：** <https://haofeixu.github.io/pointdit/>
- **配套论文：** [arXiv:2607.02515](https://arxiv.org/abs/2607.02515) — [`sources/papers/pointdit_arxiv_2607_02515.md`](../papers/pointdit_arxiv_2607_02515.md)
- **代码：** <https://github.com/google-research/pointdit> — [`sources/repos/pointdit.md`](../repos/pointdit.md)
- **入库日期：** 2026-09-05

## 一句话摘要

Google / ETH / 图宾根 / Microsoft / KE:SAI / TUM 官方站点：单张 RGB + 高斯噪声 → 一步稠密 3D 点图；交互对照 MoGe-2、GeometryCrafter，并展示 1–4 步细化与「同架构回归 vs 生成」受控实验。

## 公开信息要点（截至 2026-09-05 复核）

- **页首：** ICML 2026；作者单位 1 Google · 2 ETH Zurich · 3 University of Tübingen / Tübingen AI Center · 4 Microsoft · 5 KE:SAI · 6 TUM。
- **步骤 2.5：** 页上给出论文 / 项目页叙事，官方仓见 README 与 GitHub org `google-research/pointdit`；权重在 Hugging Face `haofeixu/pointdit`。→ **已开源**。
- **核心叙事：** 回归平均掉细结构；潜扩散被 VAE 瓶颈先毁掉细节；像素空间扩散保住椅背细杆与透明物。
- **方法一句：** 噪声点图按 XYZ 通道分 patch；冻结 DINOv3 编码干净图像；plain Transformer 去噪。无 VAE、无两阶段 tokenizer。
- **受控实验：** 只把噪声与 timestep 换成确定性零 → 同网变成回归器；生成式 BF1 10.90 → **13.92**。
- **前瞻：** 同一 ViT 配方可扩到外观+几何联合、相机/多视图条件，走向无 VAE 的 3D/4D 生成。

## 为何值得保留

- 步骤 2.5 的项目页入口；3D 交互对照是论文表读不懂时的主证据。
- 把「点图 = 多通道图像」写进读者可拖拽的演示，而不是只留在 PDF。

## 关联资料

- 论文：[`sources/papers/pointdit_arxiv_2607_02515.md`](../papers/pointdit_arxiv_2607_02515.md)
- 代码：[`sources/repos/pointdit.md`](../repos/pointdit.md)
- Wiki：[wiki/entities/paper-pointdit.md](../../wiki/entities/paper-pointdit.md)
