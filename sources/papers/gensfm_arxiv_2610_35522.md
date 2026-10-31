# GenSfM 论文

> 来源归档

- **标题：** Generative Structure from Motion with Native 3D Diffusion
- **论文：** <https://arxiv.org/html/2610.35522v1>（arXiv v1，2026-10-26）
- **项目：** <https://dotb1ue.github.io/GenSfm/>
- **代码：** <https://github.com/dotb1ue/GenSfm>
- **作者：** Haobin Duan, Binbin Huang, Mochu Xiang, Yiqun Zhao, Zibo Zhao, Yi Ma, Shenghua Gao
- **机构：** 香港大学、Transengram、腾讯
- **入库日期：** 2026-10-31
- **映射：** [GenSfM](../../wiki/entities/paper-gensfm.md)

## 核心摘录

1. 多模态 DiT 共同去噪共享形状和每视角 UV 体积位姿潜变量；DLT 和 RQ 分解恢复相机。
2. 细阶段用估计相机投影活跃体素，从多视图采样语义与纹理特征并平均，生成细几何和外观。
3. Toys4k 和 GSO 各以 400 个物体、每物体两组四视图评测；与 TRELLIS-M 对比应遵循论文相同的相似 ICP 对齐协议。
4. 限制：论文评估聚焦四视图物体，分布外相机误差会影响细阶段，背面手性歧义未由相对姿态指标评估。
