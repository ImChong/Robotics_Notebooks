# GenIA: Generative Reconstruction with Test-Time Input Alignment

> 来源归档（论文摘录；知识归纳见 wiki 实体页）

- **标题：** GenIA: Generative Reconstruction with Test-Time Input Alignment
- **类型：** paper
- **作者：** Stefano Esposito, Naama Pearl, Polina Karpikova, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Andreas Geiger, Jonathon Luiten
- **机构：** 图宾根大学图宾根人工智能中心（Tübingen AI Center, University of Tübingen）；元宇宙人工智能（Meta Reality Labs）
- **arXiv：** https://arxiv.org/abs/2610.12388（v1，2026-10-08）
- **项目页：** https://facebookresearch.github.io/GenIA/
- **代码：** https://github.com/facebookresearch/GenIA
- **入库日期：** 2026-10-13
- **一句话说明：** 在推理时以深度、可见性和可微渲染约束冻结的 SAM 3D 生成先验，将其从单图静态重建扩展到多视角和动态物体，同时保留生成式补全能力。

## 核心摘录（策展，非全文）

1. **问题与定位：** SAM 3D 等生成式 3D 基础模型能补全观测之外的几何，但生成结果可能与输入的几何、外观或姿态不一致。GenIA 将对齐放在测试时完成，不重训基础模型。
2. **几何与姿态：** 由深度几何估计物体平移和尺度，保留生成模型的旋转先验，再通过 ICP 将粗形状配准到输入点图；多视角/视频用跨观测聚合提高姿态稳定性。
3. **外观对齐：** 根据粗几何投影可见性偏置 SAM 3D 的注意力，采样过程中加入可微渲染的 RGB、结构相似度、感知和深度引导，并按体素可见性融合多视角/帧预测。
4. **可选测试时优化（TTR）：** 每物体 100 次迭代，仅优化外观 latent、轻量高斯解码器适配器及物体位姿；所有预训练主体参数冻结，latent 锚定项帮助保留未观测区域的生成先验。
5. **动态扩展：** 对逐帧外部形状建立规范空间对应，跨时间融合外观并恢复稳定世界位姿。该版本依赖 ActionMesh 提供动态形状/形变，并非从视频直接预测完整动态几何。
6. **评测口径：** 静态基准为 GSO-30 与 CO3D，动态为 ActionBench 与 DAVIS。报告训练视角前景 PSNR、held-out LPIPS/CLIP-I，部分设置报告 co-PSNR；DAVIS 没有测试视角，使用合成新视角近似评估。作者称在保持生成补全的同时改善输入对齐，真实场景的 held-out 指标尤其有利。
7. **运行代价：** 项目页报告 SAM 3D 在 A100 上单图中位数约 25 秒；动态 DAVIS 管线约为 SAM 3D 单图推理的 13 倍（含 ActionMesh、含 TTR），对照 Lift4D/HiMoR 的 160/250 倍。CO3D 单视图峰值显存为 19.7 GB（不含 TTR）、31.8 GB（含 TTR），SAM 3D 为 18.0 GB。
8. **边界：** 粗形状仍由生成先验给出，尚未直接以观测轮廓/深度稳健约束；动态形状取决于外部预测器；旋转仍以先验为主，ICP 对噪声敏感。大帧间运动下几何和位姿误差会传播。

## 对 wiki 的映射

- [GenIA 论文与代码实体页](../../wiki/entities/paper-genia-arxiv-2610-12388.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2610.12388>
- 官方项目页：<https://facebookresearch.github.io/GenIA/>
- 官方代码：<https://github.com/facebookresearch/GenIA>
