# GenCAD: Image-Conditioned Computer-Aided Design Generation（arXiv:2409.16294）

> 来源归档（ingest）

- **标题：** GenCAD: Image-Conditioned Computer-Aided Design Generation with Transformer-Based Contrastive Representation and Diffusion Priors
- **缩写：** **GenCAD**
- **类型：** paper / CAD 程序生成 / 图像条件生成 / 对比学习 / 潜扩散
- **arXiv：** <https://arxiv.org/abs/2409.16294>（PDF：<https://arxiv.org/pdf/2409.16294>）
- **项目页：** <https://gencad.github.io/>
- **代码：** <https://github.com/ferdous-alam/GenCAD>
- **作者：** Md Ferdous Alam, Faez Ahmed（MIT）
- **入库日期：** 2026-05-21
- **一句话说明：** **图像条件**生成 **参数化 CAD 命令序列（CAD program）**，经几何内核编译为 **B-rep** 实体；管线为 **因果 Transformer 自编码器** + **CAD–图像对比对齐** + **潜空间条件扩散先验** + **序列解码**，在 **DeepCAD** 数据划分上支持 **生成** 与 **检索**。

## 摘要级要点

- **动机：** 工程 CAD 以 **B-rep** 与 **可编辑命令历史** 为真值，而 mesh / 体素 / 点云便于学习却牺牲 **精度与可修改性**；GenCAD 直接输出可编译的 **CAD program**。
- **四段架构（项目页与论文一致）：**
  1. **CAD 序列自编码：** 因果 Transformer 将 CAD program 映射到连续潜空间（延续 DeepCAD 式 **60×17** 命令–参数矩阵与因果掩码叙事）。
  2. **多模态对比：** 对齐 **CAD 潜向量** 与 **CAD 渲染图** 潜向量（CLIP 式对比损失；条件实验用 **等轴测渲染**）。
  3. **条件潜扩散：** 以图像潜为条件，在 CAD 潜空间上训练 **denoising prior**（ResNet-MLP 去噪器 + 条件拼接）。
  4. **解码：** 将采样得到的 CAD 潜向量解码为 **参数化命令序列**，再经 **OpenCascade** 等内核编译 B-rep。
- **任务：** **图像 → CAD 生成**（同图可多采样多样性）、**图像 → CAD 检索**（在约 7k 级程序库中 top-k）。
- **评测：** 与 DeepCAD、BrepGen 等基线共享 **COV / MMD / JSD** 等生成多样性指标（论文与 rebuttal 强调在统一重采样规模下与 DeepCAD 设定对齐）；图像条件实验在测试渲染图上逐图生成再聚合指标。
- **交互演示：** 项目页提供浏览器端 **图像条件生成** 与 **检索** Demo（以站点当前版本为准）。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/gencad.md`](../../wiki/entities/gencad.md)
- 互链：[`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md)、[`wiki/entities/gencad-3d.md`](../../wiki/entities/gencad-3d.md)（后续 3D 模态扩展）
