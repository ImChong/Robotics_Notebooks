# sparsh-ssl.github.io（Sparsh 项目页）

- **标题：** Sparsh: Self-supervised touch representations for vision-based tactile sensing
- **类型：** site / project-page
- **URL：** <https://sparsh-ssl.github.io/>
- **入库日期：** 2026-09-23
- **配套论文：** [Sparsh（arXiv:2410.24090 / CoRL 2024）](https://arxiv.org/abs/2410.24090) — 归档见 [`sources/papers/sparsh_arxiv_2410_24090.md`](../papers/sparsh_arxiv_2410_24090.md)
- **机构：** Meta FAIR；University of Washington；Carnegie Mellon University
- **代码 / 数据（截至 2026-09-23 项目页核查）：**
  - **Code：** [`github.com/facebookresearch/sparsh`](https://github.com/facebookresearch/sparsh)
  - **Dataset / Models：** [Hugging Face `facebook/sparsh` collection](https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328)

## 一句话摘要

Sparsh 官方项目页：展示 **460k+ SSL 预训练**、**TacBench 六任务** 雷达图、**跨 DIGIT / GelSight** 传感器泛化，以及 **法向/剪切场实时解码**、SE(2) pose、Bead Maze 策略等定性结果。

## 公开信息要点（截至入库日）

- **顶部资源按钮：** Paper / **Code** / **Dataset** 三键齐全。
- **Overview：** MAE、DINO、IJEPA、V-JEPA 家族；背景减除 + ~80 ms 窗口；661k 策展数据。
- **Demo 区块：** Normal and shear field decoding、Pose estimation、Bead Maze（Diffusion Policy + frozen Sparsh）。
- **实现致谢：** 基于 MAE / DINO / IJEPA / VJEPA / Diffusion Policy 等开源仓库。

## 为何值得保留

- **开源入口聚合：** 页上 Code/Dataset 链为 lint 步骤 2.5 的一手证据。
- **非 PDF 证据：** TacBench 任务示意图与跨传感器图比表格更易交叉引用。

## 关联资料

- 论文归档：[`sources/papers/sparsh_arxiv_2410_24090.md`](../papers/sparsh_arxiv_2410_24090.md)
- Wiki 交叉：[OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)（Sparsh 作触觉 encoder 对照）
