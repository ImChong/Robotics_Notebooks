# Sparsh: Self-supervised Touch Representations for Vision-based Tactile Sensing

> 来源归档（ingest）

- **标题：** Sparsh: Self-supervised touch representations for vision-based tactile sensing
- **类型：** paper / tactile-representation / ssl / foundation-model / vbts / tacbench
- **会议：** CoRL 2024（Conference on Robot Learning）
- **OpenReview：** <https://openreview.net/forum?id=xYJn2e1uu8>
- **arXiv abs：** <https://arxiv.org/abs/2410.24090>
- **arXiv HTML：** <https://arxiv.org/html/2410.24090>
- **PDF：** <https://arxiv.org/pdf/2410.24090>
- **项目页：** <https://sparsh-ssl.github.io/>
- **机构：** Meta FAIR；University of Washington；Carnegie Mellon University
- **入库日期：** 2026-09-23
- **一句话说明：** 在 **460k+** 无标触觉图像上用 **MAE / DINO / IJEPA / VJEPA** 等 SSL 训练 **跨 DIGIT、GelSight 2017、GelSight Mini** 的通用触觉表征；**TacBench** 六任务 frozen probe 评测平均较 task-specific E2E **+95.1%**（33–50% 标注预算），**DINO / IJEPA**  latent 空间最优。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://sparsh-ssl.github.io/> — Paper / **Code** / **Dataset** 按钮 |
| GitHub 实现 | **已开源** — [`github.com/facebookresearch/sparsh`](https://github.com/facebookresearch/sparsh)（PyTorch；仓库状态 **ARCHIVED**） |
| 预训练权重 | **已发布** — [Hugging Face `facebook/sparsh`](https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328) |
| 训练数据 | 项目页 Dataset 链至 HF collection（含 Touch-Slide 等） |
| 后续扩展 | [`facebookresearch/sparsh-multisensory-touch`](https://github.com/facebookresearch/sparsh-multisensory-touch) — Sparsh-X / Sparsh-Skin（CoRL 2025，非本文主仓） |
| 结论 | **已开源**（代码 + 权重 + 基准数据入口） |

## 摘要级要点

- **问题：** 视觉触觉传感器（VBTS）任务/传感器专用 E2E 模型重复造轮子；力、滑移等标注难规模化；传感器照明/标记差异大。
- **Sparsh 家族：** 适配 MAE、DINO/DINOv2、IJEPA、V-JEPA；**背景减除** + **~80 ms**（两帧 stride-5 @60 FPS）通道拼接 tokenization 对 slip/pose 等时序任务关键。
- **数据：** 整合 Touch-Slide、YCB-Slide、Touch-and-Go、ObjectFolder 等，合计 **~661k** 图，**462.7k** 用于 SSL。
- **TacBench [T1–T6]：** 力估计/力场可视化、滑移检测、SE(2) 位姿、抓取稳定、织物识别、Bead Maze 扩散策略；frozen encoder + attentive probe vs E2E。
- **下游：** 实时法向/剪切场 DPT 解码；Bead Maze 上 Sparsh 特征略优于 E2E encoder。
- **对比：** 与 T3、UniT 等并发工作；Sparsh 覆盖 **三种主流 VBTS 家族** + 标准化 TacBench。

## 核心论文摘录（MVP）

### 1) SSL 配方：背景减除 + 短窗时序 token

- **链接：** <https://arxiv.org/html/2410.24090#S3>
- **摘录要点：** markerless DIGIT / GelSight Mini 做背景减除；$I_t \oplus I_{t-5} \rightarrow 6$ 通道；V-JEPA 用 4 帧 clip；推理可达 **112 FPS**（RTX 3080）。
- **对 wiki 的映射：**
  - [触觉传感](../../wiki/concepts/tactile-sensing.md) — VBTS 表征学习轴
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 2) TacBench 分层：触觉属性 / 物理感知 / 操作规划

- **链接：** <https://arxiv.org/html/2410.24090#S4>
- **摘录要点：** [T1] 三轴力 RMSE；[T2] slip；[T3] SE(2) pose；[T4] grasp stability；[T5] textile；[T6] Bead Maze + Diffusion Policy；33% 标注下 Sparsh 平均 **+95.1%** vs E2E。
- **对 wiki 的映射：**
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 下游策略可插拔 Sparsh 编码器

### 3) 力场解码与操作策略

- **链接：** <https://arxiv.org/html/2410.24090#S5>；项目页 Normal and shear field / Bead Maze
- **摘录要点：** frozen Sparsh + DPT 解码器 photometric warp 损失预测法向/剪切场；Franka teleop Bead Maze 上 Sparsh 特征策略略优 E2E。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)

## 对 wiki 的映射（汇总）

- 交叉引用：[OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)（Sparsh 作触觉 backbone 对照）
- 概念：[触觉传感](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
- 项目页归档：[`sources/sites/sparsh-ssl-github-io.md`](../sites/sparsh-ssl-github-io.md)

## 当前提炼状态

- [x] CoRL 2024、SSL 数据规模、TacBench 结果、Meta GitHub/HF 开源核查已摘录
- [x] 与 [`sources/sites/sparsh-ssl-github-io.md`](../sites/sparsh-ssl-github-io.md) 互证

## BibTeX

```bibtex
@inproceedings{higuera2024sparsh,
  title={Sparsh: Self-supervised touch representations for vision-based tactile sensing},
  author={Higuera, Carolina and Sharma, Akash and Bodduluri, Chaitanya Krishna and Fan, Taosha and Lancaster, Patrick and Kalakrishnan, Mrinal and Kaess, Michael and Boots, Byron and Lambeta, Mike and Wu, Tingfan and Mukadam, Mustafa},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024},
  url={https://openreview.net/forum?id=xYJn2e1uu8},
  eprint={2410.24090},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
}
```
