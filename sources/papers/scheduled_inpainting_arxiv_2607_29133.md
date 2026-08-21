# Interactive Generative Motion Editing via Scheduled Inpainting

> 来源归档（ingest）

- **标题：** Interactive Generative Motion Editing via Scheduled Inpainting
- **类型：** paper
- **机构：** DisneyResearch\|Studios · ETH Zürich
- **原始链接：**
  - Disney Research 项目页：<https://studios.disneyresearch.com/2026/07/30/interactive-generative-motion-editing-via-scheduled-inpainting/>（见 [`sources/sites/disney-scheduled-inpainting-gme.md`](../sites/disney-scheduled-inpainting-gme.md)）
  - arXiv：<https://arxiv.org/abs/2607.29133>
  - PDF：<https://studios.disneyresearch.com/app/uploads/2026/07/Interactive-Generative-Motion-Editing-via-Scheduled-Inpainting-Paper.pdf>
- **入库日期：** 2026-08-21
- **一句话说明：** 提出 **interactive generative motion editing（GME）** 任务与 **scheduled inpainting** 推理框架：在 **无需额外训练** 的前提下，用用户可控的 inpainting 日程 + 时空 mask，在保留已有 MoCap/关键帧片段的同时做结构性编辑（延长、拼接、合成、直接操控），并接 IBMM / SF-control 等 direct-manipulation 扩散模型。

## 核心论文摘录（MVP）

### 1) 任务：统一「生成式创作」与「保留式编辑」

- **链接：** <https://arxiv.org/abs/2607.29133>
- **摘录要点：** 传统 motion warping 只能小改，大改会 warp 伪影；纯生成模型可用稀疏空间约束从零创作，却 **不能交互式编辑来自外部的 exemplar clip**。文本编辑（MotionFix / MotionLab）粒度太粗。本文定义 **interactive generative motion editing**：对 **非模型自生成** 的已有动画做 **大结构改动**（延长、拼接、合成），同时保留 **空间+时间上的 direct manipulation**。
- **对 wiki 的映射：**
  - [Scheduled Inpainting / GME（Disney）](../../wiki/entities/paper-scheduled-inpainting-gme.md) — 任务定义与范式
  - [Generative Motion Rig（Disney）](../../wiki/entities/generative-motion-rig.md) — 同组 DCC generative keyframing 姊妹（插件未开源）
  - [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md) — 艺术家端编辑 vs 机器人轨迹后处理

### 2) Scheduled inpainting：日程 + 时空 mask + 对齐空间

- **链接：** PDF 同上
- **摘录要点：**
  - 推理时混合：$\widehat{\mathcal{M}_{gen}^{0}} = \alpha^{t}\mathcal{M}_{base} + (1-\alpha^{t})\mathcal{M}_{gen}^{0}$，其中 $\alpha^{t}=\alpha_{time}^{t}\times\alpha_{mask}$。
  - **Inpainting schedule** $\sigma_s/\sigma_e$：高噪声段完全保留 base，低噪声段完全生成，中间线性插值——用户调节 **保留 vs 生成** 强度。
  - **Spatiotemporal mask** $\alpha_{mask}\in\mathbb{R}^{T\times J}$：按帧/关节控制保留权重；direct manipulation 时在约束邻域用 Gaussian kernel 降低 inpainting 权重。
  - **Inpainting space**：首帧对齐原点、首尾方向对齐 +x、逐序列 0-mean/1-var 归一化；root 用差分坐标，其余关节 root-relative——避免反向行走/不同速度 clip 混合时的静态平均伪影。
  - **Training-free**：可接任意支持 direct manipulation 的预训练扩散模型（文内主结果用 **IBMM**、**SF-control**）。
- **对 wiki 的映射：**
  - [Scheduled Inpainting / GME](../../wiki/entities/paper-scheduled-inpainting-gme.md) — 公式、Mermaid 流程与 ablation
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 推理期编辑技法补充

### 3) 应用：延长、拼接、合成、循环适配、重定时

- **链接：** PDF 同上
- **摘录要点：**
  - **Extension**：在 $[t_s,t_e)$ 设 mask=0 生成新段，其余 inpaint 保留；生成段仍可被 direct manipulation。
  - **Stitching**：两段 base clip 中间生成过渡；naive stitch 有 discontinuity，scheduled inpainting 生成自然过渡（依赖 normalization）。
  - **Compositing**：沿 **关节维** 混合不同 clip（如上半身/下半身）。
  - **Animation cycle 适配**：root 完全生成或约束到用户曲线，其余关节 inpaint 循环动作——消除 foot sliding 同时保持 Zombie/Chicken 等 stylized cycle。
  - **Retiming**：warp 后作 base，以 $\alpha_{mask}=0.8$ 做 generative recovery。
- **对 wiki 的映射：**
  - [Scheduled Inpainting / GME](../../wiki/entities/paper-scheduled-inpainting-gme.md) — 应用表与制片读法
  - [机器人关键帧与运动编辑工具](../../wiki/entities/robot-motion-keyframe-editors.md) — 机器人侧 stitch/extend 对照（URDF/NPZ vs 生成式 prior）

### 4) 评测、基线与艺术家可用性

- **链接：** PDF 同上
- **摘录要点：**
  - **基线**：MotionLab（0.88 s/样本、非交互）、CondEditor（CondMDI + 编辑训练）、DNO（~10.3 s/20 步）、noise-inversion（25 步不足、400 步才高保真且非实时）。
  - **定量**：随机位移末帧时 foot-sliding 仅增 ~1 mm/frame，L2P/L2R 相对 IBMM 明显更低（Table 1）；schedule ablation 显示 $\sigma_s=500,\sigma_e=50$ 在保留与编辑间最佳平衡。
  - **Usability**：两位专业数字艺术家 1 小时内用 running/crawling clip 库完成 parkour 镜头修复（穿透、naive stitch 断点）；非破坏性编辑与 crawl→run 自动过渡获好评；扩散开销导致帧率低于传统插值。
- **对 wiki 的映射：**
  - [Scheduled Inpainting / GME](../../wiki/entities/paper-scheduled-inpainting-gme.md) — 「结论」与局限

## 开源核查（步骤 2.5）

| 项 | 状态（截至 2026-08-21） |
|----|-------------------------|
| Disney Research 页 | **Download Publication PDF** + arXiv 链；**无 GitHub / HF / 演示代码** |
| 代码 / DCC 插件 | **确认未开源** |
| 结论 | 按 **闭源方法论文** 归档；知识以 PDF、arXiv 与配套视频为准 |

## 当前提炼状态

- [x] 方法（schedule / mask / inpainting space）与应用已摘录
- [x] 基线对比与艺术家测试已摘录
- [x] 开源边界已核查（无代码）

## BibTeX

```bibtex
@article{agrawal2026scheduled,
  title   = {Interactive Generative Motion Editing via Scheduled Inpainting},
  author  = {Agrawal, Dhruv and Borer, Dominik and V{\"o}geli, Luca and
             Sumner, Robert W. and Guay, Martin and Buhmann, Jakob},
  journal = {arXiv preprint arXiv:2607.29133},
  year    = {2026}
}
```
