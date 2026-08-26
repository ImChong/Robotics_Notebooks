# dreammimic.github.io（DreamMimic 项目页）

- **标题：** DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model
- **类型：** site / project-page
- **URL：** <https://dreammimic.github.io/>
- **配套论文：** [DreamMimic（arXiv:2608.22278）](https://arxiv.org/abs/2608.22278) — 归档见 [`sources/papers/dreammimic_arxiv_2608_22278.md`](../papers/dreammimic_arxiv_2608_22278.md)
- **代码：** <https://github.com/DreamMimic/DreamMimic> — 归档见 [`sources/repos/dreammimic.md`](../repos/dreammimic.md)
- **入库日期：** 2026-08-26

## 一句话摘要

DreamMimic 官方站点：用演示视频说明 **特权教师 → 视觉学生** 的全身 loco-manipulation，并展示 SMPL-X / Unitree G1、OMOMO / BEHAVE 与 Isaac Gym→Lab 的定性结果。

## 公开信息要点（截至入库日）

- **作者：** Jie Yin（Independent）、Xingyu Lai（Tsinghua University）；共同一作。
- **页首链接：** Paper / arXiv / Video / **Code (Coming soon)**。
- **方法 TL;DR：** RSSM 潜动力学 + 交互辅助预测 + Performance-Conditioned Guidance（PCG）稳定 DAgger+RL 蒸馏。
- **感知：** 学生在线不看特权状态；深度 + 分割驱动世界模型特征。页上分别给 SMPL-X 与 G1 的深度/分割样例。
- **演示板块：**
  - SMPL-X × OMOMO：椅 / 桌 / 大箱 / 塑料箱 / 小箱 / 行李箱 × 两条序列
  - Unitree G1 × OMOMO：四物体 × 两条序列（42 DoF）
  - BEHAVE 长程接触行走
  - Sim2Sim：Isaac Gym 训练 → Isaac Lab 同序列
- **BibTeX：** `@misc{yin2026dreammimiclearningvisuomotorwholebody,...}`。

## 开源核查（步骤 2.5）

**宣称将开源 / 占位仓。** 页上 Code 按钮写 Coming soon；对应 GitHub 无训练/推理脚本。勿写成已开源。

## 为何值得保留

- **非 PDF 证据：** 跨形态与跨仿真器的接触行为比表格更能说明「仍是仿真 GT 感知」。
- **与 VisualMimic 对照：** 同为视觉全身，这里走 RSSM+PCG，那边走关键点分层 + 真机零样本。

## 关联资料

- 论文归档：[`sources/papers/dreammimic_arxiv_2608_22278.md`](../papers/dreammimic_arxiv_2608_22278.md)
- 代码仓库：[`sources/repos/dreammimic.md`](../repos/dreammimic.md)
