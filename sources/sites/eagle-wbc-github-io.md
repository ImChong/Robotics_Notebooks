# EAGLE-WBC（项目页）

- **标题：** EAGLE — Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control
- **类型：** site / project-page
- **URL：** <https://eagle-wbc.github.io/>
- **arXiv：** <https://arxiv.org/abs/2602.02960>
- **会议：** ICRA 2026
- **入库日期：** 2026-09-13
- **配套论文：** [EAGLE-WBC（arXiv:2602.02960）](../papers/eagle_wbc_arxiv_2602_02960.md)

## 一句话摘要

上海交大 / 上海 AI Lab 提出的 **跨本体人形 WBC** 官方站点：展示 **统一高维指令接口**（速度 + 高度 + pitch）与 **generalist–specialist 迭代蒸馏** 管线；仿真 **5** 机、真机 **4** 机（H1 / G1 / N1 / T1 等）共用一份策略权重的演示视频与方法图。

## 公开信息要点（截至 2026-09-13 核查）

- **机构：** 上海交通大学（SJTU）；上海人工智能实验室（Shanghai AI Lab）
- **作者：** Quanquan Peng*、Yunfeng Lin*、Yufei Xue、Jiangmiao Pang、Weinan Zhang（* equal contribution）
- **核心叙事：**
  - **统一命令接口** — $c_t = [v_t, b_t]$：线速度 $v_x, v_y$、角速度 $\omega$ + 基座高度 $h$、躯干 pitch $p$
  - **迭代蒸馏** — 每轮 fork generalist → per-robot specialist 精修 → DAgger 风格回蒸至新 generalist
  - **Fleet-level 部署** — 一份策略跨 Unitree H1、G1、Fourier N1、Booster T1 等
- **演示区块：**
  - Velocity commands（G1 / H1 / N1 / T1 变速）
  - Height commands（跨本体蹲/站）
  - Pitch commands（躯干前倾等）
- **方法图：** (a) 统一指令与观测；(b) generalist–specialist 蒸馏循环
- **代码 / 数据（步骤 2.5）：** 页面 **无** GitHub、Hugging Face、Zenodo 或 ModelScope 链接；Header / Footer / Resources 区均未列出代码仓库 → **确认未开源**（截至入库日）。
- **BibTeX：** 页面提供 `@misc{peng2026eaglewbc, ...}`

## 为何值得保留

- **非 PDF 证据：** 四机真机并排视频比 arXiv 摘要更易判断「一份权重跨本体」的工程可信度。
- **指令接口可视化：** 速度 / 高度 / pitch 三分演示区直接对应论文统一命令设计。
- **开源跟进锚点：** 后续若放出仓库，应从此页与论文双向更新 `sources/repos/`。

## 关联资料

- 论文归档：[`sources/papers/eagle_wbc_arxiv_2602_02960.md`](../papers/eagle_wbc_arxiv_2602_02960.md)
- Paper Notebooks 深读：[`humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md`](../papers/humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md)
- Wiki 实体：[`wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md`](../../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md)
