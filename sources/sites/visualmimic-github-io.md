# visualmimic.github.io（VisualMimic 项目页）

- **标题：** VisualMimic — Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **类型：** site / project-page
- **URL：** <https://visualmimic.github.io/>
- **配套论文：** [VisualMimic（arXiv:2509.20322）](https://arxiv.org/abs/2509.20322) — 归档见 [`sources/papers/visualmimic_arxiv_2509_20322.md`](../papers/visualmimic_arxiv_2509_20322.md)
- **代码：** <https://github.com/visualmimic/VisualMimic> — 归档见 [`sources/repos/visualmimic.md`](../repos/visualmimic.md)
- **入库日期：** 2026-06-12

## 一句话摘要

Stanford **VisualMimic** 官方站点：展示 **egocentric 视觉 + 全身 loco-manipulation** 在 **多样时空**（晨/昏/夜、校园多地点）与 **多样接触模式**（手推、肩推、脚推、抬箱、踢球、盘带）下的真机结果；强调 **teacher–student 分层** 与 **关键点接口** 消融。

## 公开信息要点（截至入库日）

- **机构：** Stanford University（Shaofeng Yin*、Yanjie Ze*、Hong-Xing Yu；C. Karen Liu†、Jiajun Wu†）。
- **页首卖点：** *generalizable visuomotor skills across time & space*；*whole-body dexterity*。
- **演示板块：**
  - **时空泛化** — 同一 push box 任务在不同时段与 Stanford 校园多地标
  - **全身 dexterity** — Kick Box、Kick Ball、Lift Box、Push Box（手/肩/脚/双手）
  - **Framework** — 两阶段训练总览图
  - **Key Designs** — 高层/低层 **关键点接口** vs 仅 3 关键点；低层 **teacher–student 蒸馏** 消融
  - **Diverse Simulation Tasks** — Push/Kick/Drib/Lift/Reach/Balance 等
  - **Real-World Results** — 含失败恢复、gentle kick、foot–gantry 碰撞等
  - **Sim2Sim Results** — kick/lift/push 跨仿真迁移
- **Related Work 区块：** 链向 **TWIST**、**3D Diffusion Policies (IDP3)** 等同组工作。
- **BibTeX：** `@article{yin2025visualmimic,...}`。

## 为何值得保留

- **非 PDF 证据：** 户外光照/地面变化、多接触部位 push 比表格更直观呈现 **视觉 sim2real 鲁棒性**。
- **消融可视化：** 「仅 3 关键点」与「无蒸馏」对照是理解方法设计的关键非文字证据。
- **与 arXiv / GitHub 三角互证：** Sim2Sim 任务名与仓库 `sim2sim.py --task` 一致。

## 关联资料

- 论文归档：[`sources/papers/visualmimic_arxiv_2509_20322.md`](../papers/visualmimic_arxiv_2509_20322.md)
- 代码仓库：[`sources/repos/visualmimic.md`](../repos/visualmimic.md)
