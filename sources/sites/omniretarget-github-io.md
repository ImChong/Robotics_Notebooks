# omniretarget.github.io（OmniRetarget 项目页）

- **标题：** OmniRetarget — Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction
- **类型：** site / project-page
- **URL：** <https://omniretarget.github.io/>
- **PDF 镜像：** <https://omniretarget.github.io/static/images/paper.pdf>
- **会议：** ICRA 2026（页面页眉标注）
- **入库日期：** 2026-06-08
- **配套论文：** [OmniRetarget（arXiv:2509.26633）](https://arxiv.org/abs/2509.26633) — 归档见 [`sources/papers/omniretarget_arxiv_2509_26633.md`](../papers/omniretarget_arxiv_2509_26633.md)

## 一句话摘要

Amazon FAR 等人提出的 **OmniRetarget** 官方站点：展示 **交互保留重定向** 在搬运、攀台、爬行、翻滚等 **loco-manipulation / 场景交互** 上的真机 RL 视频；提供 **物体位姿 / 尺寸 / 地形高度 / 跨 embodiment** 的交互式 3D 增广演示，以及与 GMR/PHC 基线的并排对比。

## 公开信息要点（截至入库日）

- **机构：** Amazon FAR、MIT、UC Berkeley、Stanford、CMU（* internship at Amazon FAR；† Amazon FAR team co-lead）。
- **真机声明：** 页面强调所有机器人视频为 **实时** 播放；策略仅用 **共享 5 项 reward + 4 项 domain randomization**，且 **仅依赖本体感知**。
- **数据规模（项目页摘要）：** 重定向轨迹 **9+ 小时**（论文正文为 8+ 小时；以各版本表述为准）。
- **演示板块：**
  - **Agile scene interaction** — Rolling、多组 Platform Climbing、Stepping、Crawling
  - **Loco-manipulation** — 8 组不同风格的 Box Carrying
  - **Data augmentation** — 单演示 → 物体初始位姿 / 尺寸 / 地形高度变化（每行中间列为原始演示）
  - **Interactive Demo** — 物体位姿、尺寸、地形、embodiment（T1/H1）的 iframe 3D 可视化
  - **Baseline comparison** — GMR/PHC vs OmniRetarget（Object Task / Climbing Task）
  - **LAFAN1 Retargeting** — Dance / Ground / Multiple Actions / Obstacles 序列可视化
- **代码 / 数据入口（页面未直接链出，与论文承诺一致）：**
  - 代码：<https://github.com/amazon-far/holosoma> — 归档见 [`sources/repos/holosoma.md`](../repos/holosoma.md)
  - 数据集：<https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset> — 归档见 [`sources/sites/omniretarget-dataset-huggingface.md`](omniretarget-dataset-huggingface.md)

## 为何值得保留

- **非 PDF 证据：** 增广对比、基线并排与 LAFAN1 序列比摘要更直观呈现 **interaction mesh** 与 **硬约束** 的收益。
- **与 arXiv 三角互证：** 项目页强调 ICRA 2026 与 9+ 小时数据，便于维护者核对与论文差异。
- **下游锚点：** [PHP（arXiv:2602.15827）](../papers/php_parkour_arxiv_2602_15827.md) 明确以 OmniRetarget 构建跑酷原子技能库。

## 关联资料

- 论文归档：[`sources/papers/omniretarget_arxiv_2509_26633.md`](../papers/omniretarget_arxiv_2509_26633.md)
- 代码框架：[`sources/repos/holosoma.md`](../repos/holosoma.md)
- 公开数据集：[`sources/sites/omniretarget-dataset-huggingface.md`](omniretarget-dataset-huggingface.md)
- 下游跑酷：[`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
