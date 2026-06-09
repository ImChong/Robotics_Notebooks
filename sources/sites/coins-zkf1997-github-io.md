# zkf1997.github.io/COINS（COINS 项目页）

- **标题：** COINS — Compositional Human-Scene Interaction Synthesis with Semantic Control
- **类型：** site / project-page
- **URL：** <https://zkf1997.github.io/COINS/index.html>
- **配套论文：** [COINS（arXiv:2207.12824）](https://arxiv.org/abs/2207.12824) — 归档见 [`sources/papers/coins_arxiv_2207_12824.md`](../papers/coins_arxiv_2207_12824.md)
- **代码：** <https://github.com/zkf1997/COINS> — 归档见 [`sources/repos/coins.md`](../repos/coins.md)
- **入库日期：** 2026-06-09

## 一句话摘要

ETH Zürich / Google 团队的 **COINS** 官方站点：展示 **动作–物体实例** 语义控制下的虚拟人–场景交互合成、**未见 action–object 组合** 的 retarget、以及 **仅用原子数据训练** 的 **复合交互** 生成；含 **浏览器 3D 交互 demo**、**PROX-S** 数据说明与相对基线的定性对比。

## 公开信息要点（截至入库日）

- **方法卖点：** Transformer 生成模型将人体表面点与 3D 物体 **联合编码** 于统一潜空间；交互语义经 **positional encoding** 嵌入；支持 **骨盆 → 身体** 两阶段生成。
- **演示板块：**
  - **Overview** — 单行/未见组合/复合交互三行 teaser
  - **Demo** — 按 `action-object category-object id` 列表随机生成并可视化（WebGL 查看器）
  - **Method** — 场景 + 高亮物体实例 + 语义动作的两阶段 pipeline 图
  - **PROX-S Dataset** — 实例分割 + 逐帧 SMPL-X + action–object 语义列表
  - **Results** — 相对基线物理可行性；未见 action–object；体型控制；噪声分割物体
  - **Video** — 嵌入演示视频
- **数据下载：** 指向 Google Drive（与仓库 README 一致）；代码仓库提供 `load_interaction` 等工具脚本。
- **BibTeX：** 页面提供 ECCV 2022 引用块。

## 为何值得保留

- **非 PDF 证据：** 交互式 demo 与 loft 场景填充视频直观展示 **语义组合** 与 **实例级物体选择**，便于核对论文 Fig.1/7。
- **与 arXiv / 仓库三角互证：** 项目页强调 PROX-S 字段定义与 demo 交互格式（`action-object category-object id`），与 [`two_stage_sample.py`](https://github.com/zkf1997/COINS/blob/main/interaction/two_stage_sample.py) 的 `--interaction 'sit on-chair+touch-table'` 语法一致。
- **机器人侧锚点：** 合成 **带语义的人–场景交互** 可用于 AR/VR 与 **感知算法训练数据**；与 [CRISP](../../wiki/methods/crisp-real2sim.md) 共享 PROX 基准语境。

## 关联资料

- 论文归档：[`sources/papers/coins_arxiv_2207_12824.md`](../papers/coins_arxiv_2207_12824.md)
- 代码仓库：[`sources/repos/coins.md`](../repos/coins.md)
