# demomimic.github.io（DemoMimic 项目页）

- **标题：** DemoMimic — One Demonstration, Many Objects
- **类型：** site / project-page
- **URL：** <https://demomimic.github.io/>
- **配套论文：** [DemoMimic（Stanford，2026）](../papers/demomimic_stanford_2026.md) — *One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry*
- **PDF：** <https://demomimic.github.io/static/demomimic_paper.pdf>
- **入库日期：** 2026-09-03

## 一句话摘要

Stanford **DemoMimic**（*Dexterous Motion Mimic*）官方站点：从 **单次人类示范** 出发，用 **接触局部几何** + **接触中心奖励（AR/SCR）** 在仿真中训 RL 教师，再蒸馏为 **腕部深度 + 本体** 的闭环模仿策略；真机 **16 物体 / 四类任务 / 两种灵巧手** 平均 **71%** 成功率，且相对 DexMachina* / HERMES* 基线 **sim-to-real gap 最小**。

## 公开信息要点（截至 2026-09-03 核查）

- **机构：** Stanford University（Satvik Sharma*、Samrat Sahoo*、Huang Huang*；Fei-Fei Li、Jiajun Wu、Dorsa Sadigh、Jeannette Bohg；* 同等贡献）。
- **页首按钮：**
  - **Paper** → 站内 PDF（`static/demomimic_paper.pdf`）
  - **arXiv · coming soon**（`aria-disabled="true"`）
  - **Code · coming soon**（`aria-disabled="true"`）
- **开源结论（步骤 2.5）：** **宣称将开源 / 待发布** — 截至入库日 **无** GitHub / Hugging Face / Zenodo 链接；勿写成已开源。
- **方法图（`method_figure.svg`）：** 训练：单次人类示范 → 仿真 **接触中心 RL 策略** \(\pi_{RL}\)（重度 domain randomization + AR/SCR）→ rollout 离线数据集 → 蒸馏 **深度条件模仿策略** \(\pi_{IL}\)。推理：\(\pi_{IL}\) 另条件于高层 **腕部轨迹引导** \(\pi_H\)（接近阶段粗引导），之后靠 **腕部相机深度 + 本体** 闭环完成任务。
- **接触中心奖励：**
  - **AR（Alignment Reward）** — 对齐接触连杆表面法向与物体法向，抑制仿真接触 artifact。
  - **SCR（Sustained Contact Reward）** — 在示范窗口内奖励连续接触 streak（二次增长），抑制仿真中「间歇松手」在真机上导致不可逆滑落。
- **真机结果板块（`chart.js` 数据，每物体 20 rollouts）：**
  - **Open the Box：** Wooden Box **82.74%**、Toolbox **87.33%**、Robot Hand Box **39.39%**、Shoe Box **77.69%**（页注：Robot Hand Box 盖几何与仿真差异大）。
  - **Lift the Lid：** Finemade Waffleiron **57.36%**、Gourima Waffleiron **44.82%**、Breakfast Maker **64.35%**。
  - **Move the Bottle：** Sunscreen **93.75%**、Souvenir Cup **75%**、Canned Drink **73.75%**、Water Bottle **78.75%**、Vitamin Container **86.25%**。
- **Sim→Real 对比（`chart.js` S2R，仿真 N=300 / 真机 N=20）：**
  - **Open the Box：** DexMachina* 95.8→**21.72%**；HERMES* 93.7→**3.37%**；DemoMimic 84.4→**82.71%**。
  - **Lift the Lid：** DexMachina* 80.1→**29.91%**；HERMES* 98.4→**39.1%**；DemoMimic 70.0→**70.59%**。
  - 消融：−AR−SCR / −SCR / −AR 真机显著低于完整奖励。
- **摘要声称：** 四类任务、两种灵巧手具身、**71%** 跨物体平均成功率；项目页 Tab 仅展示上述 **三项** 任务视频与 per-object 柱状图（第四类任务细节以 PDF 为准）。

## 为何值得保留

- **非 PDF 证据：** 多物体材质/尺度/摩擦变化下的真机视频与 **sim-vs-real 并排柱状图**，比摘要数字更直观呈现 **接触奖励对 sim-to-real 的决定性**。
- **开源边界清晰：** 页眉明确 Code/arXiv *coming soon*，便于 wiki 与 lint 跟进后续发布。
- **与 DexMachina 同族对照：** 项目页直接把 DexMachina* 作基线，并强调 **接触精度奖励** 而非仅位置/VOC 课程。

## 关联资料

- 论文归档：[`sources/papers/demomimic_stanford_2026.md`](../papers/demomimic_stanford_2026.md)
- 灵巧操作任务页：[wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md)
- 基线实体：[DexMachina](../../wiki/entities/paper-dexmachina.md)
