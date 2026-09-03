# DemoMimic: One Demonstration, Many Objects — Generalizing Manipulation via Local Contact Geometry（Stanford，2026）

> 来源归档（ingest）

- **标题：** One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry
- **简称：** DemoMimic（*Dexterous Motion Mimic*）
- **类型：** paper / dexterous-manipulation / sim2real / imitation-learning / reinforcement-learning / contact-rich / single-demo
- **项目页：** <https://demomimic.github.io/> — 归档见 [`sources/sites/demomimic-github-io.md`](../sites/demomimic-github-io.md)
- **PDF：** <https://demomimic.github.io/static/demomimic_paper.pdf>
- **arXiv：** <https://arxiv.org/abs/2609.01938>（[具身智能小站 2026-09-03 盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md) 已链；项目页按钮仍可能显示 coming soon）
- **代码：** 项目页标注 **Code · coming soon**（截至 2026-09-03 **无** GitHub URL）
- **机构：** Stanford University（Satvik Sharma*、Samrat Sahoo*、Huang Huang*；Fei-Fei Li、Jiajun Wu、Dorsa Sadigh、Jeannette Bohg）
- **入库日期：** 2026-09-03
- **一句话说明：** 单次人类示范 → 仿真 **接触中心 RL**（AR/SCR + 重度 DR）→ 蒸馏 **深度+本体** 低层模仿策略；高层 **腕部轨迹** 引导接近；真机 **16 物体 × 4 任务 × 2 灵巧手** 平均 **71%** SR，sim-to-real gap 小于 DexMachina* / HERMES*。

## 开源状态（项目页核查，2026-09-03）

| 组件 | 状态 |
|------|------|
| 项目页 PDF | 已发布（`static/demomimic_paper.pdf`） |
| arXiv | **待发布**（按钮 disabled，文案 *coming soon*） |
| 训练 / 推理代码 | **待发布**（按钮 disabled，文案 *coming soon*） |
| 权重 / 数据集 | **未见** 公开链接 |

**结论：** **宣称将开源 / 待发布** — 可引用项目页与 PDF 作方法证据；复现入口待后续更新。

## 摘要级要点

- **问题：** 多指灵巧操作数据难采集；人类示范可提供接触策略先验，但近期 sim-to-real RL 往往 **缺少显式精确接触奖励**，真机表现差，且 **难泛化到未见物体实例**。
- **核心思想：** 策略关注 **接触点局部物体几何**（local contact geometry），而非全局物体 mesh 或纯关键点位置匹配。
- **训练管线：**
  1. 单次人类示范 → 仿真中训 **\(\pi_{RL}\)**（PPO 类 on-policy RL + 接触中心奖励 + 广泛 domain randomization）。
  2. Rollout \(\pi_{RL}\) 生成离线数据集 → 蒸馏 **\(\pi_{IL}\)**（腕部相机 **depth** + proprioception 条件模仿策略）。
  3. 推理时 **\(\pi_H\)** 提供粗 **腕部轨迹** 引导接近阶段；之后 \(\pi_{IL}\) 闭环执行。
- **接触中心奖励：**
  - **AR** — 接触连杆法向与物体表面法向对齐。
  - **SCR** — 在示范时间窗内维持连续接触（二次 streak 奖励）。
- **真机（项目页 + 摘要）：** **71%** 平均成功率，覆盖 **16 物体、4 任务、2 灵巧手**；基线 DexMachina* / HERMES* 仿真高但真机骤降；完整 AR+SCR 使真机接近仿真。

## 核心摘录（面向 wiki 编译）

### 1) 真机 per-object 成功率（项目页 `chart.js`，20 rollouts/object）

| 任务 | 物体 | SR (%) |
|------|------|--------|
| Open the Box | Wooden Box | 82.74 |
| Open the Box | Toolbox | 87.33 |
| Open the Box | Robot Hand Box | 39.39 |
| Open the Box | Shoe Box | 77.69 |
| Lift the Lid | Finemade Waffleiron | 57.36 |
| Lift the Lid | Gourima Waffleiron | 44.82 |
| Lift the Lid | Breakfast Maker | 64.35 |
| Move the Bottle | Sunscreen | 93.75 |
| Move the Bottle | Souvenir Cup | 75.00 |
| Move the Bottle | Canned Drink | 73.75 |
| Move the Bottle | Water Bottle | 78.75 |
| Move the Bottle | Vitamin Container | 86.25 |

> 项目页 Tab 展示以上 **12 物体 / 3 任务**；摘要另称第四类任务与两种手型，细节以 PDF 为准。

### 2) Sim-to-real 成功率对比（项目页 `chart.js`）

**Open the Box**

| 方法 | Sim (%) | Real (%) |
|------|---------|----------|
| DexMachina* | 95.8 | 21.72 |
| HERMES* | 93.7 | 3.37 |
| Ours −AR−SCR | 82.9 | 28.16 |
| Ours −SCR | 83.2 | 37.83 |
| Ours −AR | 84.4 | 57.32 |
| **DemoMimic** | 84.4 | **82.71** |

**Lift the Lid**

| 方法 | Sim (%) | Real (%) |
|------|---------|----------|
| DexMachina* | 80.1 | 29.91 |
| HERMES* | 98.4 | 39.1 |
| Ours −AR−SCR | 76.0 | 37.94 |
| Ours −SCR | 69.9 | 50.9 |
| Ours −AR | 67.3 | 47.66 |
| **DemoMimic** | 70.0 | **70.59** |

### 3) 对 wiki 的映射

| 主题 | 目标页 |
|------|--------|
| 论文实体 | `wiki/entities/paper-demomimic.md` |
| 灵巧操作任务语境 | `wiki/tasks/manipulation.md` |
| Sim2Real / 接触奖励 | `wiki/concepts/sim2real.md`（交叉引用） |
| DexMachina 基线 | `wiki/entities/paper-dexmachina.md`（对照更新） |

## 推荐继续阅读

- [DemoMimic 项目页](https://demomimic.github.io/) — 真机多物体视频、sim-to-real 柱状图、方法总览图
- [DexMachina（arXiv:2505.24853）](../../wiki/entities/paper-dexmachina.md) — 项目页列作 * 号基线
- [CHORD](../../wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md) — 另一路「接触空间」奖励设计（wrench vs 局部几何+法向）
