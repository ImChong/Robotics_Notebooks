# humanoid-robot-learning-paper-notebooks

> 来源归档（ingest）

- **标题：** Robot Learning Paper Notebooks
- **类型：** site
- **链接：** https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html
- **关联仓库：** https://github.com/ImChong/Robot_Learning_Paper_Notebooks（公开，BSD-3-Clause）
- **阅读进度：** https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json
- **入库日期：** 2026-08-03
- **更名日期：** 2026-08-28（GitHub / Pages：`Humanoid_Robot_Learning_Paper_Notebooks` → `Robot_Learning_Paper_Notebooks`）
- **一句话说明：** 姊妹站点，把人形机器人学习方向的论文按 14 个分类整理成逐篇深读笔记（Jekyll 静态站），是本库 `wiki/overview/paper-notebook-category-*` 分类页与 `wiki/entities/paper-notebook-*` 实体页的上游原始资料。

## 站点规模（入库日核对）

| 口径 | 数值 | 说明 |
|------|------|------|
| 站点首页自述 | 305 papers | 首页 `📄 305 papers` 徽标 |
| 已生成 HTML 深读笔记 | 289 条 | 本库快照 [`schema/paper-notebook-index.json`](../../schema/paper-notebook-index.json) |
| 分类页清单条目 | 518 条 | [`schema/paper-notebook-categories.json`](../../schema/paper-notebook-categories.json) 的 `count`，含「待深读」与「见 wiki 实体页」条目，且高影响力精选与各主题分类存在跨类重复计入 |

> 三个口径不相等是**正常**的：`index.json` 只收录已经写完 HTML 笔记的条目，`categories.json` 记录分类页上列出的全部论文（含尚未深读的），站点首页统计口径又与两者不同。引用数字时请注明口径。

## 分类结构（14 类 → 本库分类页）

| 站点目录 | 分类 | 清单条目 | 本库分类页 |
|----------|------|----------|------------|
| `01_Foundational_RL` | Foundational RL（基础强化学习） | 15 | [01-foundational-rl](../../wiki/overview/paper-notebook-category-01-foundational-rl.md) |
| `02_Motion_Retargeting` | Motion Retargeting（运动重定向） | 4 | [02-motion-retargeting](../../wiki/overview/paper-notebook-category-02-motion-retargeting.md) |
| `03_High_Impact_Selection` | High Impact Selection（高影响力精选） | 26 | [03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md) |
| `04_Loco-Manipulation_and_WBC` | Loco-Manipulation and WBC（运动操作与全身控制） | 142 | [04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) |
| `05_Locomotion` | Locomotion（行走运动） | 82 | [05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md) |
| `06_Manipulation` | Manipulation（灵巧操作） | 55 | [06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md) |
| `07_Teleoperation` | Teleoperation（遥操作） | 24 | [07-teleoperation](../../wiki/overview/paper-notebook-category-07-teleoperation.md) |
| `08_Navigation` | Navigation（导航） | 19 | [08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md) |
| `09_State_Estimation` | State Estimation（状态估计） | 14 | [09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md) |
| `10_Sim-to-Real` | Sim-to-Real（仿真到现实） | 10 | [10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md) |
| `11_Simulation_Benchmark` | Simulation Benchmark（仿真与基准） | 22 | [11-simulation-benchmark](../../wiki/overview/paper-notebook-category-11-simulation-benchmark.md) |
| `12_Hardware_Design` | Hardware Design（硬件设计） | 39 | [12-hardware-design](../../wiki/overview/paper-notebook-category-12-hardware-design.md) |
| `13_Physics-Based_Animation` | Physics-Based Animation（物理动画） | 27 | [13-physics-based-animation](../../wiki/overview/paper-notebook-category-13-physics-based-animation.md) |
| `14_Human_Motion` | Human Motion（人体动作分析与生成） | 39 | [14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md) |

## 仓库更名（2026-08-28）

- **新仓库：** https://github.com/ImChong/Robot_Learning_Paper_Notebooks
- **新站点：** https://imchong.github.io/Robot_Learning_Paper_Notebooks/
- **旧名：** `Humanoid_Robot_Learning_Paper_Notebooks` / Humanoid Robot Learning Paper Notebooks / Humanoid Paper Notebooks
- 本归档文件名 `humanoid-robot-learning-paper-notebooks.md` 与 wiki 总索引文件名保持不变，避免站内 related 断链。

## 源码与开放性核查（2026-08-03）

- **仓库公开**：GitHub 仓库可直接访问，许可证 BSD-3-Clause；`papers/` 目录按上述 14 个分类组织笔记源文件，根目录含 `progress.json` 与 `papers/PROGRESS.md` 两份进度清单。
- **站点构建**：Jekyll 静态站 + Python 预处理脚本，笔记以 Markdown 撰写、发布为逐篇 HTML。
- **上游来源**：站点自述取材自 `awesome-humanoid-robot-learning` 资源集，并对带官方实现的论文标注 `⭐️ open-source code link in note`。
- **对本库的含义**：分类页与实体页的深读链接均指向本站 HTML；站点改版或重排分类目录会导致这些外链失效，届时需重跑 `make paper-notebook-links` 与 `make paper-notebook-summaries` 重新对齐。

## 对 wiki 的映射

- [Robot Learning Paper Notebooks 总索引](../../wiki/overview/humanoid-paper-notebooks-index.md) — 14 个分类的统一入口
- `wiki/overview/paper-notebook-category-01..14-*.md` — 逐类论文清单（见上表）
- `wiki/entities/paper-notebook-*.md` — 逐篇论文实体页，深读链接回指本站
- 同作者姊妹演示站：[rl-sim2sim-demo-website](rl-sim2sim-demo-website.md)

## 维护脚本

| 命令 | 作用 |
|------|------|
| `make paper-notebook-bootstrap` | 依 `progress.json` 补齐未映射论文的 sources/ 与实体页 |
| `make paper-notebook-links` | 向已有 wiki 页注入深读笔记链接 |
| `make paper-notebook-summaries` | 同步笔记摘要 |
| `make paper-notebook-dedupe` | 去重分类树中的重复节点 |

## 参考来源（原始）

- [Robot Learning Paper Notebooks（站点首页）](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
- [ImChong/Robot_Learning_Paper_Notebooks（GitHub 仓库）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)
- [progress.json（阅读进度）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
