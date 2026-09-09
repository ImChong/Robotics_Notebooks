# Benchmarking Dexterity of Multifingered Robot Hands: A Review and Perspective

> 来源归档（ingest）

- **标题：** Benchmarking Dexterity of Multifingered Robot Hands: A Review and Perspective
- **类型：** paper / survey / review / dexterous-manipulation / benchmark / multifingered-hand / in-hand-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.05585>
- **提交日期：** 2026-09-04
- **拟发表：** *Annual Review of Control, Robotics, and Autonomous Systems*, Volume 10, 2027
- **项目页：** <https://hand-erc.github.io/benchmarking/>
- **机构：** NSF HAND Engineering Research Center — 西北大学（Northwestern，通讯）、德州农工大学（Texas A&M）、卡内基梅隆大学（CMU）、佛罗里达农工大学（FAMU）等
- **作者：** Anthony Shilati\*、Anunth Ramaswami、Luke Batteas、Sylvia Tan、Anthony Barcio、Sairam Umakanth、Preksha Rao、David McDougall、Landry Graves、Ahmet A. Ozkan、Yunsoo Yoon、Arushi Pradhan、Michael G. Henry、Rohan Kota、Damian Gonzalez、Gray C. Thomas、Gary K. Fedder、J. Edward Colgate、Kevin M. Lynch（\*多单位联合；通讯：anthony.shilati@northwestern.edu, kmlynch@northwestern.edu）
- **入库日期：** 2026-09-09
- **一句话说明：** HAND ERC 视角下的**多指灵巧手 dexterity 评测综述**：提出 application / system / hand / component 四层框架以缓解**系统级归因问题**，回顾文献代表性 benchmark，并为旗舰 **DexNex** 测试台发布 **16 项系统级原子任务**与 hand/component 指标树；侧重 **in-hand manipulation** 与精细接触力控。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://hand-erc.github.io/benchmarking/> | 任务规格 JSONC、指标表、DexNex 页 |
| HAND ERC | <https://hand-erc.org> | NSF EEC 2330040 资助的工程研究中心 |
| 仿真灵巧 IL 榜 | [DexVerse](https://arxiv.org/abs/2607.08751) | 100 任务多具身 Isaac Lab bench；策略均值 SR 34% |
| 工业灵巧规格 | [DexBench](https://dexbench.org/en/) | OSC + 18 原子任务；评测仓待发布 |
| 双手 RL + CWS | [CHORD](../../wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md) | 4,739 项接触丰富双手 benchmark |
| 组件测试台 | Finger Testbed（Northwestern） | 单指 strength/speed/impedance/bandwidth 硬件评测 |

## 摘要级要点

- **工作定义：** dexterity = 通过接触高效、可靠地改变物理世界或学习其性质的能力；强调手–物–环境动态协商，而非单纯轨迹跟踪。
- **为何聚焦多指手：** 平行夹爪/钩爪可完成非抓取或固定抓取操作，但**指内操作（in-hand manipulation）**、工具化与类人基础设施复用需要多指 + 精细接触力控。
- **四层 benchmark：** Application（社会经济 adoption，本文不展开）→ **System**（短程原子任务，需臂+视觉+AI 全集成）→ **Hand**（手/腕本体性能）→ **Component**（执行器/传感/皮肤等单元测试）。越高层越难把性能变化归因到单一设计变量。
- **归因问题：** 记设计变量 $D$、系统分数 $S(D)$；理想反馈需 $(\partial S/\partial D)|_{D_0}$，但 jar-capping 等系统任务上难以判断「触觉空间分辨率」等组件改动的影响——故需 hand/component 层 benchmark。
- **文献回顾：** 覆盖 YCB Box and Blocks、NIST Assembly Taskboard、POMDAR、KaRMA、GRASP taxonomy、Yale OpenHand 系列等；Table 1 对照既有 system/hand/component 工作。
- **DexNex 新套件（<20 任务）：** 偏向 (1) 测不同 dexterity 特征的原子任务；(2) 难度与任务特性跨度大；(3) 文献有先例；(4) **受益于 in-hand manipulation**（超越固定抓取）。完整 16 项见项目页 `system.jsonc`。
- **Hand 层指标族：** Mobility（Kapandji、GRASP taxonomy、grasping volume）、Strength（wrap/pinch/pullout）、Speed（grasp cycle）、Design（DoF、workspace、KaRMA 等）。
- **Component 层：** Output inertia、effort control bandwidth、density、backdrive effort、responsiveness 等；强调反射惯量 $N^2$ 缩放对**机械透明度（mechanical transparency）**的损害。
- **开源结论：** 项目页**无官方训练/仿真代码仓**；规范以静态站 + JSONC 发布；DexNex 为 evolving hardware testbed。

## 核心摘录（面向 wiki 编译）

### 1) 操作类型三分（Figure 1）

| 类型 | 含义 | 与 dexterity 关系 |
|------|------|-------------------|
| Nonprehensile | 推、滑等无抓取 | 简单夹爪可胜任 |
| Fixed grasp | 抓取后由臂完成运动 | 指主要固定物体 |
| In-hand manipulation | 抓取后物相对掌运动 | **多指 + 接触相对运动**；本文主轴 |

### 2) 系统级 16 项原子任务（项目页 v1 alpha）

| 任务 | 主指标（归纳） | dexterity 侧重 |
|------|----------------|----------------|
| Box and Blocks | 60s 内转移块数 | 杂乱抓取、运输、时间压力（YCB Protocol 3a） |
| Peg-in-Hole | Pegs inserted | 精密对齐（NIST M1） |
| Pick Up Flat Object | 完成时间 | 薄物/平面抓取 |
| Tie a Knot | 完成时间 | 柔性体、双手协调 |
| Twist Lid on Jar | 完成时间 | 螺纹旋紧、持续力控 |
| Use Screwdriver | 完成时间 | 工具使用 |
| Use Scissors | 完成时间 | 刃口对准、双手 |
| Fasten Button | 完成时间 | 小尺度装配 |
| In-Hand Reorienting | 完成时间 | **指内重定向** |
| Spin a Top | Spin duration | 动态接触 |
| Use Chopsticks | 块数/时间 | 工具 + 非抓取辅助 |
| Blindly Retrieve from Cover | 完成时间 | 遮挡、探索 |
| Bundle Socks | 完成时间 | 可变形体 |
| Zip a Zipper | 完成时间 | 精细滑动接触 |
| Paper Folding | 完成时间 | 可变形、折痕 |
| Bundle Dowels w/ Rubber Band | 完成时间 | 弹性约束捆绑 |

### 3) 与站内代表性 benchmark 对比（概念层）

| 平台 | 主轴 | 分层/归因 | 与 HAND 关系 |
|------|------|-----------|--------------|
| DexVerse | 仿真 IL/VLA 多任务 SR | 系统级策略榜 | HAND 补 **硬件设计归因 + 真机原子任务** |
| DexBench | 工业 OSC + 18 任务规格 | 任务状态复杂度 | HAND 更偏 **ERC 研究手 + DexNex**；勿混名 |
| DexHoldem | 真机扑克 SPSR | 桌面博弈协议 | HAND 是 **通用手型 dexterity 综述** |
| CHORD | RL + 接触力旋量 | 双手仿真 RL | HAND 不替代 RL 训练轴 |

## 对 wiki 的映射

- [HAND ERC 灵巧评测综述](../../wiki/entities/paper-hand-erc-benchmarking-dexterity.md) — 四层框架、任务清单、归因与 DexNex
- [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧 benchmark 索引
- [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — ③ 层 + 硬件归因横切
- [DexVerse](../../wiki/entities/paper-dexverse.md) — 仿真多任务 IL 对照
- [DexBench](../../wiki/entities/dexbench.md) — 工业 OSC 规格对照

## 引用（arXiv BibTeX）

```bibtex
@article{shilati2026benchmarking,
  title   = {Benchmarking Dexterity of Multifingered Robot Hands: A Review and Perspective},
  author  = {Shilati, Anthony and Ramaswami, Anunth and Batteas, Luke and Tan, Sylvia and Barcio, Anthony and Umakanth, Sairam and Rao, Preksha and McDougall, David and Graves, Landry and Ozkan, Ahmet A. and Yoon, Yunsoo and Pradhan, Arushi and Henry, Michael G. and Kota, Rohan and Gonzalez, Damian and Thomas, Gray C. and Fedder, Gary K. and Colgate, J. Edward and Lynch, Kevin M.},
  journal = {arXiv preprint arXiv:2609.05585},
  year    = {2026}
}
```
