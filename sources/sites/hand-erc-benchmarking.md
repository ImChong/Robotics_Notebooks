# HAND Benchmarking（hand-erc.github.io/benchmarking）

- **标题：** HAND Benchmarking — Benchmarks for tracking the progress of robot hand dexterity
- **类型：** site / benchmark-spec / survey-companion
- **URL：** <https://hand-erc.github.io/benchmarking/>
- **机构：** 美国国家科学基金会 HAND Engineering Research Center（NSF HAND ERC；西北大学 Northwestern、德州农工大学 Texas A&M、卡内基梅隆大学 CMU、佛罗里达农工大学 FAMU 等）
- **论文：** [arXiv:2609.05585](https://arxiv.org/abs/2609.05585) — *Benchmarking Dexterity of Multifingered Robot Hands: A Review and Perspective*
- **ERC 主页：** <https://hand-erc.org>
- **入库日期：** 2026-09-09
- **一句话说明：** NSF HAND ERC 发布的**多指灵巧手 dexterity 评测框架**配套站：四层 benchmark（application / system / hand / component）、16 项系统级原子任务、hand/component 指标树与 DexNex 旗舰测试台说明；v1 alpha（2026-08）。

## 开源状态（步骤 2.5，截至 2026-09-09）

| 项 | 结论 |
|----|------|
| 项目页 | **已开放**：<https://hand-erc.github.io/benchmarking/>（论文 PDF 链、多页 benchmark 规范、指标表、DexNex 介绍） |
| 独立代码仓 | **未开源**：页头仅链 arXiv PDF；Footer 除 Academic Project Page Template 外**无 GitHub / Hugging Face / 数据集**链接 |
| 规范数据 | **静态 JSONC 随站发布**（`data/*.jsonc`）：任务定义、物品清单、指标树可由前端 JS 加载；**不是**可一键训练/评测的仿真环境或 ROS 包 |
| DexNex 测试台 | **硬件/系统集成在研**：站页描述为 HAND ERC 旗舰双臂 dexterity testbed，随新手型演进；无公开仿真镜像 |
| Application-level | **Coming soon**：`application.jsonc` 标 `comingSoon: true`，正文暂不展开社会经济 adoption 指标 |

**判定：规范已公开 / 无可运行官方代码。** 适合作为**多层级灵巧评测口径与任务清单**引用；勿写成「已开源 benchmark 环境」或「可直接 pip install 的评测包」。

## 站点结构

| 入口 | 内容 |
|------|------|
| `/` | 总览：论文链 + 三大主题卡片 |
| `/benchmarks.html` | **Multi-Level Hand Benchmarks**：application / system / hand / component 四层任务卡 + Component Testbeds + Performance Metrics |
| `/attribution.html` | **System-Level Attribution Problem**：设计变量 $D$ → 系统分数 $S(D)$ 的灵敏度/归因叙事 |
| `/dexnex.html` | **DexNex（Dexterity Nexus）**：集成手–臂–视觉–AI–遥操作的旗舰测试台 |
| `data/system.jsonc` 等 | 各层 benchmark 机器可读定义（物品 YCB 编号、流程、指标单位） |

## 四层 benchmark 框架（编译自站页 + 论文）

| 层级 | 测什么 | 本站状态（2026-09） |
|------|--------|---------------------|
| **Application** | ROI、人机接受度、社会经济 adoption | Coming soon |
| **System** | 短程「原子」灵巧任务：手 + 臂 + 视觉 + 控制 + AI + 遥操作全集成 | **16 项 Atomic Tasks** 已列规格 |
| **Hand** | 手/腕本体：mobility、strength、speed、design | 指标树已列；部分 design 项暂藏 Design-Based tab |
| **Component** | 执行器、传动、传感、皮肤等单元测试 | 执行器/传动等条目 + Northwestern Finger Testbed |

## 系统级 16 项原子任务（DexNex 套件）

Box and Blocks · Peg-in-Hole · Pick Up Flat Object · Tie a Knot · Twist Lid on Jar · Use Screwdriver · Use Scissors · Fasten Button · In-Hand Reorienting · Spin a Top · Use Chopsticks · Blindly Retrieve an Object from Cover · Bundle Socks · Zip a Zipper · Paper Folding · Bundle Dowels with a Rubber Band

主指标类型：块数/peg 数、**总任务完成时间**、spin duration 等（见 `hand-metrics.jsonc`）。

## 为何值得保留

- **补「灵巧 benchmark 分层 + 归因」视角**：相对 [DexVerse](../../wiki/entities/paper-dexverse.md)（仿真 IL 榜）、[DexBench](./dexbench-org.md)（工业 OSC 规格），HAND 强调 **system ↔ hand ↔ component** 灵敏度，服务**手型设计**而不只是策略排行榜。
- **in-hand manipulation 主轴**：任务选取偏向固定抓取之外的指内操作、工具使用与精细接触（拧盖、纽扣、筷子、纸折等）。
- **与 DexNex 硬件线绑定**：规范不是纯文献综述，而是 ERC 旗舰台未来数年的演进评测口径。

## 对 wiki 的映射

- [HAND ERC 灵巧评测综述](../../wiki/entities/paper-hand-erc-benchmarking-dexterity.md) — 论文 + 站页归纳
- [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧操作 benchmark 索引
- [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — ③ 层相邻：真机/系统级 dexterity + 硬件归因

## 参考链接

- 项目页：<https://hand-erc.github.io/benchmarking/>
- 论文：<https://arxiv.org/abs/2609.05585>
- HAND ERC：<https://hand-erc.org>
