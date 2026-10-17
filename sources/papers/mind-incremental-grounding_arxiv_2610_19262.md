# MIND: Instruction-Conditioned Incremental Grounding for Robots（arXiv:2610.19262）

> 来源归档（ingest）

- **标题：** MIND: Instruction-Conditioned Incremental Grounding for Robots
- **arXiv：** <https://arxiv.org/abs/2610.19262>（HTML：<https://arxiv.org/html/2610.19262v1>），cs.RO / cs.AI，2026-10-13 提交，9 页 4 图；ICRA 2027 投稿（匿名）
- **代码：** <https://anonymous.4open.science/r/mind-icra27>（匿名评审仓库，[sources/repos/mind-icra27.md](../repos/mind-icra27.md)）
- **项目页：** <https://mind-icra27.github.io/>（[sources/sites/mind-icra27-project.md](../sites/mind-icra27-project.md)）
- **作者：** Mohamed Abouagour、Byung-Cheol Min（Indiana University Bloomington）；Junhyun Nam、Wonse Jo（Incheon National University）
- **入库日期：** 2026-10-17
- **一句话说明：** 把持久语言指令编译为依赖变量，事件发生时只刷新受影响房间的语义判断，可达性与目标选择分开维护。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面列出生成器、提示词、manifest、模型配置、结果日志在匿名仓库；论文 PDF 评审期间不公开；未署名机构。
- **仓库核查（2026-10-17）：** 抓取匿名仓库页只返回标题 "Anonymous Github"，无法读到文件树与 README（**未核实**）。
- **结论：** **匿名评审仓库，内容未核实**；非正式开源发布，可能在评审后撤换。

## 摘录 1：方法

- 每条指令由语言模型编译一次，得到两个依赖集合，覆盖五类状态（库存、区域标签、位置、历史、拓扑）：一个用于房间语义资格，一个用于目标选择；库存只计入资格集合。
- 每个观测事件报告改变的变量与受影响房间估计；控制器将语义依赖与事件足迹求交，仅刷新重叠的房间判断。
- 可达性单独维护，拓扑或位置变化时刷新；确定性选择器在可行集变化或报告变量落入任一依赖集合时重跑。
- 保证：在固定评估器与保守的依赖和足迹覆盖下，决策与全量重算一致；不保证编译器预测或评估器精度。

## 摘录 2：结果

- ProcTHOR 240 条五事件序列（1,200 个决策）：平均语义评估 30.6 降到 11.2，决策全部保持；冻结评估器 Phi-4 与 Qwen3-14B 同样 1,200/1,200，全量重算的 episode 准确率为 0.342 与 0.275（即方法不提升任务准确率）。
- 对照：仅库存静态规则 episode 准确率 0.479、失效召回 0.703；无编译器逐事件相关性基线 0.658（13.1 次评估）。项目页消融：静态规则与全量重算的决策一致率为 0.958 与 0.983。
- 依赖编译器（69 条留出请求）：Phi-4（K=2，部署用）精确签名恢复 0.667、包含率 1.000；gpt-oss-120b 精确恢复 0.935。
- AI2-THOR 32 个闭环 episode：最终目标均与全量重算一致，评估 10.47 对 26.00（2.48x）。观测推导的移除足迹在 54 个留出房屋中全部包含受影响房间（单侧 95% 下界 0.946）。
- Phi-4 计时：评估器时间约降 2.49x；含 0.56 s 编译成本后每序列净省 0.30 s。
- Unitree G1：26 次计分运行，4 个场景，另有 8 次排除；场景 1 评估 3 对 9，场景 2 为 3 对 9，场景 4 为 4 对 15 至 18（两种策略均弃权）。
- 局限：假设已有语义地图与受限状态词表；依赖标签未独立标注；冻结判决带隔离了维护保真度与任务正确性；足迹低估会损害正确性；多候选区域谓词、跨候选基数约束、未来状态预测不在范围；硬件仅一个布局 4 个场景。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-mind-incremental-grounding.md`。
