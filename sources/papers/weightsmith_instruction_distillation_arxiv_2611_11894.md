# Distilling Instructions into Weights（arXiv:2611.11894）

> 来源归档（paper）

- **标题：** Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy's Misreading
- **类型：** paper
- **作者 / 机构：** Ali-Adeeb Abbas（Northeastern University RIVeR Lab；General Motors）、Esmaeil Seraj（General Motors）
- **arXiv：** <https://arxiv.org/abs/2611.11894>（读的是 v1，2026-11-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.11894v1>
- **代码：** https://github.com/adeeb10abbas/weightsmith（MIT；README 已静态读过）
- **项目页：** <https://ali-adeeb.com/weightsmith/>（已静态读首屏与目录，见下）
- **入库日期：** 2026-11-11
- **一句话说明：** 对 zero-shot π0.5 在 RoboLab 仿真中「把立方体放进碗里再把香蕉放进碗里」的过早松手，用「改写指令里的 MLP 激活差」做多键闭式最小范数权重编辑（无梯度、无训练数据、指令不变）；保留种子 27/128 → 106/128。
- **核对范围：** 下载 v1 PDF（23 页），逐段读正文第 1–6 节（含图 1–5 的图注与表 1、表 2）与参考文献页；附录 A（失败层级表 4）、B（逐项配对比较表 5 / 图 6、噪声底表 6）、C（捕获与张量、消融表 7、随机方向对照、诊断细节）、D（感知 / 能力 / 接触 / 夹爪层级）、E（迁移表 9、butter 重跑、保留键）、F（盲编辑器设计与逐任务发现）、G（落点构造与表 10）、H（可复现性）读了文字与表格；图为矢量图，读了图注与图内文字，未逐像素核对；未运行论文代码，也未下载或核对仓库里的 study/ 原始数据与 paper/main.pdf；项目页只读了首屏、目录与跨架构一节的开头，GR00T N1.7 / Cosmos3 Edge 的结果未逐项核对（这些不在 v1 论文正文中）。
- **通读：** 已完成

## 开源状态

- **已开源**：仓库 README 称含 `owsp` 工具、`weightsmith distill`（标「Experimental」）、`weightsmith mcp`、`weightsmith edit`、基准与 study 数据；需 24 GB NVIDIA GPU、约 60 GB 磁盘（README Quickstart）。仓库内容本次未运行。

## 核心摘录（均出自 v1 论文）

1. **设定（第 1、3 节）：** OpenPI 的 π0.5 DROID joint-position 检查点（约 3.35B 参数），在 Isaac Sim 的 16 个 RoboLab 任务上 zero-shot 评估；成功 = RoboLab 终止成功标志；每次比较都在相同模拟器种子上与未编辑策略配对（Wilson 区间、精确 McNemar、20,000 次 bootstrap）。
2. **诊断（第 2 节）：** 原指令下未编辑策略 9/64 成功，63/64 先抓立方体、抬几厘米后在碗前松手（「过早松手」）。回放 21 个录下的过早松手决策、只改指令：改成 "Put the cube in the bowl then put the banana in the bowl"（显式）后仅 2/21 仍松手（图 2a）；作者读作「in the bowl」绑到了错误物体（绑定错误）。显式措辞在不改权重时 22/32 对 3/32。
3. **方法（第 2 节、式 1）：** 对 VLM 第 0–16 层，键 K 为原指令下 MLP 下投影输入（6 个录下决策取均值；72 个键 = 40 个提示 / 状态位置 + 32 个图像格），值 V 为显式指令减原指令的层残差增量；ΔW = Kᵀ(KKᵀ + λI)⁻¹V，λ = 1e-3·tr(KKᵀ)/72，每层秩 ≤ 72；完整编辑改动 450,468,837 个存储值（13.4%）（附录 C）。
4. **主结果（4.1 节）：** 128 个保留种子 27/128 → 106/128，修好 81 条、弄坏 2 条，配对差 +61.7 点（95% 区间 52.3–70.3），McNemar p = 7.2×10⁻²²；同层、同秩、同奇异值的随机方向更新 9/64 对未编辑 13/64（p = 0.45）。显式措辞给未编辑策略为 57/64。
5. **小编辑与迁移（4.2、4.3 节）：** 仅图像格、第 8–16 层的低秩因子 5.3M 参数（0.16%）：102/128 对 19/128；但与完整编辑正面对比 41/64 对 55/64（p = 0.007）。同一编辑原样用到其余 15 个任务：eyeglasses 22/64 → 44/64、cube left of bowl 8/64 → 36/64，cube right of bowl 仍 0/32，其余 13 个任务差异均不显著（p ≥ 0.12）。
6. **编辑位置（4.4 节、表 1）：** VLM 55/64 对未编辑 5/64；动作专家 20/64；动作头最小范数 0/64；动作头门控（盲编辑器）30/64；VLM + 动作专家 43/64；只有 VLM 编辑可迁移。
7. **工具（4.5 节）：** `weightsmith distill` 在全新的 cube-then-banana 项目上，用规则选决策，8/64 → 36/64（33 修好、5 弄坏，p = 4.3×10⁻⁶），仅这一个任务。
8. **盲编辑器（4.6 节、表 2）：** 7 个 Claude Opus 盲编辑器各一任务，仅 cube-then-banana 提交的 1,024 值秩一动作头门控编辑通过保留评分（11/64 → 27/64，p = 0.004，Holm 校正后 0.026）；4 个提交未编辑策略。
9. **其他层级无效（4.7 节）：** 感知（stack yellow on red）、缺失能力（flat box 0/64）、接触物理（mustard）、夹爪阈值（单偏置 117/240 对 117/240）的编辑均无提升；stacking 的门控编辑把「空夹爪闭合」replan 从 73 降到 5，成功率仍 39/64 对 35/64（p = 0.585）。
10. **噪声底（4.8 节、附录 B）：** 同一未编辑策略 butter 同种子两次 14/16 与 8/16；约 30% 配对结果翻转；80% 功效下 32 / 64 / 128 / 256 对可检出约 27 / 19 / 14 / 10 点。
11. **作者自述局限（第 5 节）：** 编辑低秩但不稀疏；需要已知可行的措辞作输入；工具只在一个任务上跑过；位置对比只覆盖一个任务一个措辞；迁移事前不可预测；全部是单一 zero-shot 检查点的仿真结果，每个比较 16 至 128 对；失败层级划分无独立评分者；未来工作需测试任务微调检查点、更小编辑、其他任务与真机。

## 与项目页 / 仓库的差异（本次核对）

- 项目页标题为 "Fixing weak language grounding in a robot policy's weights"，称在「三种策略架构」上测试（π0.5、世界–动作模型 Cosmos3 Edge、GR00T N1.7）；v1 论文正文只有 π0.5 一个检查点。项目页的这些扩展结果不在已通读的 v1 论文里，本次未核对。
- 仓库 README 的结果表与论文一致（106/128、102/128、36/64、27/64 等）。

## 对 wiki 的映射

- [paper-weightsmith-instruction-distillation](../../wiki/entities/paper-weightsmith-instruction-distillation.md)
