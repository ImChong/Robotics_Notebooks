# reform_iclr_2026

> 来源归档（ingest）

- **标题：** ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation
- **类型：** paper
- **会议：** ICLR 2026
- **OpenReview：** <https://openreview.net/forum?id=YvFsyRReeN>
- **PDF（OpenReview）：** <https://openreview.net/pdf?id=YvFsyRReeN>
- **项目页：** <https://mit-realm.github.io/reform/>
- **代码：** <https://github.com/MIT-REALM/reform>
- **入库日期：** 2026-09-30
- **一句话说明：** 用 **有界源分布上的 BC flow 策略** 刻画行为策略 **support**，再用 **reflected flow** 在 support 内操纵噪声以最大化 Q——**无需** 对行为策略做统计距离正则，在 OGBench **40 任务**（clean/noisy）上以 **固定超参** 在 performance profile 上压过 hand-tuned 的 flow 类 offline RL 基线。

## 核心论文摘录（MVP）

### 1) 问题：Offline RL 的 OOD 与表达力两难（Abstract / 项目页 Overview）

- **链接：** <https://mit-realm.github.io/reform/>、<https://openreview.net/pdf?id=YvFsyRReeN>
- **核心贡献：** 固定数据集 offline RL 中，策略若离开训练分布会产生 **OOD Q 误差**；传统做法用 **统计距离正则** 把策略拉向行为策略，但 **限制策略改进** 且 **难完全杜绝 OOD**，并常需 **逐任务调正则权重**。扩散 / flow 策略可表达 **多模态动作**，但如何在保持表达力的同时 **结构性满足 support 约束** 仍不清楚。
- **对 wiki 的映射：**
  - [ReFORM（ICLR 2026）](../../wiki/entities/paper-reform-iclr-2026.md)
  - [Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md)

### 2) 方法：BC flow + reflected noise manipulation（项目页 Method）

- **链接：** <https://mit-realm.github.io/reform/>
- **核心贡献：** **ReFORM** 先学 **BC flow 策略**（灰箭头）：将有界源分布 \(q_\mathrm{BC}=\mathcal U(\mathcal B_l^d)\)（半径 \(l\) 的超球均匀）映射到匹配数据集 \(\mathcal D\) 的 \(p_\mathrm{BC}\)，从而 **有界源 → 行为 support**。并行学习 **reflected flow 噪声生成器**（蓝箭头），输出操纵后的源 \(\tilde q_\mathrm{BC}\)，使 \(\tilde p_\mathrm{BC}\) 在 **BC support（红区）内** 最大化 Q，从而 **按构造满足较弱的 support 约束**，避免 OOD 又保留 **多模态**。
- **对 wiki 的映射：**
  - [ReFORM](../../wiki/entities/paper-reform-iclr-2026.md)「核心原理」
  - [Probability Flow](../../wiki/formalizations/probability-flow.md)

### 3) 实验：OGBench 40 任务与固定超参（项目页 Tasks / Results）

- **链接：** <https://mit-realm.github.io/reform/>
- **核心贡献：** 在 **antmaze-large、cube-single、cube-double、scene** 四类环境的 **singletask** 任务上共 **40 任务**；**clean**（专家随机轨迹）与 **noisy**（高噪声次优策略轨迹）两种数据。指标为跨算法 min–max **归一化回报** 的 **performance profile** 与 **IQM** 柱状图。论文主张：**ReFORM 全任务共用一套超参**（步数等环境差异除外），而 **FQL / IFQL / DSRL** 等 flow 结构基线使用 **逐任务 hand-tuned** 超参；clean 集上 profile 整体占优，noisy 集上除 \(\tau\approx0.9\) 附近与 FQL(S) 接近外仍主导。
- **对 wiki 的映射：**
  - [ReFORM](../../wiki/entities/paper-reform-iclr-2026.md)「实验与评测」

### 4) 官方 Jax 实现与复现入口（MIT-REALM/reform README）

- **链接：** <https://github.com/MIT-REALM/reform>
- **核心贡献：** **已开源**（MIT License）：`pip install -e .`；`python scripts/train.py reform --env-name <ogbench-singletask-env>` / `scripts/test.py --path logs/...`；同仓提供 **fql / ifql / dsrl** 对照；依赖 **OGBench** 的 `singletask` 环境名（非 goal-conditioned）。
- **对 wiki 的映射：**
  - [reform 仓库索引](../repos/reform_mit_realm.md)
  - [ReFORM 实体页](../../wiki/entities/paper-reform-iclr-2026.md)「源码运行时序图」

## BibTeX（项目页提供）

```bibtex
@inproceedings{zhang2026reform,
      title={Re{FORM}: Reflected Flows for On-support Offline {RL} via Noise Manipulation},
      author={Zhang, Songyuan and So, Oswin and Ahmad, H M Sabbir and Yu, Eric Yang and Cleaveland, Matthew and Black, Mitchell and Fan, Chuchu},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
}
```

## 当前提炼状态

- [x] 项目页 Method / Results 与摘要对齐
- [x] 源码开放核查（GitHub 官方实现）
- [x] wiki 实体页映射确认
