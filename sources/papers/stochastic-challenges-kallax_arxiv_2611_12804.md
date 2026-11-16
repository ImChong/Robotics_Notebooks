# Stochastic Challenges / KALLAX 基准（arXiv:2611.12804）

> 来源归档（paper · 原文通读）

- **标题：** Benchmarking Real-World Robotic Manipulation through Stochastic Challenges
- **类型：** paper（CoRL 2026，Austin TX）
- **arXiv：** <https://arxiv.org/abs/2611.12804>（v1，2026-11-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.12804v1>
- **项目页：** <https://rpl-bonn.github.io/stochastic-challenges/>（基准网站 <https://klxbench.rpl.uni-bonn.de/>）
- **作者与机构：** Elias Gläßer、Gülhan Şikaroğlu、Pavlo Kupyn、Marc Pollefeys、Zuria Bauer、Hermann Blum；波恩大学、Lamarr 研究所、苏黎世联邦理工、微软
- **入库日期：** 2026-11-11
- **一句话说明：** 不固定测试集，而固定「场景与任务的抽样分布」：用随机种子生成 IKEA KALLAX 货架场景，各实验室按种子复建并执行，用方差分解给出比较两台机器人需要多少种子。
- **核对范围：** 通读 arXiv v1 PDF 正文 §1–§7、表 1–2、公式 (1)–(3)、参考文献列表与附录 A–D（估计量、方差、比较、扩展）；图 1–3 只读图注，未逐图细看。项目页（<https://rpl-bonn.github.io/stochastic-challenges/>）静态读过全文文字；KLX 基准网站与 Stretch-Compose 等引用仓库未访问，评测栈细节未核实。论文未给代码链接。
- **通读：** 已完成

## 开源状态

- 论文未提供代码仓库。项目页写「Benchmark website」（klxbench.rpl.uni-bonn.de，随机种子生成与结果上传服务）；项目页抓取时 arXiv 链接仍标「coming soon」。是否开放可下载的生成器源码：**未核实**。
- 所引评测栈 Stretch-Compose（github.com/rpl-bonn/stretch-compose，论文 [25]）未核查。

## 核心摘录

1. **范式（§1、§3）：** 标准化「场景与任务的分布」而非固定测试集；任何实验室可由种子复建具体场景，参与者只见到全部可能场景的一小部分，类比数据集基准的隐藏测试集。
2. **估计量与方差（§3.1–3.2）：** 三层平均（重复尝试、场景内任务、场景）得 μ̂；Var(μ̂)=(σ²_seed + B/J)/K，J→∞ 时方差不趋零，下限 σ²_seed/K。
3. **样本量（§3.3–3.4，式 1–3）：** 独立种子时每台机器人所需种子数 K ≥ 2(z_{1-α/2}+z_{1-β})²(σ²_seed+B/J)/δ²；共享种子比例 p 时 σ²_seed 项乘 (1−p)。
4. **KALLAX 基准（§4）：** 3×4 货架，随机选定空格数（空比例 0.5 即 6 个空位），其余格从至多 82 种兼容插件（门、抽屉）抽取，门约为抽屉的 2.5 倍，每个抽屉插件含两个抽屉；物体搜索任务另从 79 个 YCB 物体抽样。
5. **实验（§5）：** Hello Robot Stretch 3 与带臂 Spot，两国两实验室，共 568 次真机试验；仅评开门、开抽屉（打开至少 10 cm 才算成功）。门：Spot 62 次 / 6 种子，Stretch 130 次 / 7 种子；抽屉：Spot 88 次 / 6 种子，Stretch 288 次 / 8 种子。
6. **结果（表 1–2）：** 全部种子平均成功率 门 Spot 0.7109 vs Stretch 0.1802（差 0.5307，可检测余量 0.333）；抽屉 0.8792 vs 0.6352（差 0.2439，余量 0.232）。α=0.10、检验功效 80%。p=0.5 的混合设计余量降为 0.256 与 0.183。
7. **局限（§6）：** 需大量试验才能分辨小差距；共享种子场景由同一研究者在两个实验室复建，未量化不同团队复建带来的差异；依赖诚实且校准良好的成功上报。

## 对 wiki 的映射

- [paper-stochastic-challenges-kallax](../../wiki/entities/paper-stochastic-challenges-kallax.md)
