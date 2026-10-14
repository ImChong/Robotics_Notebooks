# What Matters in Quality-Aware VLA Pretraining: A Benchmark and Empirical Study（arXiv:2610.18869）

> 来源归档（ingest）

- **短名：** AXIS-Bench
- **类型：** paper / benchmark
- **作者：** Mingxuan Yan、Litian Gong、Dihong Huang、Zhixu Li、Tianyu Zhang、Wenqian Zhang、Hefeifei Jiang、Peihao Li、Jian Zhang、Mengfei Zhao、Yikai Tang、Hai Zhai、Zehao Wang、Ruijian Liang、Yanjia Huang、Lin Shao、Changliu Liu、Kaiyu Hang、Zhiwen Fan、Masayoshi Tomizuka、Jianfei Yang、Jiachen Li
- **机构：** Georgia Institute of Technology；Axis Robotics；University of California, Riverside；Carnegie Mellon University；University of California, Berkeley；Texas A&M University；National University of Singapore；Rice University；Nanyang Technological University
- **arXiv：** <https://arxiv.org/abs/2610.18869>（v1，2026-10-13）
- **项目页：** <https://axis-bench.github.io/>
- **代码 / 数据：** 项目页截至 2026-10-14 标记为 Coming Soon
- **入库日期：** 2026-10-14
- **一句话说明：** 在固定数据量与训练预算下比较示范质量的加权、过滤和条件化方式，并发布 AXIS-Bench 支持可复核的质量标签研究。

## 核心摘录（面向 wiki 编译）

1. **受控比较：** 使用 5,000 条自然次优的人类示范与 24 个目标任务，跨三个 VLA 比较 weighting、filtering、conditioning，并匹配每个 VLA 内的数据量与训练预算。
2. **标签粒度：** conditioning 平均更受益于 episode 级质量标签，filtering 平均更受益于 chunk 级标签；不同模型会出现反转，不能把平均规律当作固定配方。
3. **质量标签作用：** approach-aware episode conditioning 平均优于纯行为克隆；条件预训练后用固定质量标签微调可提升均值，推理时带不带标签均有收益。
4. **基准设计：** AXIS-Bench 基于众包仿真示范，保存物体与接触状态，允许用同一轨迹重新计算质量标签；轨迹可回放并支持工作区对齐渲染。
5. **真机迁移与开放状态：** 真实实验中仿真预训练把成功率从 9.6% 提高到 31.7%；项目页代码和数据仍标为即将发布。

**对 wiki 的映射：** [paper-axis-bench-quality-aware-vla-pretraining](../../wiki/entities/paper-axis-bench-quality-aware-vla-pretraining.md)

## 版本与引用

- arXiv v1，提交于 2026-10-13。
- 来源以 arXiv 摘要和 AXIS-Bench 项目页为准。
---
