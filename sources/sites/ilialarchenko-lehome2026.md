# Learning to Fold — LeHome Challenge 2026（项目博客）

> 来源归档（ingest）

- **标题：** Learning to Fold — LeHome Challenge 2026
- **类型：** site / blog（竞赛方案演示页，含成功/失败推理视频）
- **URL：** <https://ilialarchenko.com/projects/lehome2026>
- **论文：** <https://arxiv.org/abs/2606.27163>
- **代码：** <https://github.com/IliaLarchenko/lehome_solution>
- **权重：** <https://huggingface.co/IliaLarchenko/lehome_sim> · <https://huggingface.co/IliaLarchenko/lehome_real>
- **作者 / 机构：** Ilia Larchenko（独立）
- **入库日期：** 2026-07-22
- **一句话说明：** LeHome 2026 夺冠方案的工程叙事页：架构、异步 RL 飞轮、奖励/advantage、Thompson 推理调参、仿真榜单与失败案例视频、sim-to-real 配方与官方决赛 rollout；页尾给出 tech report / 代码 / 双 HF 权重。

## 开源状态（项目页核查 2026-07-22）

- **已开源：** § Paper, code & citation 明确列出 Code → GitHub、`lehome_sim` / `lehome_real` HF、arXiv:2606.27163。
- **演示资产：** 含 reward overlay 的 rollout 视频、仿真成功/失败案例、相机 overlay 对齐、自家机 DAgger 与自主成功、决赛现场官方 rollout（画质较粗）。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| The challenge | 四类衣物、SO-ARM101 双臂、仿真/真机评分差异、约束组合 |
| Model architecture | π₀.₅ 改版、辅助预测头、garment token、RECAP/AdaRMS、关键点未来头 |
| RL loop | AWR+RECAP；Trainer / rollout / 人工纠偏经 HF Hub 异步协调 |
| Reward & advantage | 关键点进度 + 成功/完成预测 + 跨衣相对成功 → GAE；debug overlay S/A/R/C/T |
| Inference-time | 执行长度、速度、overlap、CFG、温度、Best-of-N；Thompson bandit |
| Online-round results | **1/62，79.63%**；Top-5 表；成功与失败视频 |
| Sim to real | 剥特权头、三桶数据、相机对齐、重增强、DAgger；**2nd，865/1080** |
| Takeaways | 建议把仿真 RL 与真机 BC+DAgger 合成统一配方 |
| Citation | BibTeX `larchenko2026lehome` |

## 对 wiki 的映射

- 主实体：[Learning to Fold（论文实体）](../../wiki/entities/paper-lehome-learning-to-fold.md)
- 论文摘录：[lehome_learning_to_fold_arxiv_2606_27163.md](../papers/lehome_learning_to_fold_arxiv_2606_27163.md)
- 仓库：[lehome_solution.md](../repos/lehome_solution.md)
- 权重：[huggingface-lehome-sim.md](huggingface-lehome-sim.md) · [huggingface-lehome-real.md](huggingface-lehome-real.md)
