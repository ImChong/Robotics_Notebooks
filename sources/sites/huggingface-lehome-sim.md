# IliaLarchenko/lehome_sim（Hugging Face）

> 来源归档（ingest）

- **标题：** LeHome Challenge 2026 — Simulation Policy（`lehome_sim`）
- **类型：** model / huggingface
- **链接：** <https://huggingface.co/IliaLarchenko/lehome_sim>
- **论文：** <https://arxiv.org/abs/2606.27163>
- **博客：** <https://ilialarchenko.com/projects/lehome2026>
- **代码：** <https://github.com/IliaLarchenko/lehome_solution>
- **姊妹权重：** <https://huggingface.co/IliaLarchenko/lehome_real>
- **入库日期：** 2026-07-22
- **一句话说明：** LeHome 2026 **线上仿真赛道第 1 名**（62 队）提交策略：基于 π₀.₅ 的 VLA，经 Isaac Sim 内异步 RL + DAgger 训练；与官方 Collection「LeHome Challenge 2026 solution」同组。

## 模型卡要点

| 字段 | 内容 |
|------|------|
| **赛道成绩** | Online / simulation：**1st / 62**；总体成功率约 **79.63%**（博客榜单） |
| **任务** | 双臂 SO-ARM101 叠四类衣物（品类评测时隐藏） |
| **训练域** | NVIDIA Isaac Sim；异步 RL + DAgger |
| **用法** | 见仓库 `scripts/run_eval.py`：`hf download IliaLarchenko/lehome_sim --local-dir outputs/checkpoints/lehome_sim` |

## 对 wiki 的映射

- [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
- [lehome_solution 仓库](../repos/lehome_solution.md)
- [真机权重 lehome_real](huggingface-lehome-real.md)
