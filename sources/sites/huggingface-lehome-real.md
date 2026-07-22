# IliaLarchenko/lehome_real（Hugging Face）

> 来源归档（ingest）

- **标题：** LeHome Challenge 2026 — Real-World Policy（`lehome_real`）
- **类型：** model / huggingface
- **链接：** <https://huggingface.co/IliaLarchenko/lehome_real>
- **论文：** <https://arxiv.org/abs/2606.27163>
- **博客：** <https://ilialarchenko.com/projects/lehome2026>
- **代码：** <https://github.com/IliaLarchenko/lehome_solution>
- **姊妹权重：** <https://huggingface.co/IliaLarchenko/lehome_sim>
- **入库日期：** 2026-07-22
- **一句话说明：** LeHome 2026 **线下实体赛道第 2 名**策略：在仿真策略之上做真机 BC 微调（官方 teleop + 自家 teleop/DAgger + 强增强仿真回放混合）；物理 SO-ARM101 双臂部署。

## 模型卡要点

| 字段 | 内容 |
|------|------|
| **赛道成绩** | Real-world final：**2nd**；官方综合分 **865 / 1080**（博客榜单） |
| **任务** | 真机双臂叠衣；评委打分，未见衣物额外加分 |
| **训练域** | 真机 BC（非仿真 RL 飞轮）；从较晚但非最新的 sim ckpt 迁移并剥离特权头 |
| **用法** | 见仓库 `scripts/serve.py` + `record_real_dagger.py`：`hf download IliaLarchenko/lehome_real --local-dir outputs/checkpoints/lehome_real` |

## 对 wiki 的映射

- [Learning to Fold](../../wiki/entities/paper-lehome-learning-to-fold.md)
- [lehome_solution 仓库](../repos/lehome_solution.md)
- [仿真权重 lehome_sim](huggingface-lehome-sim.md)
- [Sim2Real](../../wiki/concepts/sim2real.md) · [DAgger](../../wiki/methods/dagger.md)
