# Learning Sim-to-Real Humanoid Locomotion in 15 Minutes — 项目页

> 来源归档（ingest · Amazon FAR 项目站）

- **标题：** Learning Sim-to-Real Humanoid Locomotion in 15 Minutes
- **类型：** site
- **URL：** <https://younggyo.me/fastsac-humanoid/>
- **论文：** <https://arxiv.org/abs/2512.01996>
- **代码：** <https://github.com/amazon-far/holosoma>（项目页写明 open-source implementation at **Holosoma repository**）
- **机构：** Amazon FAR（Frontier AI & Robotics）
- **作者（页面）：** Younggyo Seo*, Carmelo Sferrazza*, Juyue Chen, Guanya Shi, Rocky Duan, Pieter Abbeel（* equal contribution）
- **入库日期：** 2026-09-21
- **一句话说明：** 面向 **Unitree G1 / Booster T1** 的 **FastSAC / FastTD3** 人形 sim-to-real 配方：单张 **RTX 4090**、数千并行环境、极简奖励 + 强域随机化，**约 15 分钟**训出全关节行走；同一栈可加速 **whole-body tracking**；官方实现落在 **Holosoma** 开源仓。

---

## 开源核查（步骤 2.5，2026-09-21）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — [amazon-far/holosoma](https://github.com/amazon-far/holosoma)（Apache-2.0；项目页直接指向该仓） |
| **权重 / 数据** | 训练 checkpoint 经 Wandb / 官方脚本加载；无单独 HF 权重页（以 Holosoma README 为准） |
| **项目页 Code 区** | 摘要段明确 *An open-source implementation … is available at Holosoma repository* |

## 页面能力要点（策展）

1. **算法栈：** **FastSAC / FastTD3** — 面向大规模并行仿真重新调参的 off-policy 变体（相对经典 SAC/TD3）。
2. **训练墙钟：** 单卡 RTX 4090，**~15 min** 端到端训出 **全关节 locomotion**（含动力学随机、粗糙地形、推扰、action-rate 课程）。
3. **平台：** **G1 / T1** 行走、侧走、转向；推扰鲁棒；**WBT** 演示含 box lifting、dancing、push 等。
4. **与 Holosoma 关系：** 项目页是 **论文 + 真机视频** 叙事入口；**可复现训练/部署** 以 Holosoma 三子包为准（`holosoma` / `holosoma_inference` / `holosoma_retargeting`）。

## 对 wiki 的映射

- [paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md) — 论文实体（本次深化）
- [holosoma（实体）](../../wiki/entities/holosoma.md) — 官方开源框架
- [FlashSAC（方法页）](../../wiki/methods/flashsac.md) — 后继 scaling 式 off-policy
- [sources/repos/holosoma.md](../repos/holosoma.md) — 仓库归档
- [sources/papers/humanoid_pnb_learning-sim-to-real-humanoid-locomotion-in-15-m.md](../papers/humanoid_pnb_learning-sim-to-real-humanoid-locomotion-in-15-m.md) — Paper Notebooks 溯源

## 参考来源（原始）

- 项目页：<https://younggyo.me/fastsac-humanoid/>
- 论文：<https://arxiv.org/abs/2512.01996>
- 代码：<https://github.com/amazon-far/holosoma>
