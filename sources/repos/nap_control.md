# NaP-Control（chiawenchen/NaP）

> 来源归档

- **标题：** NaP-Control
- **类型：** repo
- **来源：** ETH Zurich（Chia-Wen Chen, Yan Wu, Korrawe Karunratanakul, Siyu Tang）
- **链接：** <https://github.com/chiawenchen/NaP>
- **项目页：** <https://chiawenchen.github.io/nap-control-project/>
- **论文：** <https://arxiv.org/abs/2605.20209>
- **入库日期：** 2026-08-24
- **一句话说明：** Isaac Gym 上预训练扩散动作先验 + PPO 噪声导航策略的训测代码，覆盖多任务与崎岖地形扩展。
- **开源状态：** **已开源**（截至 **2026-08-24**）；子模块含 `UniPhys`；checkpoint 经 `download_checkpoints.sh` 获取；SMPL 与 Isaac Gym Preview 4 需自备。
- **沉淀到 wiki：** [`paper-nap-control.md`](../../wiki/entities/paper-nap-control.md)

## 仓库概况（2026-08-24）

| 字段 | 值 |
|------|-----|
| 仿真 | Isaac Gym Preview 4；30 Hz；SMPL-like 24 关节角色 |
| 依赖 | Python 3.8；PyTorch 2.3.1；`poselib`；`UniPhys` 子模块 |
| 先验 | 因果 Transformer 扩散模型（训练范式沿用 UniPhys） |
| RL | `rl_games` + PPO；冻结扩散先验与 PULSE decoder |
| 任务脚本 | `nap/scripts/*_{train,test,eval}.sh`（far_goal / agile_goal / velocity / sit / traj / terrain_*） |
| 评测 | `nap/evaluation/run_evaluate.py`；motion `.pkl` → 指标 |
| 集群 | `train_singularity_velocity.sh` / `submit_velocity.sh`（Singularity + SLURM） |

## 复现入口（README）

```bash
git clone -b main --recursive git@github.com:chiawenchen/NaP.git
conda create python=3.8 -n nap && conda activate nap
# PyTorch + Isaac Gym + pip install -r requirements.txt + poselib
bash download_data.sh
# 放置 SMPL 于 assets/data/smpl/
sh nap/scripts/far_goal_train.sh   # 示例：远目标训练
sh nap/scripts/far_goal_test.sh    # 示例：测试
```

## 对 wiki 的映射

- [NaP-Control 论文实体](../../wiki/entities/paper-nap-control.md)
- [NaP-Control 项目页](../sites/nap-control-project.md)
- [论文 source](../papers/nap_control_arxiv_2605_20209.md)
- 方法基线：[UniPhys 实体](../../wiki/entities/paper-bfm-40-uniphys.md)
