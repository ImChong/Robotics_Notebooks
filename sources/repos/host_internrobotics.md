# HoST（InternRobotics / OpenRobotLab）

> 来源归档

- **标题：** HoST: Humanoid Standing-up Control
- **类型：** repo
- **主仓库：** <https://github.com/InternRobotics/HoST>
- **历史/镜像克隆 URL（README）：** <https://github.com/OpenRobotLab/HoST.git>
- **论文：** <https://arxiv.org/abs/2502.08378>
- **项目页：** <https://taohuang13.github.io/humanoid-standingup.github.io/>
- **许可：** MIT
- **入库日期：** 2026-06-05
- **一句话说明：** RSS 2025 论文官方 PyTorch 实现：Isaac Gym + legged_gym + rsl_rl PPO，提供 G1 四地形起身训练/评测/可视化与俯卧姿态训练脚本，含真机部署说明。

## 技术栈（README 摘要）

| 组件 | 说明 |
|------|------|
| 仿真 | NVIDIA **Isaac Gym** |
| 训练框架 | **legged_gym** + **rsl_rl**（PPO） |
| 深度学习 | PyTorch 1.10 + CUDA 11.3（`conda_env.yml`） |
| 默认机器人 | **Unitree G1**（`g1_ground` / `g1_platform` / `g1_wall` / `g1_slope` 等 task） |
| 扩展 | Unitree **H1** 训练代码；**High Torque Mini Pi**、**DroidUp** 支持（部分 code coming soon） |

## 主要命令（摘录）

```bash
# 训练（terrain ∈ ground, platform, slope, wall）
python legged_gym/scripts/train.py --task g1_${terrain} --run_name test_g1

# 回放 checkpoint
python legged_gym/scripts/play.py --task g1_${terrain} --checkpoint_path /path/to/ckpt.pt

# 评测：成功率、脚程、平滑度、能耗（评测期亦开域随机化）
python legged_gym/scripts/eval/eval_${terrain}.py --task g1_${terrain} --checkpoint_path /path/to/ckpt.pt
```

- **运动可视化：** `motion_collection.py` + `trajectory_hands_feet.py` / `trajectory_head_pelvis.py`（UMAP / 3D 关键帧轨迹，对应论文 Fig.4）。
- **俯卧训练：** 独立 README 段（left-side / prone / right-side）；TODO 含全地形联合训练、仰卧+俯卧联合。

## 对 wiki 的映射

- 沉淀实体页：[wiki/entities/paper-host-humanoid-standingup.md](../../wiki/entities/paper-host-humanoid-standingup.md)
- 论文 source：[host_humanoid_standingup_arxiv_2502_08378.md](../papers/host_humanoid_standingup_arxiv_2502_08378.md)
- 项目页 source：[host-humanoid-standingup-project.md](../sites/host-humanoid-standingup-project.md)
