# Causal Diffusion Planner（匿名代码仓库）

> 来源归档

- **标题：** Causal_Diffusion_Planner
- **类型：** repo（Anonymous GitHub / 4open.science 镜像）
- **代码：** https://anonymous.4open.science/r/Causal_Diffusion_Planner-0181
- **论文：** <https://arxiv.org/abs/2610.12684>
- **核查日期：** 2026-10-12（通过 4open.science 文件接口读取 README 与目录）
- **许可：** 含 LICENSE 文件（类型未核对）
- **说明：** README 称论文 “currently under review for ICRA 2027”。仓库包含 TSP 分配层、扩散轨迹规划器、移动机器人推物环境与基线。
- **沉淀到 wiki：** [`wiki/entities/paper-causal-diffusion-nonprehensile-multi-object-placement.md`](../../wiki/entities/paper-causal-diffusion-nonprehensile-multi-object-placement.md)

## 目录（顶层）

`baselines/`（CompDiffuser、Diffusion Policy、SAM；README 注明 CompDiff 与 SAM 因在该任务上不成功已从主结果表中移除）、`envs_common/`、`envs_test/`、`envs_training/`、`pusht/`、`skills/`、`trajectory_planner/`、`tsp_planner/`、`requirements.txt`、`setup.py`。

## 关键文件

- `trajectory_planner/training_two_phase_diffusion.py`：因果（两阶段）扩散训练；`multi_gpu_training.sh` 为 SLURM DDP 脚本。
- `trajectory_planner/interleave_diffusion.py`、`interleave_MC_guide.py`：交错（即因果分解）采样与蒙特卡洛引导；`joint_diffusion.py`、`joint_guid_diffusion.py`：联合基线；`energy_functions.py`：引导代价；`cem.py`：CEM 基线。
- `tsp_planner/`：基于 LKH 的箱子—目标分配与排序（`solve_tsp.py`、`planner.py`）。
- `envs_training/training_offline_methods/collector.py`：遥操作采集（方向键 + Q/E），存为 Zarr。
- `envs_test/*_env.py`：训练分布、U-Shape、Pallet Staging、Cluster Delivery 等评测环境；`envs_test/eval_scripts/*.sh`：各方法评测脚本（README 注明 interleave = Causal Factorization）。

## 复现命令（README）

```bash
python tsp_planner/test_planner.py
python trajectory_planner/training_two_phase_diffusion.py --zarr_path /path/to/replay.zarr --log_dir runs/ --ckpt_dir checkpoints/
python envs_test/training_dist_1_env.py --model_type interleave --ckpt_path /path/to/checkpoint.pt --n_episodes 50 --goal_tolerance 0.04 --seed 42 --assignment greedy --eval_out_dir eval_results/
```

依赖 Python 3.10+、PyTorch、SPFA 最短路模块（IvanIZ/spfa）与 LKH 求解器。模型权重和机器人演示视频通过 Dropbox 链接提供（权重仅含在这些任务上成功的模型；链接未下载核对）。
